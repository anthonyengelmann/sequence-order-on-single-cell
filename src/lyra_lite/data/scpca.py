from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# in-memory cache
_MEMO: dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

# fixed seed for the HVG subsample, so the gene set is reproducible
_HVG_FIT_SEED = 0


_SPIN = "|/-\\"


def _bar(done: int, total: int, width: int = 26) -> str:
    filled = int(width * done / max(total, 1))
    return "#" * filled + "." * (width - filled)


def _progress(done: int, total: int, label: str) -> None:
    spin = _SPIN[done % len(_SPIN)]
    sys.stdout.write(f"\r  {spin} {label:<18} [{_bar(done, total)}] {done:>3}/{total}")
    sys.stdout.flush()


def _stage(msg: str) -> float:
    sys.stdout.write(f"\r  .. {msg:<58}")
    sys.stdout.flush()
    return time.time()


def _done(t0: float, msg: str) -> None:
    sys.stdout.write(f"\r  OK {msg:<50} {time.time() - t0:6.1f}s\n")
    sys.stdout.flush()


def _banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n  {title}\n{line}")


def _artifact_tag(n_hvgs: int, max_cells_per_sample: int | None, seed: int) -> str:
    if max_cells_per_sample is None:
        return f"{n_hvgs}hvg_full"
    return f"{n_hvgs}hvg_max{max_cells_per_sample}_seed{seed}"


def _write_manifest(path: Path, tag: str, params: dict,
                    genes: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_source_files: int, build_seconds: float) -> None:
    """Write a small sidecar json describing the cached artifact."""
    from importlib.metadata import PackageNotFoundError, version
    versions = {"numpy": np.__version__}
    for pkg in ("scanpy", "anndata"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    manifest = {
        "artifact": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(build_seconds, 1),
        "params": params,
        "stats": {
            "n_cells": int(len(y)),
            "n_genes": int(len(genes)),
            "n_patients": int(len(set(groups.tolist()))),
            "blast_prevalence": round(float(y.mean()), 4),
            "n_source_files": int(n_source_files),
        },
        "patients": sorted(set(groups.tolist())),
        "versions": versions,
    }
    path.write_text(json.dumps(manifest, indent=2))


def _summary(prefix: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, secs: float) -> None:
    print(f"  {prefix}  X={tuple(X.shape)}  blast={float(y.mean()):.1%}  "
          f"patients={len(set(groups.tolist()))}  ({secs:.2f}s)")


def _build_scpca_dataset(
    data_dir: str, file_glob: str, label_col: str, positive_label: str,
    exclude_labels: List[str], group_col: str, n_hvgs: int, hvg_flavor: str,
    max_cells_per_sample: int | None, seed: int, hvg_fit_cells_per_sample: int, tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # import lazily so the cache-hit path needs neither installed
    import anndata as ad
    import scanpy as sc

    _banner(f"LYRA-Lite | building ScPCA artifact: {tag}")
    files = sorted(Path(data_dir).rglob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matching {file_glob!r} under {data_dir}")

    def _load_kept(path):
        """Read one sample, keep logcounts, drop excluded cells."""
        a = sc.read_h5ad(path)
        if label_col not in a.obs:
            return None
        # grab the logcounts matrix
        mat = a.X if a.X is not None else (a.layers["logcounts"] if "logcounts" in a.layers else None)
        if mat is None:
            from importlib.metadata import version
            raise ValueError(
                f"{path.name}: read returned X=None (anndata {version('anndata')}). "
                f"The logcounts exist on disk but your reader didn't load them -- try "
                f"`pip install -U 'anndata==0.11.4' 'scanpy==1.11.5'` and re-run."
            )
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32)
        a.X = mat
        # drop unused layers/matrices to save memory
        for slot in (a.layers, a.obsm, a.varm, a.obsp, a.uns):
            slot.clear()
        a.raw = None
        return a[~a.obs[label_col].isin(exclude_labels)].copy()

    # Pass 1: pick cohort HVGs from a subsample of each sample
    hvg_slices = []
    for i, f in enumerate(files, 1):
        _progress(i, len(files), "scanning HVGs")
        a = _load_kept(f)
        if a is None:
            continue
        if a.n_obs > hvg_fit_cells_per_sample:
            sc.pp.subsample(a, n_obs=hvg_fit_cells_per_sample, random_state=_HVG_FIT_SEED)
        hvg_slices.append(a)
    sys.stdout.write("\n")
    if not hvg_slices:
        raise RuntimeError(f"No samples contained the column {label_col!r}.")

    fit_cells = sum(s.n_obs for s in hvg_slices)
    t = _stage(f"selecting top {n_hvgs} HVGs from a {fit_cells:,}-cell subsample")
    hvg_ad = ad.concat(hvg_slices, join="inner")
    del hvg_slices
    sc.pp.highly_variable_genes(hvg_ad, n_top_genes=n_hvgs, flavor=hvg_flavor)
    hvg_genes = hvg_ad.var_names[hvg_ad.var["highly_variable"]].to_numpy().astype(str)
    del hvg_ad
    _done(t, f"HVGs selected ({len(hvg_genes)} genes)")

    # Pass 2: stream every cell, keep only the HVG columns
    X_parts, ys, groups_list = [], [], []
    for i, f in enumerate(files, 1):
        _progress(i, len(files), "building matrix")
        a = _load_kept(f)
        if a is None:
            continue
        if max_cells_per_sample is not None and a.n_obs > max_cells_per_sample:
            sc.pp.subsample(a, n_obs=max_cells_per_sample, random_state=seed)
        ys.append((a.obs[label_col].astype(str) == positive_label).astype(np.float32).values)
        groups_list.append(a.obs[group_col].astype(str).values)
        sub = a[:, hvg_genes].X
        sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        X_parts.append(sub.astype(np.float32))
        del a, sub
    sys.stdout.write("\n")

    t = _stage("stacking HVG matrix")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(ys).astype(np.float32)
    groups = np.concatenate(groups_list).astype(str)
    del X_parts, ys, groups_list
    _done(t, "stacked")

    assert X.shape[0] == len(y) == len(groups), "X/y/groups length mismatch!"
    assert X.shape[1] == len(hvg_genes), "gene-count mismatch!"
    return X, y, groups, hvg_genes


def make_scpca_dataset(
    data_dir: str,
    file_glob: str,
    label_col: str,
    positive_label: str,
    exclude_labels: List[str],
    group_col: str,
    n_hvgs: int,
    hvg_flavor: str,
    max_cells_per_sample: int | None,
    seed: int,
    hvg_fit_cells_per_sample: int = 300,
    cache_dir: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X float32 [n_cells, n_hvgs], y float32 [n_cells], groups str [n_cells])."""
    t0 = time.time()
    cdir = Path(cache_dir) if cache_dir else Path(data_dir) / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    tag = _artifact_tag(n_hvgs, max_cells_per_sample, seed)
    cache_path = cdir / f"cached_scpca_{tag}.npz"
    manifest_path = cdir / f"cached_scpca_{tag}.json"
    key = str(cache_path.resolve())

    # tier 1: in-memory memo
    if key in _MEMO:
        X, y, groups = _MEMO[key]
        _summary(f"[memo hit] {tag}", X, y, groups, time.time() - t0)
        return X, y, groups

    # tier 2: on-disk cache
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=False)
        X, y, groups = blob["X"], blob["y"], blob["groups"]
        _MEMO[key] = (X, y, groups)
        _summary(f"[cache hit] {cache_path.name}", X, y, groups, time.time() - t0)
        return X, y, groups

    # tier 3: build once, then cache
    X, y, groups, genes = _build_scpca_dataset(
        data_dir, file_glob, label_col, positive_label, exclude_labels,
        group_col, n_hvgs, hvg_flavor, max_cells_per_sample, seed,
        hvg_fit_cells_per_sample, tag,
    )

    t = _stage(f"caching artifact -> {cache_path.name}")
    np.savez_compressed(cache_path, X=X, y=y, groups=groups, genes=genes)
    _write_manifest(
        manifest_path, tag,
        params={
            "n_hvgs": n_hvgs, "hvg_flavor": hvg_flavor, "label_col": label_col,
            "positive_label": positive_label, "exclude_labels": list(exclude_labels),
            "group_col": group_col, "file_glob": file_glob,
            "max_cells_per_sample": max_cells_per_sample, "seed": seed,
            "hvg_fit_cells_per_sample": hvg_fit_cells_per_sample,
            "regime": "full" if max_cells_per_sample is None else "subsampled",
        },
        genes=genes, y=y, groups=groups, n_source_files=0, build_seconds=time.time() - t0,
    )
    _done(t, "cached")

    _MEMO[key] = (X, y, groups)
    _summary(f"built {tag}", X, y, groups, time.time() - t0)
    return X, y, groups
