"""Rare-class limit-of-detection eval for finished training runs."""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from lyra_lite.data.representation import encode_cells

from lyra_lite.analysis.mrd_eval import (
    evaluate_mrd_checkpoint,
    evaluate_specificity_controls,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _held_out_test_split(X, y, groups, seed):
    """Same 70/15/15 patient split as train.py; return only the test slice."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    _, temp_idx = next(gss1.split(X, y, groups=groups))
    Xt, yt, gt = X[temp_idx], y[temp_idx], groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    _, test_idx = next(gss2.split(Xt, yt, groups=gt))
    return Xt[test_idx], yt[test_idx], gt[test_idx]


def find_runs(root):
    """All runs under root that have both .hydra/config.yaml and best_model.pt."""
    root = Path(root).resolve()
    if (root / ".hydra" / "config.yaml").exists() and (root / "best_model.pt").exists():
        return [root]
    runs = [ckpt.parent for ckpt in root.rglob("best_model.pt")
            if (ckpt.parent / ".hydra" / "config.yaml").exists()]
    return sorted(runs)


def evaluate_one_run(run_dir, repo_root, eval_yaml, ckpt_name="best_model.pt"):
    """Run the rare-class + specificity eval for one finished run."""
    # load the run's own config snapshot
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.update(cfg, "paths.project_root", str(repo_root), force_add=True)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # rebuild data + the same held-out patient split
    X, y, groups = instantiate(cfg.data)
    X_test, y_test, groups_test = _held_out_test_split(X, y, groups, cfg.seed)
    log.info(f"  model={cfg.model._target_.split('.')[-1]} | seed={cfg.seed} | "
             f"test {X_test.shape[0]} cells, {len(np.unique(groups_test))} patients, "
             f"blast {y_test.mean():.3f}")

    # rebuild model + load weights
    model = instantiate(cfg.model)
    ckpt = run_dir / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    # build the same input transform CellDataset uses
    ev = OmegaConf.load(eval_yaml)
    rep_cfg = cfg.representation
    if rep_cfg.representation == "tokens":
        use_value = rep_cfg.get("use_value", True)
        ordering = rep_cfg.get("ordering", "rank")

        # importance ordering: re-fit the same train-split probe
        gene_importance = None
        if str(ordering).startswith("importance"):
            from lyra_lite.data.representation import compute_gene_importance
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=cfg.seed)
            train_idx, _ = next(gss.split(X, y, groups=groups))
            gene_importance = compute_gene_importance(X[train_idx], y[train_idx], seed=cfg.seed)

        def transform_fn(arr):
            tokens, values = encode_cells(
                arr,
                top_k=rep_cfg.get("top_k", 256),
                ordering=ordering,
                pad_id=rep_cfg.get("pad_id", 2000),
                importance=gene_importance,
            )
            if use_value:
                packed = np.stack([tokens.astype(np.float32), values.astype(np.float32)], axis=-1)
                return torch.tensor(packed, dtype=torch.float32).to(device)
            return torch.tensor(tokens, dtype=torch.long).to(device)
    elif rep_cfg.representation == "dense":
        transform_fn = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
    else:
        raise ValueError(f"Unknown representation: '{rep_cfg.representation}' (use 'tokens' or 'dense').")

    lod_df = evaluate_mrd_checkpoint(
        model, X_test, y_test, groups_test, transform_fn=transform_fn, device=device,
        dilution_series=list(ev.dilution_series), n_bootstraps=ev.n_bootstraps,
        min_blasts=ev.min_blasts, min_pool_size=ev.min_pool_size, target_fpr=ev.target_fpr,
        ece_n_bins=ev.ece_n_bins, background=ev.background, seed=ev.seed,
    )
    spec_df = evaluate_specificity_controls(
        model, X_test, y_test, groups_test, transform_fn=transform_fn, device=device,
        thresholds=list(ev.specificity_thresholds),
    )

    # save results into the run dir
    out = run_dir / "mrd"
    out.mkdir(exist_ok=True)
    lod_df.to_csv(out / "mrd_lod.csv", index=False)
    if not spec_df.empty:
        spec_df.to_csv(out / "mrd_specificity.csv", index=False)
    OmegaConf.save(ev, out / "eval_config.yaml")
    log.info(f"  saved -> {out / 'mrd_lod.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="a single run dir, OR a parent dir (timestamp/day) to batch every nested run")
    p.add_argument("--eval_config", default=None,
                   help="eval yaml (default: <repo>/configs/eval/mrd.yaml)")
    p.add_argument("--ckpt_name", default="best_model.pt")
    args = p.parse_args()

    root = Path(args.run_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    eval_yaml = Path(args.eval_config) if args.eval_config else repo_root / "configs" / "eval" / "mrd.yaml"

    runs = find_runs(root)
    if not runs:
        log.error(f"No valid runs (need .hydra/config.yaml + {args.ckpt_name}) under {root}")
        return

    log.info(f"Found {len(runs)} run(s) under {root}\n")
    ok = failed = 0
    for i, run in enumerate(runs, 1):
        label = run.name if run == root else run.relative_to(root)
        log.info(f"[{i}/{len(runs)}] {label}")
        try:
            evaluate_one_run(run, repo_root, eval_yaml, args.ckpt_name)
            ok += 1
        except Exception as e:
            log.warning(f"  SKIPPED: {type(e).__name__}: {e}")
            failed += 1
    log.info(f"\nDone. {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
