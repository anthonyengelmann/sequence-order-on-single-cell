from pathlib import Path

import numpy as np
import pandas as pd

# canonical marker genes for the sanity-check panel
MARKERS = ["CD19", "CD34", "DNTT", "CD79A", "CD3D", "NKG7", "LYZ", "CD14"]


def blast_prevalence_per_patient(y, groups):
    """Per-patient blast fraction (%) and its median."""
    pct = pd.Series(y, index=groups).groupby(level=0).mean() * 100
    return pct, float(pct.median())


def patient_shift_embedding(X, y, groups, n_sub=30_000, seed=0):
    """UMAP embedding + a 'which patient?' probe (distribution-shift check)."""
    import scanpy as sc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
    Xs, ys, gs = X[idx], y[idx], groups[idx]

    ad = sc.AnnData(Xs)
    sc.pp.pca(ad, n_comps=50)
    sc.pp.neighbors(ad, n_neighbors=15)
    sc.tl.umap(ad)
    emb, pcs = ad.obsm["X_umap"], ad.obsm["X_pca"]

    Xtr, Xte, gtr, gte = train_test_split(pcs, gs, test_size=0.3, stratify=gs, random_state=seed)
    bal_acc = balanced_accuracy_score(gte, LogisticRegression(max_iter=1000).fit(Xtr, gtr).predict(Xte))
    chance = 1 / len(np.unique(gs))
    return dict(emb=emb, ys=ys, gs=gs, bal_acc=bal_acc, chance=chance)


def sparsity_stats(X, top_k=256, n_sub=8000, seed=0):
    """Genes-per-cell and cumulative expression captured by the top-k genes."""
    rng = np.random.default_rng(seed)

    genes_per_cell = (X > 0.0).sum(axis=1)
    median_gpc = int(np.median(genes_per_cell))
    pad_frac = float((genes_per_cell < top_k).mean())

    # cumulative expression captured by the top-k ranked genes (subsample)
    sub = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
    expr = np.expm1(sub)
    expr = expr[expr.sum(1) > 0]
    srt = np.sort(expr, axis=1)[:, ::-1]
    frac = np.cumsum(srt, axis=1)
    frac = frac / frac[:, -1:]
    mean_frac = frac.mean(0)
    lo, hi = np.percentile(frac, [10, 90], axis=0)
    cap_at_k = mean_frac[top_k - 1]

    return dict(genes_per_cell=genes_per_cell, median_gpc=median_gpc, pad_frac=pad_frac,
                mean_frac=mean_frac, lo=lo, hi=hi, cap_at_k=cap_at_k,
                top_k=top_k, n_ranks=srt.shape[1])


def normal_availability(y, groups, series=(0.1, 0.01, 0.001, 0.0001), min_blasts=10):
    """Per-patient normal-cell counts and unique normals needed per prevalence pi."""
    pats = np.unique(groups)
    norm_per = np.array([int(((groups == p) & (y == 0)).sum()) for p in pats])
    N_total = int((y == 0).sum())
    need = [int(np.ceil(min_blasts / pi * (1 - pi))) for pi in series]
    return dict(norm_per=norm_per, N_total=N_total, median=int(np.median(norm_per)),
                series=list(series), need=need)


def load_marker_matrix(
    data_dir="../data/SCPCP000008_ann-data/SCPCP000008_single-cell",
    cache_path="../data/cache/scpca_markers.npz",
    markers=MARKERS,
):
    """Load (or build + cache) a [N, n_markers] marker-expression matrix + blast label."""
    import anndata as ad

    cache_path = Path(cache_path)
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        return d["Xm"], d["y"], list(d["markers"])

    files = sorted(Path(data_dir).rglob("*_processed_rna.h5ad"))
    var = ad.read_h5ad(files[0], backed="r").var
    sym2ens = dict(zip(var["gene_symbol"].astype(str), var.index.astype(str)))
    names = [m for m in markers if sym2ens.get(m)]
    cols = [sym2ens[m] for m in names]

    mats, labs = [], []
    for f in files:
        a = ad.read_h5ad(f, backed="r")
        if "submitter_celltype_annotation" not in a.obs:
            continue
        keep = ~a.obs["submitter_celltype_annotation"].isin(["Submitter-excluded"])
        sub = a[keep.values, cols].to_memory().X
        mats.append(np.asarray(sub.todense()) if hasattr(sub, "todense") else np.asarray(sub))
        labs.append((a.obs["submitter_celltype_annotation"][keep].astype(str) == "Blast").values.astype(np.int8))
    Xm = np.vstack(mats).astype(np.float32)
    ym = np.concatenate(labs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, Xm=Xm, y=ym, markers=np.array(names))
    return Xm, ym, names
