import numpy as np

def encode_cells(X, top_k=256, ordering="rank", pad_id=None, seed = 0, importance=None):
    """Turn a dense expression matrix into padded (token, value) sequences."""
    if pad_id is None:
        pad_id = X.shape[1]

    N_cells, n_hvgs = X.shape

    # sort genes by expression (descending) and keep the top_k
    order = np.argsort(-X, axis=1)
    top_indices = order[:, :top_k]
    top_values = np.take_along_axis(X, top_indices, axis=1)

    # real genes are the non-zero ones, the rest is padding
    is_expressed = top_values > 0

    tokens = np.full((N_cells, top_k), pad_id, dtype=np.int64)
    values = np.zeros((N_cells, top_k), dtype=np.float32)

    if ordering == "rank":
        tokens[is_expressed] = top_indices[is_expressed]
        values[is_expressed] = top_values[is_expressed]

    else:
        # ordering ablations: reorder the real genes per cell
        for i in range(N_cells):
            n_real = is_expressed[i].sum()
            if n_real == 0:
                continue

            real_idx = top_indices[i, :n_real]
            real_val = top_values[i, :n_real]

            if ordering == "alphabetical":
                sort_mask = np.argsort(real_idx)
            elif ordering == "random":
                rng = np.random.default_rng([seed, i])
                sort_mask = rng.permutation(n_real)

            elif ordering == "ascending":
                # lowest expression first, highest last
                sort_mask = np.argsort(real_val)
            elif ordering in ("importance_first", "importance_last"):
                if importance is None:
                    raise ValueError("Must provide 'importance' array for importance ordering.")
                imp = np.abs(importance[real_idx])
                sort_mask = np.argsort(-imp) if ordering == "importance_first" else np.argsort(imp)

            else:
                raise ValueError(f"Unknown ordering: {ordering}")

            tokens[i, :n_real] = real_idx[sort_mask]
            values[i, :n_real] = real_val[sort_mask]

    return tokens, values


def compute_gene_importance(X, y, seed=0, max_cells=20000, C=1.0):
    """Rank genes by |logistic-regression coefficient| on the train split."""
    from sklearn.linear_model import LogisticRegression

    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y).ravel()

    # subsample rows to keep the probe fast
    if max_cells and Xf.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        sel = rng.choice(Xf.shape[0], size=max_cells, replace=False)
        Xf, yf = Xf[sel], yf[sel]

    clf = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    clf.fit(Xf, yf)
    return np.abs(clf.coef_[0]).astype(np.float32)
