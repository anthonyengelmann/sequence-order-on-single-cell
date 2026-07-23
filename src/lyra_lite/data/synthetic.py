import numpy as np


def make_synthetic_dataset(n_cells=4000 , n_genes=2000 , n_disease_genes= 50, class_balance= 0.3, seed = 42, disease_effect = 2.0, noise_scale = 1.0, n_groups = 10):
    rng = np.random.default_rng(seed)

    # random expression with a disease signal in the first genes
    X = rng.normal(0, noise_scale, (n_cells, n_genes))
    y = (rng.random(n_cells) < class_balance).astype(np.float32)
    X[y == 1, :n_disease_genes] += disease_effect

    X = np.clip(X, 0, None)

    # group cells into patients and add a small per-patient shift
    groups = rng.integers(0, n_groups, n_cells)
    for g in range(n_groups):
        group_mask = (groups == g)
        X[group_mask] += rng.normal(0, noise_scale * 0.2, (group_mask.sum(), n_genes))

    return X, y, groups
