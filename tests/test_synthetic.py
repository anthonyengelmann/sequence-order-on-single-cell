import numpy as np

from lyra_lite.data.synthetic import make_synthetic_dataset


def test_shapes():
    X, y, groups = make_synthetic_dataset(n_cells=200, n_genes=50, seed=0)
    assert X.shape == (200, 50)
    assert y.shape == (200,)
    assert groups.shape == (200,)


def test_labels_binary_and_roughly_balanced():
    _, y, _ = make_synthetic_dataset(n_cells=1000, class_balance=0.3, seed=0)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert 0.2 < y.mean() < 0.4


def test_disease_signal_is_present():
    X, y, _ = make_synthetic_dataset(n_cells=1000, n_genes=100, n_disease_genes=10,
                                     disease_effect=2.0, seed=0)
    assert X[y == 1, :10].mean() > X[y == 0, :10].mean()


def test_reproducible_given_seed():
    a = make_synthetic_dataset(n_cells=100, seed=42)[0]
    b = make_synthetic_dataset(n_cells=100, seed=42)[0]
    assert np.array_equal(a, b)
