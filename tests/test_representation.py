import numpy as np

from lyra_lite.data.representation import encode_cells


def test_output_shapes_and_dtype():
    X = np.random.default_rng(0).random((20, 100))
    tokens, values = encode_cells(X, top_k=32, ordering="rank", pad_id=100)
    assert tokens.shape == (20, 32)
    assert values.shape == (20, 32)
    assert tokens.dtype == np.int64


def test_rank_ordering_is_descending():
    # one cell, 6 "genes"; rank should sort by expression, highest first
    X = np.array([[0.0, 5.0, 0.0, 9.0, 2.0, 0.0]])
    tokens, values = encode_cells(X, top_k=4, ordering="rank", pad_id=6)
    assert list(values[0]) == [9.0, 5.0, 2.0, 0.0]
    assert tokens[0, 0] == 3   # gene index of the value 9
    assert tokens[0, 3] == 6   # padding (only 3 genes are expressed)


def test_padding_for_sparse_cells():
    # cell with 2 expressed genes, top_k=5 -> 3 padding slots
    X = np.array([[0.0, 3.0, 0.0, 1.0, 0.0]])
    tokens, values = encode_cells(X, top_k=5, ordering="rank", pad_id=5)
    assert int((tokens[0] == 5).sum()) == 3
    assert int((values[0] == 0.0).sum()) == 3


def test_random_ordering_keeps_the_same_token_set():
    X = np.random.default_rng(1).random((5, 50))
    t_rank, _ = encode_cells(X, top_k=20, ordering="rank", pad_id=50, seed=0)
    t_rand, _ = encode_cells(X, top_k=20, ordering="random", pad_id=50, seed=0)
    for i in range(5):
        assert set(t_rank[i]) == set(t_rand[i])
