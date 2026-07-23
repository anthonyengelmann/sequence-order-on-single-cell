import pytest

torch = pytest.importorskip("torch")   # skip the whole file if torch isn't installed


def test_fnn_forward_shape():
    from lyra_lite.models.fnn import FNNBaseline
    model = FNNBaseline(n_genes=50, hidden_dim=16, dropout=0.0)
    out = model(torch.randn(4, 50))
    assert out.shape == (4, 1)


def test_lstm_forward_shape():
    from lyra_lite.models.lstm import LSTMBaseline
    model = LSTMBaseline(vocab_size=51, embedding_dim=16, hidden_dim=16,
                         num_layers=1, dropout=0.0, pad_id=50)
    model.eval()
    ids = torch.randint(0, 50, (4, 8)).float()      # real tokens (no padding)
    vals = torch.randn(4, 8)
    out = model(torch.stack([ids, vals], dim=-1))   # [batch, top_k, 2]
    assert out.shape == (4, 1)


def test_transformer_forward_shape():
    pytest.importorskip("entmax")                   # transformer imports entmax15
    from lyra_lite.models.transformer import TransformerClassifier
    model = TransformerClassifier(vocab_size=51, pad_id=50, d_model=16, n_heads=2, num_layers=1)
    model.eval()
    ids = torch.randint(0, 50, (4, 8)).float()
    vals = torch.randn(4, 8)
    out = model(torch.stack([ids, vals], dim=-1))
    assert out.shape == (4, 1)
