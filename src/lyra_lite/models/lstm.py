import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.3,
        pad_id=None
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.pad_id = pad_id

        self.value_proj = nn.Linear(1, embedding_dim, bias=False)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 3:
            # rank + value: embed gene ids and values, then add
            ids = x[..., 0].long()
            vals = x[..., 1].float()

            id_emb = self.embedding(ids)
            val_emb = self.value_proj(vals.unsqueeze(-1))

            x_seq = id_emb + val_emb

        else:
            # pure rank: only gene ids
            ids = x.long()
            x_seq = self.embedding(ids)

        # pack so the LSTM skips padding, then take the last hidden state
        lengths = (ids != self.pad_id).sum(dim=1).clamp(min=1)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            x_seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hn, _) = self.lstm(packed_seq)

        final_hidden = hn[-1]

        drop_hidden = self.dropout(final_hidden)
        return self.fc(drop_hidden)
