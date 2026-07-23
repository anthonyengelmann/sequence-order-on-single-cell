import torch
import torch.nn as nn
import math
from entmax import entmax15

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CustomSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_type="dense"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_type = attention_type

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _attn_weights(self, scores, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        if self.attention_type == "dense":
            return scores.softmax(dim=-1)
        elif self.attention_type == "sparse":

            return entmax15(scores, dim=-1)
        else:
            raise ValueError(f"Unknown Attention: '{self.attention_type}'")

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self._attn_weights(scores, mask)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, d_model=128, n_heads=4, num_layers=2,
             attention="dense", positional_encoding="none", num_classes=1):
        super().__init__()
        self.pad_id = pad_id

        self.positional_encoding = positional_encoding

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.value_proj = nn.Linear(1, d_model, bias=False)

        if self.positional_encoding == "sinusoidal":
            self.pe_module = PositionalEncoding(d_model, max_len=2000)
        elif self.positional_encoding != "none":
            raise ValueError(f"Unbekanntes PE: {self.positional_encoding}")
        self.attention_layers = nn.ModuleList([
            CustomSelfAttention(d_model, n_heads, attention_type=attention)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers * 2)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            token_ids = x[..., 0].long()
            values = x[..., 1].float()
        else:
            token_ids, values = x.long(), None

        mask = (token_ids != self.pad_id)

        h = self.embedding(token_ids)
        if values is not None:
            h = h + self.value_proj(values.unsqueeze(-1))

        if self.positional_encoding == "sinusoidal":
            h = self.pe_module(h)

        for i in range(len(self.attention_layers)):
            h = self.layer_norms[i*2](h + self.attention_layers[i](h, mask))
            h = self.layer_norms[i*2+1](h + self.ffn_layers[i](h))

        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)
