from lyra_lite.data.representation import encode_cells
import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        representation="dense",
        top_k=256,
        ordering="rank",
        use_value=True,
        pad_id=None,
        importance=None):

        self.representation = representation.lower()

        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        if pad_id is None:
            pad_id = X.shape[1]
        self.pad_id = pad_id

        # dense = raw vector, tokens = gene-id/value sequences
        if self.representation == "dense":
            self.X = torch.tensor(X, dtype=torch.float32)

        elif self.representation == "tokens":
            tokens, values = encode_cells(X, top_k=top_k, ordering=ordering, pad_id=self.pad_id, importance=importance)

            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
            values_tensor = torch.tensor(values, dtype=torch.float32)

            # stack token ids and values into one tensor
            if use_value:
                tokens_tensor = tokens_tensor.float()
                self.X = torch.stack((tokens_tensor, values_tensor), dim=-1)
                pass
            else:
                self.X = tokens_tensor
                pass
        else:
            raise ValueError(f"Unbekannte Repräsentation: {self.representation}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
