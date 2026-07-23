from lyra_lite.analysis.ece import expected_calibration_error
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(model, dataloader, criterion, device, n_bins):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_score = []
    y_true = []

    for x, y in tqdm(dataloader, desc="Evaluatiion", leave=False, dynamic_ncols=True):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            # sigmoid -> probabilities, then to numpy for the metrics
            logits = torch.sigmoid(logits).squeeze(-1)
            logits = logits.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            y_score.append(logits)
            y_true.append(y)

    y_score = np.concatenate(y_score)
    y_true = np.concatenate(y_true)

    auroc = roc_auc_score(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score, n_bins)
    auprc = average_precision_score(y_true, y_score)

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "ece": float(ece)
    }

    return total_loss / total_examples, metrics
