import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve
from lyra_lite.analysis.ece import expected_calibration_error


def sensitivity_at_fpr(y_true, y_prob, target_fpr=0.001):
    # sensitivity at a fixed false-positive-rate budget
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]
    return 0.0 if len(valid) == 0 else float(tpr[valid[-1]])


def _predict_probs(model, X_np, transform_fn, device, batch_size=1024):
    """Predict blast probabilities in mini-batches."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            logits = model(transform_fn(X_np[i:i + batch_size]))
            if logits.dim() > 1 and logits.shape[1] > 1:
                out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            else:
                out.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(out) if out else np.array([])


def create_mrd_spike_in_pools(y, groups, target_pi, pool_size,
                              n_bootstraps=5, background="shared", seed=42):
    """Yield spike-in pools as index arrays: blasts per patient, normals from a shared/per-patient pool."""
    rng = np.random.default_rng(seed)

    n_blasts = max(1, int(pool_size * target_pi))
    n_normals = pool_size - n_blasts
    global_normal_idx = np.where(y == 0)[0]

    for patient in np.unique(groups):
        blast_idx = np.where((groups == patient) & (y == 1))[0]
        if len(blast_idx) == 0:
            continue

        if background == "per_patient":
            normal_src = np.where((groups == patient) & (y == 0))[0]
            if len(normal_src) == 0:
                normal_src = global_normal_idx
        else:
            normal_src = global_normal_idx

        for b in range(n_bootstraps):
            sn = rng.choice(normal_src, size=n_normals, replace=True)
            sb = rng.choice(blast_idx, size=n_blasts, replace=True)
            idx = np.concatenate([sn, sb]); rng.shuffle(idx)

            yield {"idx": idx, "patient": patient, "bootstrap_id": b, "pi": target_pi,
                   "unique_normals": len(np.unique(sn)), "unique_blasts": len(np.unique(sb))}


def evaluate_mrd_checkpoint(model, X_test, y_test, groups_test, transform_fn=None, device="cpu",
                            dilution_series=(0.1, 0.01, 0.001), n_bootstraps=5,
                            min_blasts=10, min_pool_size=5000, target_fpr=0.001,
                            ece_n_bins=10, background="shared", seed=42):
    model.eval(); model.to(device)
    if transform_fn is None:
        transform_fn = lambda x: torch.tensor(x, dtype=torch.float32).to(device)

    # predict every test cell once, then resample by index
    probs_all = _predict_probs(model, X_test, transform_fn, device)

    print(f"{'Pi (%)':<8} | {'Base-PR':<9} | {'AUPRC':<9} | {'Sens@FPR':<9} | {'ECE':<8} | {'uNorm':<7}")
    print("-" * 62)
    results = []

    for i, pi in enumerate(dilution_series):
        pool_size = max(min_pool_size, int(np.ceil(min_blasts / pi)))
        a, s, e, un = [], [], [], []
        for pool in create_mrd_spike_in_pools(y_test, groups_test, target_pi=pi, pool_size=pool_size,
                                              n_bootstraps=n_bootstraps, background=background, seed=seed + i):
            idx = pool["idx"]
            yp, pp = y_test[idx], probs_all[idx]
            a.append(average_precision_score(yp, pp))
            s.append(sensitivity_at_fpr(yp, pp, target_fpr))
            e.append(expected_calibration_error(yp, pp, n_bins=ece_n_bins))
            un.append(pool["unique_normals"])

        print(f"{pi*100:<7.2f}% | {pi:<9.5f} | {np.mean(a):<9.4f} | {np.mean(s):<9.4f} | {np.mean(e):<8.4f} | {int(np.mean(un)):<7}")

        results.append({"pi": pi, "pool_size": pool_size,
                        "auprc_mean": np.mean(a), "auprc_std": np.std(a),
                        "sens_at_fpr_mean": np.mean(s), "sens_at_fpr_std": np.std(s),
                        "ece_mean": np.mean(e), "ece_std": np.std(e),
                        "mean_unique_normals": np.mean(un)})

    return pd.DataFrame(results)


def evaluate_specificity_controls(model, X_test, y_test, groups_test, transform_fn=None,
                                  device="cpu", thresholds=(0.5, 0.99)):
    """False-alarm rate on blast-free held-out patients."""
    model.eval(); model.to(device)
    if transform_fn is None:
        transform_fn = lambda x: torch.tensor(x, dtype=torch.float32).to(device)

    probs_all = _predict_probs(model, X_test, transform_fn, device)

    rows = []
    for patient in np.unique(groups_test):
        idx = np.where(groups_test == patient)[0]

        if np.any(y_test[idx] == 1):
            continue

        probs = probs_all[idx]
        row = {"patient": patient, "n_cells": len(probs)}

        for t in thresholds:
            row[f"fpr@{t}"] = float(np.mean(probs > t))
        rows.append(row)

    if not rows:
        print("No blast-free patients in this split (no specificity controls).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
