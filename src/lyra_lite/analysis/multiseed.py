import glob
import json
import os
import numpy as np


def bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean()
                  for _ in range(n_boot)]
    return np.percentile(boot_means, [2.5, 97.5])


def aggregate_sweep(run_dir):
    """Read every metrics.json under run_dir, print and return summary stats."""
    paths = sorted(glob.glob(os.path.join(run_dir, "*", "metrics.json")))
    assert paths, f"No metrics.json found under {run_dir}"
    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))

    # auroc (test / val / legacy)
    if "test_auroc" in records[0]:
        aurocs = np.array([r["test_auroc"] for r in records])
        metric_suffix = "test"
    elif "val_auroc" in records[0]:
        aurocs = np.array([r["val_auroc"] for r in records])
        metric_suffix = "val"
    else:
        aurocs = np.array([r["auroc"] for r in records])
        metric_suffix = "legacy"

    # ece
    if "test_ece" in records[0]:
        eces = np.array([r["test_ece"] for r in records])
    elif "val_ece" in records[0]:
        eces = np.array([r["val_ece"] for r in records])
    else:
        eces = np.array([r["ece"] for r in records])

    seeds = [r["seed"] for r in records]
    mean = float(aurocs.mean())
    std = float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0
    lo, hi = bootstrap_ci(aurocs)
    model = records[0]["model"]

    print(f"{model} | n={len(aurocs)} seeds {seeds} ({metric_suffix} metrics)")
    print(f"AUROC: {mean:.4f} ± {std:.4f} | 95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"ECE: {eces.mean():.4f} ± {(eces.std(ddof=1) if len(eces) > 1 else 0.0):.4f}")

    summary = {
        "model": model,
        "metric_type": metric_suffix,
        "mean_auroc": mean,
        "std_auroc": std,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "mean_ece": float(eces.mean()),
        "std_ece": float(eces.std(ddof=1)) if len(eces) > 1 else 0.0,
        "n": len(aurocs),
        "seeds": seeds
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def aggregate_mrd_sweep(run_dir):
    """Aggregate per-seed rare-class results (mrd_lod.csv) across a sweep, by prevalence pi."""
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(run_dir, "**", "mrd_lod.csv"), recursive=True))
    assert paths, f"No mrd_lod.csv found under {run_dir}"
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    metrics = {"auprc": "auprc_mean", "sensitivity": "sens_at_fpr_mean", "ece": "ece_mean"}
    metrics = {name: col for name, col in metrics.items() if col in df.columns}

    summary = []
    for pi, g in df.groupby("pi"):
        row = {"pi": float(pi), "n_seeds": int(len(g))}
        for name, col in metrics.items():
            row[f"{name}_mean"] = float(g[col].mean())
            row[f"{name}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        summary.append(row)
    summary.sort(key=lambda r: -r["pi"])

    with open(os.path.join(run_dir, "mrd_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Aggregated {len(paths)} run(s) over {df['pi'].nunique()} dilution level(s) -> mrd_summary.json")
    for r in summary:
        cells = " | ".join(f"{name.upper()} {r[f'{name}_mean']:.3f}±{r[f'{name}_std']:.3f}" for name in metrics)
        print(f"  pi={r['pi']:<8} n={r['n_seeds']} | {cells}")
    return summary
