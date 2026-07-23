#!/usr/bin/env python
"""Paired within-seed comparison of the FNN / LSTM / Transformer ladder."""
import argparse, glob, json, os
import numpy as np
import pandas as pd

MODEL_KEY = {"fnn": "FNNBaseline", "lstm": "LSTMBaseline", "transformer": "TransformerClassifier"}
# (metric column, higher_is_better)
BAL_METRICS = [("test_auroc", True), ("test_ece", False)]
RARE_METRICS = [("auprc", True), ("sens", True), ("ece", False)]


def _latest_per_seed(sweep_dir, want_model):
    """seed -> (run_dir, metrics) keeping the newest matching run."""
    best = {}
    for mj in glob.glob(os.path.join(sweep_dir, "**", "metrics.json"), recursive=True):
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        if want_model and want_model not in str(m.get("model", "")):
            continue
        seed = m.get("seed")
        if seed is None:
            continue
        t = os.path.getmtime(mj)
        if seed not in best or t > best[seed][0]:
            best[seed] = (t, os.path.dirname(mj), m)
    return {s: (d, m) for s, (t, d, m) in best.items()}


def collect(sweep_dirs):
    """Read balanced (metrics.json) + rare-class (mrd_lod.csv) per (model, seed)."""
    bal, rare = [], []
    for model, sd in sweep_dirs.items():
        for seed, (run_dir, m) in _latest_per_seed(sd, MODEL_KEY.get(model, "")).items():
            bal.append({"seed": seed, "model": model,
                        "test_auroc": m.get("test_auroc"), "test_ece": m.get("test_ece")})
            hits = glob.glob(os.path.join(run_dir, "**", "mrd_lod.csv"), recursive=True)
            if hits:
                d = pd.read_csv(hits[0])
                for _, r in d.iterrows():
                    rare.append({"seed": seed, "model": model, "pi": float(r["pi"]),
                                 "auprc": float(r["auprc_mean"]), "sens": float(r["sens_at_fpr_mean"]),
                                 "ece": float(r["ece_mean"])})
    return pd.DataFrame(bal), pd.DataFrame(rare)


def paired(df, metric, higher_better, group_cols=(), ref="fnn", others=("lstm", "transformer")):
    """Paired delta (ref - other) within seed."""
    rows = []
    keys = ["seed"] + list(group_cols)
    if df is None or getattr(df, "empty", True) or "model" not in df.columns or metric not in df.columns:
        return pd.DataFrame(columns=["contrast", "metric", "n", "mean_delta", "se", "std", "ref_better", "winner", *group_cols])
    for other in others:
        a = df[df.model == ref][keys + [metric]].rename(columns={metric: "ref"})
        b = df[df.model == other][keys + [metric]].rename(columns={metric: "oth"})
        mrg = a.merge(b, on=keys).dropna()
        groups = mrg.groupby(list(group_cols)) if group_cols else [((), mrg)]
        for gval, g in groups:
            delta = (g["ref"] - g["oth"]).to_numpy()
            n = len(delta)
            if n == 0:
                continue
            mean = float(delta.mean())
            se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            sd = float(delta.std(ddof=1)) if n > 1 else float("nan")
            k_better = int((delta > 0).sum() if higher_better else (delta < 0).sum())
            row = {"contrast": f"{ref.upper()}-{other.upper()}", "metric": metric, "n": n,
                   "mean_delta": mean, "se": se, "std": sd, "ref_better": f"{k_better}/{n}",
                   "winner": (ref.upper() if (mean > 0) == higher_better else other.upper())}
            if group_cols:
                for c, v in zip(group_cols, (gval if isinstance(gval, tuple) else (gval,))):
                    row[c] = v
            rows.append(row)
    return pd.DataFrame(rows)


def all_pairwise(df, metrics, group_cols=()):
    """Every model pair: FNN-LSTM, FNN-Transformer, LSTM-Transformer, per metric."""
    fr = []
    for m, hb in metrics:
        fr.append(paired(df, m, hb, group_cols=group_cols, ref="fnn", others=("lstm", "transformer")))
        fr.append(paired(df, m, hb, group_cols=group_cols, ref="lstm", others=("transformer",)))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def per_seed_figure(bal, out_path):
    """Per-seed lines across the 3 models (balanced AUROC)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return
    order = ["fnn", "lstm", "transformer"]
    piv = bal.pivot_table(index="seed", columns="model", values="test_auroc")
    piv = piv[[c for c in order if c in piv.columns]].dropna()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    x = range(len(piv.columns))
    for seed, r in piv.iterrows():
        ax.plot(list(x), r.values, "-o", lw=0.8, ms=3, alpha=0.7)
    ax.set_xticks(list(x)); ax.set_xticklabels([c.upper() for c in piv.columns])
    ax.set_ylabel("Balanced test AUROC"); ax.set_xlabel("model")
    ax.set_title(f"Same seed moves together (n={len(piv)} seeds)", fontsize=8, fontweight="bold")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path + ".pdf", bbox_inches="tight"); fig.savefig(out_path + ".png", dpi=200, bbox_inches="tight")
    print(f"figure -> {out_path}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnn", default="outputs/report/ladder_fnn")
    ap.add_argument("--lstm", default="outputs/report/ladder_lstm")
    ap.add_argument("--transformer", default="outputs/report/ladder_transformer")
    ap.add_argument("--root", default=None, help="if set, use this one dir for all models (model-filtered)")
    ap.add_argument("--out", default="outputs/report/paired")
    args = ap.parse_args()

    dirs = {"fnn": args.root or args.fnn, "lstm": args.root or args.lstm,
            "transformer": args.root or args.transformer}
    bal, rare = collect(dirs)
    if bal.empty:
        print("No metrics.json found under", dirs); return

    shared = sorted(set.intersection(*[set(bal[bal.model == m].seed) for m in ["fnn", "lstm", "transformer"] if m in set(bal.model)]))
    print(f"Shared seeds across all 3 models: {shared}  (n={len(shared)})\n")
    bal = bal[bal.seed.isin(shared)]; rare = rare[rare.seed.isin(shared)] if not rare.empty else rare

    print("=" * 72, "\nBALANCED TASK — paired within-seed (all model pairs)\n" + "=" * 72)
    bal_out = all_pairwise(bal, BAL_METRICS)
    print(bal_out.to_string(index=False))

    rare_out = pd.DataFrame()
    if not rare.empty:
        print("\n" + "=" * 72, "\nRARE-CLASS STRESS TEST — paired within-seed, per prevalence pi\n" + "=" * 72)
        rare_out = all_pairwise(rare, RARE_METRICS, group_cols=("pi",))
        if not rare_out.empty:
            rare_out = rare_out.sort_values(["metric", "pi", "contrast"])
            print(rare_out.to_string(index=False))
        else:
            print("(no rare-class artifacts shared across models for these seeds — run evaluate_mrd on each ladder dir)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bal_out.to_csv(args.out + "_balanced.csv", index=False)
    if not rare_out.empty:
        rare_out.to_csv(args.out + "_rare.csv", index=False)
    per_seed_figure(bal, args.out + "_by_seed")
    print(f"\nsaved: {args.out}_balanced.csv" + ("" if rare_out.empty else f" , {args.out}_rare.csv"))


if __name__ == "__main__":
    main()
