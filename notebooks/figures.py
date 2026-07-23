"""Finalized LYRA-Lite figures (notebook 02).

Each ``plot_*`` takes the arrays/dicts returned by ``lyra_lite.analysis.eda`` and
returns a Matplotlib ``Figure``; the notebook saves it via ``plot_style.save_fig``.
Plotting is lifted verbatim from notebook 02 so the notebook stays thin.
"""
from lyra_lite.data.representation import encode_cells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D 
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path

from plot_style import STYLE_COLORS, apply_paper_style

apply_paper_style()


def plot_blast_prevalence(pct, median):
    """Per-patient blast-prevalence histogram with the MRD zone + median."""
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.hist(pct, bins=np.arange(0, 101, 5), color=STYLE_COLORS["PRIMARY"], edgecolor="white", linewidth=0.5)

    mrd_max = 10
    ax.axvspan(0, mrd_max, color=STYLE_COLORS["ACCENT"], alpha=0.15, lw=0)
    y_max = ax.get_ylim()[1]
    ax.text(0.75, y_max * 0.80, "MRD\nregime", ha="left", va="top",
            color=STYLE_COLORS["ACCENT"], size=6, fontweight="bold")

    ax.hist(pct, bins=np.arange(0, 101, 5), color=STYLE_COLORS["PRIMARY"], edgecolor="white", linewidth=0.5)
    ax.axvline(median, color=STYLE_COLORS["TEXT"], lw=1, ls=":")
    ax.text(median - 2, ax.get_ylim()[1] * 0.9, f"Median: {median:.0f}%",
            ha="right", va="top", color=STYLE_COLORS["TEXT"], size=6.5, fontweight="normal")

    ax.set_xlabel("Blast prevalence per patient (%)")
    ax.set_ylabel("Number of patients")
    ax.set_xlim(0, 100)
    return fig


def plot_patient_shift(emb, ys, gs, bal_acc, chance):
    """Two-panel UMAP (by patient / by cell type) + the shift quantifier in the title."""
    plt.rcParams.update({"figure.figsize": (6.8, 2.8)})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.4), gridspec_kw={"wspace": 0.25})

    patient_codes = pd.factorize(gs)[0]
    ax1.scatter(emb[:, 0], emb[:, 1], c=patient_codes, cmap="plasma",
                s=1.5, alpha=0.3, linewidths=0, rasterized=True)
    ax1.set_title(f"By patient (n = {len(np.unique(gs))})", fontsize=7.5, fontweight="bold")

    for lab, col, name in [(0, STYLE_COLORS["CONTROL"], "Normal"), (1, STYLE_COLORS["PRIMARY"], "Blast")]:
        m = ys == lab
        ax2.scatter(emb[m, 0], emb[m, 1], c=col, s=1.5, alpha=0.3, linewidths=0, rasterized=True, label=name)
    ax2.legend(frameon=False, loc="upper right", markerscale=4, handletextpad=0.1, fontsize=5.5)
    ax2.set_title("By cell type", fontsize=7.5, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Cells cluster by patient (predictability: {bal_acc*100:.0f}% vs. {chance*100:.1f}% chance, {bal_acc/chance:.0f}×)",
                 fontsize=5, color=STYLE_COLORS["TEXT"], fontweight="medium")
    return fig


def plot_token_statistics(s):
    """Two-panel: genes-per-cell histogram + cumulative-expression-capture curve."""
    plt.rcParams.update({"figure.figsize": (7.5, 2.8)})
    fig, (axA, axB) = plt.subplots(1, 2, gridspec_kw={"wspace": 0.25})
    top_k = s["top_k"]

    # Panel A: histogram of genes detected per cell
    axA.hist(s["genes_per_cell"], bins=40, color=STYLE_COLORS["PRIMARY"], edgecolor="white", linewidth=0.5)
    y_max = axA.get_ylim()[1]
    axA.axvline(top_k, color=STYLE_COLORS["TEXT"], lw=1, ls="--")
    axA.text(top_k + 15, y_max * 0.9, f"top_k = {top_k}", ha="left", va="top",
             color=STYLE_COLORS["TEXT"], fontsize=5)
    axA.axvline(s["median_gpc"], color=STYLE_COLORS["TEXT"], lw=1, ls="-")
    axA.text(s["median_gpc"] + 8, y_max * 0.85, f"median = {s['median_gpc']}", ha="left", va="top",
             color=STYLE_COLORS["TEXT"], fontsize=5)
    axA.set_xlabel("Genes detected per cell (of 2000 HVGs)")
    axA.set_ylabel("Number of cells")
    axA.set_title(f"Sequence length available\n(median {s['median_gpc']}; {s['pad_frac']:.0%} padded)",
                  fontsize=7.5, fontweight="bold", pad=8)

    # Panel B: cumulative expression captured by top-ranked genes
    ranks = np.arange(1, s["n_ranks"] + 1)
    axB.fill_between(ranks, s["lo"], s["hi"], color=STYLE_COLORS["PRIMARY"], alpha=0.15, linewidth=0)
    axB.plot(ranks, s["mean_frac"], color=STYLE_COLORS["PRIMARY"], lw=1.5)
    axB.axvline(top_k, color=STYLE_COLORS["TEXT"], lw=1.2, ls="--")
    axB.plot([top_k], [s["cap_at_k"]], "o", color=STYLE_COLORS["TEXT"], ms=4)
    axB.annotate(f"top {top_k} genes\n≈ {s['cap_at_k']:.0%} of expression",
                 xy=(top_k, s["cap_at_k"]), xytext=(top_k + 260, s["cap_at_k"] - 0.15),
                 fontsize=6.5, color=STYLE_COLORS["TEXT"],
                 arrowprops=dict(arrowstyle="->", color=STYLE_COLORS["TEXT"], lw=0.8))
    axB.set_xlabel("Genes ranked by expression (per cell)")
    axB.set_ylabel("Cumulative fraction of expression")
    axB.set_xlim(0, s["n_ranks"])
    axB.set_ylim(0, 1.02)
    axB.set_title("Expression concentrates in top-ranked genes", fontsize=7.5, fontweight="bold", pad=8)

    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig


def plot_marker_violins(Xm, marker_names, y,
                        blast_markers=("CD19", "CD34", "DNTT", "CD79A"),
                        normal_markers=("CD3D", "NKG7", "LYZ", "CD14")):
    """Two 2x2 violin blocks: canonical blast vs. normal-lineage markers."""
    col = {m: i for i, m in enumerate(marker_names)}
    grp = np.where(y == 1, "Blast", "Normal")
    palette = {"Blast": STYLE_COLORS["PRIMARY"], "Normal": STYLE_COLORS["CONTROL"]}

    fig = plt.figure(figsize=(7.5, 3.2))
    subfigs = fig.subfigures(1, 2, wspace=0.15)

    def plot_grid(subfig, marker_list, block_title):
        axes = subfig.subplots(2, 2, sharex=False, sharey=False,
                               gridspec_kw={"wspace": 0.02, "hspace": 0.12})
        axes_flat = axes.flatten()
        for i, m in enumerate(marker_list):
            ax = axes_flat[i]
            df = pd.DataFrame({"expr": Xm[:, col[m]], "grp": grp})
            sns.violinplot(data=df, x="grp", y="expr", hue="grp", order=["Blast", "Normal"],
                           palette=palette, cut=0, inner=None, density_norm="width",
                           linewidth=0.8, legend=False, ax=ax)
            for coll in ax.collections:
                coll.set_alpha(0.85)
            ax.set_title(m, style="italic", fontsize=6, fontweight="bold", pad=2)
            ax.set_xlabel("")
            ax.set_ylabel("log-norm counts" if i % 2 == 0 else "", fontsize=6)
            ax.tick_params(axis="both", labelsize=6.5, pad=1, length=2)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax, left=False, bottom=False)
        subfig.suptitle(block_title, fontsize=8, fontweight="bold", color=STYLE_COLORS["TEXT"], y=1.02)

    plot_grid(subfigs[0], list(blast_markers), "B-lineage / Blast Markers")
    plot_grid(subfigs[1], list(normal_markers), "Normal Lineage Markers")
    return fig


def plot_normal_availability(a):
    """Two-panel: per-patient normal counts + unique normals required per pi."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"wspace": 0.25})
    norm_per, N_total = a["norm_per"], a["N_total"]
    series, need = a["series"], a["need"]
    pi_labels = ["10%", "1%", "0.1%", "0.01%"]

    # Panel A: histogram of per-patient normal-cell counts
    axA.hist(norm_per, bins=25, color=STYLE_COLORS["CONTROL"], alpha=0.9, edgecolor="white", linewidth=0.5)
    median_val = int(np.median(norm_per))
    axA.axvline(median_val, color=STYLE_COLORS["TEXT"], ls="-", lw=1.0, zorder=3)
    x_offset = norm_per.max() * 0.03
    axA.text(median_val + x_offset, 0.95, f"Median: {median_val}",
             transform=axA.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=STYLE_COLORS["TEXT"])
    axA.set_xlabel("Available normal cells per patient", fontsize=8.5)
    axA.set_ylabel("Number of patients", fontsize=8.5)
    axA.set_title("Severe per-patient normal starvation", fontsize=9.5, fontweight="bold", pad=12, loc="left")

    # Panel B: unique normals required vs. prevalence pi
    xb = np.arange(len(series))
    colors = [STYLE_COLORS["CONTROL"] if n <= N_total else STYLE_COLORS["PRIMARY"] for n in need]
    axB.bar(xb, need, width=0.6, color=colors, alpha=0.9, zorder=2)
    axB.set_yscale("log")
    axB.axhline(N_total, color=STYLE_COLORS["TEXT"], ls="-", lw=1.0, zorder=1)
    axB.text(xb[0] + 1.25, N_total * 1.25, f"Absolute Cohort Ceiling\n({N_total:,} cells)",
             ha="left", va="bottom", fontsize=6, fontweight="bold", color=STYLE_COLORS["TEXT"])
    test_pool_size = 8000
    axB.axhline(test_pool_size, color=STYLE_COLORS["TEXT"], ls=":", lw=1.0, alpha=0.8, zorder=1)
    axB.text(xb[0] - 0.25, test_pool_size * 1.25, f"Per-Run Test Pool (≈ {test_pool_size:,})",
             ha="left", va="bottom", fontsize=6, alpha=0.8, color=STYLE_COLORS["TEXT"])
    axB.set_xticks(xb)
    axB.set_xticklabels(pi_labels)
    axB.set_xlabel("Spiked-in prevalence (π)", fontsize=8.5)
    axB.set_ylabel("Unique normal cells required (log scale)", fontsize=8.5)
    axB.set_title("Global background required for low π", fontsize=9.5, fontweight="bold", pad=12, loc="left")

    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    return fig


def plot_ablation_preview(X, REP):
    # REP = {"top_k": 256, "ordering": "rank", "use_value": True, "pad_id": n_hvgs, "seed": 42}

    cell_idx = 0  # Inspect the very first cell in the matrix
    target_top_k = REP["top_k"]
    target_pad_id = REP["pad_id"]  # Defaults to n_hvgs (2000)

    # Extract the cell vector and ensure it is a dense 2D batch shape (1, n_hvgs)
    single_cell = X[cell_idx]
    if hasattr(single_cell, "toarray"):  # Safeguard if X is a SciPy sparse matrix
        single_cell = single_cell.toarray()

    cell_batch = np.atleast_2d(single_cell).astype(np.float32)

    # Run the real encoding logic and extract BOTH tokens and continuous values
    t_rank, v_rank = encode_cells(cell_batch, top_k=target_top_k, ordering="rank", pad_id=target_pad_id)
    t_asc, v_asc   = encode_cells(cell_batch, top_k=target_top_k, ordering="ascending", pad_id=target_pad_id)
    t_rand, v_rand = encode_cells(cell_batch, top_k=target_top_k, ordering="random", pad_id=target_pad_id, seed=42)

    # Filter out PAD tokens
    mask_rank = t_rank[0] != target_pad_id
    mask_asc  = t_asc[0] != target_pad_id
    mask_rand = t_rand[0] != target_pad_id

    # Extract the top 5 non-zero entries for tokens and values
    toks_r, vals_r     = t_rank[0][mask_rank][:5], v_rank[0][mask_rank][:5]
    toks_asc, vals_asc = t_asc[0][mask_asc][:5], v_asc[0][mask_asc][:5]
    toks_d, vals_d     = t_rand[0][mask_rand][:5], v_rand[0][mask_rand][:5]

    # Format as "Token ID (Expression Value)"
    df_ablation_preview = pd.DataFrame({
        "Rank-Ordered (Default)":   [f"Token {int(t)} ({v:.2f})" for t, v in zip(toks_r, vals_r)],
        "Ascending (recency probe)": [f"Token {int(t)} ({v:.2f})" for t, v in zip(toks_asc, vals_asc)],
        "Randomized (Permuted)":    [f"Token {int(t)} ({v:.2f})" for t, v in zip(toks_d, vals_d)]
    })
    return df_ablation_preview

 

def plot_balanced_sweep(df_bal, present, DISPLAY):
    if not df_bal.empty:
        fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=100)
        
        # 1. Perfect Score Line (Dezenter, wie in der Referenz)
        ax.axhline(1.0, color=STYLE_COLORS.get("TEXT", "#333333"), ls=":", lw=1.0, alpha=0.6, zorder=0)
        ax.text(len(present)-0.95, 1.001, "Perfect AUROC", color=STYLE_COLORS.get("TEXT", "#333333"), 
                fontsize=6, fontweight="bold", ha="center", va="bottom", alpha=0.8, zorder=0)

        # 2. Saturation Zone (Mit negativer zorder ganz im Hintergrund)
        ax.axhspan(0.978, 1.0, color=STYLE_COLORS.get("CONTROL"), alpha=0.4, zorder=-1)
        ax.text(0.9, 0.978, 'Saturation Zone', color=STYLE_COLORS.get("TEXT", "#333333"), 
                fontsize=6, ha='center', va='bottom', style='italic', alpha=0.8, zorder=0)
        
        # 3. Der prominente Mittelwert (Diamond)
        g_mean = df_bal.groupby("model")["auroc"].mean().reindex(present)
        ax.scatter(
            range(len(present)), g_mean,
            color=STYLE_COLORS.get("PRIMARY"),
            edgecolor="white",
            linewidth=1.0,           # Rand minimal dünner für mehr Eleganz
            s=40,
            marker="D",
            zorder=2,
            label="Mean AUROC"
        )

        # 4. Rohdaten (Seeds) im Hintergrund
        num_seeds = df_bal["seed"].nunique()
        raw_colors = [
            STYLE_COLORS.get("ACCENT"), 
            STYLE_COLORS.get("SECONDARY"), 
            STYLE_COLORS.get("TERTIARY")
        ]
        fallback_palette = sns.color_palette("muted", num_seeds)
        clean_colors = [
            raw_colors[i] if (i < len(raw_colors) and raw_colors[i] is not None) else fallback_palette[i]
            for i in range(num_seeds)
        ]

        sns.stripplot(
            data=df_bal.sort_values("seed"), 
            x="model", 
            y="auroc", 
            hue="seed",
            order=present,
            palette=clean_colors,
            jitter=0.12,          
            marker="o",             
            alpha=0.6,           
            size=4.5,            
            zorder=1,            
            ax=ax
        )

        # 5. Achsen-Styling (Exakt nach Referenz-Grundsätzen)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([DISPLAY.get(a, a) for a in present], fontsize=8)
        ax.set_ylabel("Test AUROC", fontsize=8.5)
        
        # NEU: Y-Achse atmen lassen. Wir suchen den kleinsten Wert und ziehen etwas Puffer ab.
        y_min = df_bal["auroc"].min()
        ax.set_ylim(y_min - 0.005, 1.005) 

        # Spines simpel und clean entfernen 
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)

        # 6. Legende formatieren (Clean und minimalistisch)
        # Wir bauen die Legende komplett manuell, um die vielen Seed-Einträge auf einen zu reduzieren.
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', label='Mean AUROC',
                   markerfacecolor=STYLE_COLORS.get("PRIMARY"), markersize=7, 
                   markeredgecolor='white', markeredgewidth=1.0),
            Line2D([0], [0], marker='o', color='w', label='Seeds',
                   markerfacecolor=STYLE_COLORS.get("TEXT", "#888888"), markersize=5, 
                   alpha=0.6)
        ]

        ax.legend(
            handles=legend_elements, 
            fontsize=7, 
            bbox_to_anchor=(1.05, 0.925),
            loc='upper left', 
            frameon=False
        )

        # 7. Titel 
        ax.set_title("Balanced task is saturated across all models", 
                    fontsize=9.5, fontweight="bold", pad=12, loc="left")

        plt.tight_layout()
        return fig

def plot_mrd_lod_curves(df_mrd, DISPLAY):

    if not df_mrd.empty:
        metrics = [("auprc_mean", "AUPRC"), ("sens_at_fpr_mean", "Sensitivity@FPR"), ("ece_mean", "ECE")]
        
        # 1. Standard Nature-Breite (7.5) übernehmen, wie in figures.py definiert
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), dpi=100, gridspec_kw={"wspace": 0.38})
        
        # Exakte Farbpalette
        style_palette = [
            STYLE_COLORS.get("CONTROL"),
            STYLE_COLORS.get("ACCENT"),
            STYLE_COLORS.get("PRIMARY"),
            STYLE_COLORS.get("TERTIARY"),
        ]
        
        for ax, (col, label) in zip(axes, metrics):
            # Das Grid wurde komplett entfernt: Nature-Plots verzichten zugunsten der 
            # optischen Ruhe meist auf Hintergrundlinien (wie in deinen Referenzen).
            
            for i, (model, gm) in enumerate(df_mrd.groupby("model")):
                s = gm.groupby("pi")[col].agg(["mean", "std"]).sort_index(ascending=False)
                
                # Farbe für das aktuelle Modell holen (mit Fallback)
                model_color = style_palette[i] if i < len(style_palette) and style_palette[i] is not None else sns.color_palette("muted")[i]
                
                # 3. Der Haupt-Plot: Etwas filigraner (ms=4, lw=1.5), passend zu plot_token_statistics
                ax.plot(
                    s.index, s["mean"], 
                    marker="o", 
                    ms=4,                        
                    linewidth=1.5,               
                    markeredgecolor="white",     
                    markeredgewidth=0.8,
                    color=model_color, 
                    label=DISPLAY.get(model, model),
                    zorder=2
                )
                
                # 4. Saubere Schatten: Exakte Adaption aus plot_token_statistics (alpha=0.15, linewidth=0)
                ax.fill_between(
                    s.index, 
                    s["mean"] - s["std"], 
                    s["mean"] + s["std"], 
                    color=model_color, 
                    alpha=0.15,                   
                    linewidth=0,            
                    zorder=1
                )
                
            ax.set_xscale("log")
            
            # 5. Achsenbeschriftung strikt auf Referenz-Fontsize 8.5
            ax.set_xlabel("Spiked-in prevalence (π)", fontsize=7)
            ax.set_ylabel(label, fontsize=7)
            
            # 6. Spines simpel und clean entfernen, Ticks auf Fontsize 8
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=8)

        # 7. Globale Legende oben zentriert (Fontsize 7.5, etwas engerer Text-Abstand)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, 0.98),  
            ncol=3,                      
            frameon=False, 
            fontsize=7.5,
            handletextpad=0.4
        )
        
        # 8. Titel (Linksbündig, Fontsize 9.5, Bold)
        # x=0.08 zentriert den Start des Titels perfekt über der Y-Achse des ersten Plots
        fig.suptitle(
            "MRD limit-of-detection — Where the ladder separates", 
            fontsize=9.5, 
            fontweight="bold", 
            x=0.5,     
            ha="center",  
            y=1.15      
        ) 
    return fig


def summarize_balanced_task(df_bal, present, DISPLAY, out_path=None):
    """
    Generates a mean ± std summary table for the balanced task, 
    matching models robustly regardless of case variations.
    """
    if df_bal.empty:
        print("Warning: df_bal is empty.")
        return None
        
    metrics = {"auroc": "AUROC", "auprc": "AUPRC", "ece": "ECE"}
    
    formatted_df = pd.DataFrame(index=[DISPLAY.get(m, m) for m in present])
    formatted_df.index.name = "Architecture"
    
    # Debug-Hilfe, falls es immer noch leer ist (zeigt dir, was wirklich im DataFrame steht)
    existing_models = df_bal["model"].unique()
    
    for raw_col, clean_name in metrics.items():
        column_values = []
        for model in present:
            # Case-insensitive Match: sucht z.B. 'fnn' oder 'FNN' im DataFrame
            model_mask = df_bal["model"].astype(str).str.lower() == str(model).lower()
            model_data = df_bal[model_mask][raw_col]
            
            # Falls gar nichts gefunden wurde, versuchen wir einen Teil-Match (contains)
            if len(model_data) == 0:
                model_mask = df_bal["model"].astype(str).str.lower().str.contains(str(model).lower())
                model_data = df_bal[model_mask][raw_col]
            
            if len(model_data) > 0:
                mean_val = model_data.mean()
                std_val = model_data.std()
                # Falls nur 1 Seed da ist, ist std NaN -> mit 0.0000 auffüllen
                std_val = std_val if not pd.isna(std_val) else 0.0
                column_values.append(f"{mean_val:.4f} ± {std_val:.4f}")
            else:
                column_values.append("No Data")
                
        formatted_df[clean_name] = column_values
    
    # Clean console view
    print("\nTable 1 — Balanced task performance across seeds (Mean ± Std)")
    print("=" * 65)
    print(formatted_df.to_string())
    print("=" * 65)
    
    if "No Data" in formatted_df.values:
        print(f"⚠️ Debug-Info: In 'df_bal' existieren nur diese Modelle: {existing_models}")
    print("\n")
    
    # 4. Export to LaTeX
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        latex_str = formatted_df.style.to_latex(
            column_format="l" + "c"*len(metrics)
        )
        with open(out_path, "w") as f:
            f.write(latex_str)
            
    return formatted_df


def summarize_mrd_task(df_mrd, present, DISPLAY, out_path=None):
    """
    Generates a mean ± std summary table for the MRD task stratified by prevalence (π),
    matching models robustly, avoiding MultiIndex/KeyErrors, and formatting with clean borders.
    """
    if df_mrd.empty:
        print("Warning: df_mrd is empty.")
        return None
        
    metrics = {"auprc_mean": "AUPRC", "sens_at_fpr_mean": "Sensitivity@FPR", "ece_mean": "ECE"}
    
    # 1. Alle verfügbaren Prävalenz-Stufen (pi) absteigend holen
    all_pis = sorted(df_mrd["pi"].unique(), reverse=True)
    num_pis = len(all_pis)  # <-- Hier sauber für das gesamte Skript definiert!
    
    # 2. Multi-Index für die finale Tabelle manuell aufbauen (Modell x Pi)
    index_tuples = []
    for model in present:
        clean_model_name = DISPLAY.get(model, model)
        for pi in all_pis:
            index_tuples.append((clean_model_name, pi))
            
    m_index = pd.MultiIndex.from_tuples(index_tuples, names=["Architecture", "π (Prevalence)"])
    formatted_df = pd.DataFrame(index=m_index, columns=metrics.values())
    
    # 3. Tabelle isoliert befüllen
    for model in present:
        clean_model_name = DISPLAY.get(model, model)
        model_mask = df_mrd["model"].astype(str).str.lower() == str(model).lower()
        
        if df_mrd[model_mask].empty:
            model_mask = df_mrd["model"].astype(str).str.lower().str.contains(str(model).lower())
            
        df_model = df_mrd[model_mask]
        
        for pi in all_pis:
            df_pi = df_model[df_model["pi"] == pi]
            
            for raw_col, clean_name in metrics.items():
                if not df_pi.empty and raw_col in df_pi.columns:
                    mean_val = df_pi[raw_col].mean()
                    std_col = raw_col.replace("_mean", "_std")
                    std_val = df_pi[std_col].mean() if std_col in df_pi.columns else 0.0
                    std_val = std_val if not pd.isna(std_val) else 0.0
                    
                    formatted_df.loc[(clean_model_name, pi), clean_name] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    formatted_df.loc[(clean_model_name, pi), clean_name] = "No Data"
    
    # 4. Clean console view mit perfekt ausgerichteten Spalten und Linien
    print("\nTable 2 — MRD limit-of-detection across seeds (Mean ± Std)")
    print("=" * 95)
    
    header_fmt = "{:<15} {:<10} {:^18} {:^18} {:^18}"
    row_fmt    = "{:<15} {:<10} {:<18} {:<18} {:<18}"
    
    print(header_fmt.format("Architecture", "π (Preval.)", "AUPRC", "Sensitivity@FPR", "ECE"))
    print("-" * 95)
    
    for m_idx, model in enumerate(present):
        clean_model_name = DISPLAY.get(model, model)
        
        for p_idx, pi in enumerate(all_pis):
            disp_model = clean_model_name if p_idx == 0 else ""
            disp_pi = f"{pi:.3f}"
            vals = formatted_df.loc[(clean_model_name, pi)]
            
            print(row_fmt.format(
                disp_model, 
                disp_pi, 
                vals["AUPRC"], 
                vals["Sensitivity@FPR"], 
                vals["ECE"]
            ))
            
        if m_idx < len(present) - 1:
            print("-" * 95)
            
    print("=" * 95 + "\n")
    
    # 5. Export to LaTeX (mit professionellen booktabs hlines zwischen den Blöcken)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        latex_lines = formatted_df.to_latex().split('\n')
        new_latex = []
        data_row_count = 0
        in_data = False
        
        for line in latex_lines:
            new_latex.append(line)
            if '\\midrule' in line:
                in_data = True
                continue
            if '\\bottomrule' in line:
                in_data = False
                
            if in_data and not line.strip().startswith('%'):
                if '\\\\' in line:
                    data_row_count += 1
                    if data_row_count % num_pis == 0 and data_row_count < len(formatted_df):
                        new_latex.append('\\midrule')
                        
        latex_str = '\n'.join(new_latex)
        with open(out_path, "w") as f:
            f.write(latex_str)
            
    return formatted_df

_FOREST_TITLES = {
    "ece":        ("Calibration (ECE)", "← better | worse →"),
    "sens":       ("Sensitivity@FPR",   "← worse | better →"),
    "auprc":      ("AUPRC",             "← worse | better →"),
    "test_auroc": ("Test AUROC",        "← worse | better →"),
    "test_ece":   ("Test ECE",          "← better | worse →"),
}

def get_color_variations(base_color, factor=None):

    return ["#265264FF", "#4A7B89FF", "#91BCCF"]


def plot_forest(pdf, metrics=None, group_col="pi", ci_mult=1.96, contrasts=None, figsize=None):
    """Forest plot of paired Δ ± CI with a zero line (one panel per metric)."""
    if metrics is None:
        metrics = list(dict.fromkeys(pdf["metric"]))
    if contrasts is None:
        contrasts = list(dict.fromkeys(pdf["contrast"]))
        
    has_groups = group_col in pdf.columns and pdf[group_col].nunique() > 1
    groups = sorted(pdf[group_col].unique(), reverse=True) if has_groups else [None]
    
    # 1. FARBEN: Zieht sich deine SECONDARY Farbe und baut eine Palette daraus
    base_color = STYLE_COLORS.get("SECONDARY", "#1f77b4") # Fallback, falls Dict-Key fehlt
    palette = get_color_variations(base_color, max(1, len(groups)))
    gcol = {g: palette[i] for i, g in enumerate(groups)}
    
    # 2. PROPORTIONEN: Flache Höhe (0.3), breite Koordinatensysteme (5.0 pro Metrik)
    figsize = figsize or (5.0 * len(metrics), 0.3 * len(contrasts) * max(1, len(groups)) + 1.2)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True, squeeze=False)
    
    gap = 0.8
    for ax, metric in zip(axes[0], metrics):
        title, dirlab = _FOREST_TITLES.get(metric, (metric, ""))
        # Collect present rows per contrast; groups already ordered [0.1, 0.01, 0.001].
        layout = []
        for c in contrasts:
            band = []
            for g in groups:
                sub = pdf[(pdf["contrast"] == c) & (pdf["metric"] == metric)]
                if has_groups:
                    sub = sub[sub[group_col] == g]
                if not sub.empty:
                    band.append((g, float(sub["mean_delta"].iloc[0]), float(sub["se"].iloc[0])))
            if band:
                layout.append((c, band))

        # Walk y from the TOP down: contrast #1 and π=0.1 sit at the top (deterministic, no invert).
        total = sum(len(b) for _, b in layout) + gap * max(0, len(layout) - 1)
        y, yt, yl, first = total, [], [], True
        for c, band in layout:
            ys = []
            for g, d, se in band:
                ax.errorbar(d, y, xerr=ci_mult * se, fmt="s", ms=4.5, capsize=0, lw=1.5,
                            color=gcol[g], mec="white", mew=0.5, zorder=3,
                            label=(f"{group_col} = {g:g}" if has_groups and first else None))
                ys.append(y); y -= 1
            first = False
            yt.append(float(np.mean(ys)))
            # Split only on the real separator: "FNN-LSTM" -> "FNN vs LSTM";
            # "imp-last - imp-first" -> "IMP-LAST vs IMP-FIRST" (keep internal hyphens).
            _sep = " - " if " - " in c else "-"
            yl.append(" vs ".join(t.strip() for t in c.split(_sep)).upper())
            y -= gap
        ax.set_ylim(-0.8, total + 0.4)

        # 3. WISSENSCHAFTLICHES LAYOUT: Keine Hintergründe oder störende Achsen
        ax.axvline(0, ls="--", lw=1.0, color="#666666", zorder=1)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=7.5)
        
        ax.grid(False)                       # Schaltet alle störenden Gitterlinien aus
        ax.tick_params(axis='y', length=0)   # Entfernt Striche an den Y-Labels
        
        # Außenrahmen entfernen für freistehenden Text
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False) 
        
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8)
        ax.set_xlabel(f"Paired Δ  ({dirlab})", fontsize=7.5, labelpad=5)
        
    if has_groups:
        axes[0][0].legend(title="Prevalence (π)", frameon=False, fontsize=6.5, title_fontsize=7, loc="best")
        
    fig.suptitle("Paired Within-Seed Contrasts (Δ ± 95% CI)", fontsize=9.5, fontweight="bold", y=1.03)
    fig.tight_layout(w_pad=5.5)
    
    return fig

def format_scientific_summary(df_paired, metric_filter=("auprc", "sens", "ece")):
    """Publication-style paired MODEL-comparison table: Δ ± SD (+ σ = Δ/SE), FNN contrasts first.
    Expects a paired frame with columns contrast ('FNN-LSTM'), metric, mean_delta, se [, std, pi]."""
    if df_paired is None or df_paired.empty:
        return None
    df = df_paired[df_paired["metric"].isin(list(metric_filter))].copy()
    def _parts(x):
        x = str(x); sep = " - " if " - " in x else "-"
        p = [t.strip() for t in x.split(sep)]
        return p[0].upper(), p[-1].upper()
    df["Model_A"] = df["contrast"].map(lambda x: _parts(x)[0])
    df["Model_B"] = df["contrast"].map(lambda x: _parts(x)[1])
    df["Comparison"] = df["Model_A"] + " vs. " + df["Model_B"]
    # Keep contrasts in their INPUT order (matches the forest's row order: rank, asc-vs-desc,
    # importance, PE for nb05; FNN-first for nb04) instead of sorting Comparison alphabetically.
    _corder = {c: i for i, c in enumerate(dict.fromkeys(df["contrast"]))}
    df["_corder"] = df["contrast"].map(_corder)
    has_pi = "pi" in df.columns
    df = df.sort_values(["_corder", "metric"] + (["pi"] if has_pi else []),
                        ascending=[True, True] + ([False] if has_pi else []))
    mean_col = "mean_delta" if "mean_delta" in df.columns else df.filter(like="mean").columns[0]
    std_col = "std" if "std" in df.columns else None
    df["Absolute Delta (Δ)"] = ([f"{m:+.4f} ± {s:.4f}" for m, s in zip(df[mean_col], df[std_col])]
                                if std_col else [f"{m:+.4f}" for m in df[mean_col]])
    if "se" in df.columns:
        df["σ (Δ/SE)"] = (df[mean_col] / df["se"]).map(lambda x: f"{x:+.1f}")
    idx = ["Comparison", "metric"] + (["pi"] if has_pi else [])
    cols = idx + ["Absolute Delta (Δ)"] + (["σ (Δ/SE)"] if "se" in df.columns else [])
    return df[cols].set_index(idx)


def format_pairwise_table(pdf, ci_mult=1.96):
    """Generic tidy table (any contrasts): contrast × metric [× pi] with Δ, 95% CI, and σ = Δ/SE."""
    if pdf is None or pdf.empty:
        return pdf
    df = pdf.copy()
    df["Δ (95% CI)"] = [f"{d:+.4f} [{d - ci_mult * s:+.4f}, {d + ci_mult * s:+.4f}]"
                        for d, s in zip(df["mean_delta"], df["se"])]
    df["σ = Δ/SE"] = (df["mean_delta"] / df["se"]).map(lambda x: f"{x:+.1f}")
    keep = [c for c in ["contrast", "metric", "pi", "Δ (95% CI)", "σ = Δ/SE"] if c in df.columns]
    return df[keep].reset_index(drop=True)


# ── notebook helpers: paired-contrast loaders + dumbbell plot (nb04 ladder, nb05 ordering) ──
def _short(target):
    return str(target).split(".")[-1]


def load_balanced(sweeps):
    """Every metrics.json under the sweep dir(s) -> tidy per-run frame (model, seed, auroc/auprc/ece)."""
    import glob, json
    rows = []
    for sw in sweeps:
        for mj in glob.glob(f"{sw}/**/metrics.json", recursive=True):
            d = json.load(open(mj))
            rows.append({"model": _short(d["model"]), "seed": d.get("seed"),
                         "auroc": d["test_auroc"], "auprc": d["test_auprc"], "ece": d["test_ece"]})
    return pd.DataFrame(rows)


def load_mrd(sweeps):
    """Every <run>/mrd/mrd_lod.csv, labelled by architecture (from the sibling metrics.json)."""
    import glob, json
    from pathlib import Path
    rows = []
    for sw in sweeps:
        for csv in glob.glob(f"{sw}/**/mrd_lod.csv", recursive=True):
            run = Path(csv).parent.parent
            d = pd.read_csv(csv); d["model"] = _short(json.load(open(run / "metrics.json"))["model"])
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_RARE_COL = {"auprc": "auprc_mean", "sens": "sens_at_fpr_mean", "ece": "ece_mean"}


def ordering_seed_metric(sweep_root, key="ece", rare_pi=0.001):
    """seed -> rare-class metric[key] at rare_pi, from each run's mrd/mrd_lod.csv (key in {auprc,sens,ece})."""
    import glob, os, json
    col = _RARE_COL.get(key, key); out = {}
    for csv in glob.glob(os.path.join(sweep_root, "**", "mrd_lod.csv"), recursive=True):
        run = os.path.dirname(os.path.dirname(csv))
        try:
            seed = json.load(open(os.path.join(run, "metrics.json")))["seed"]
        except Exception:
            continue
        d = pd.read_csv(csv); row = d[d["pi"] == rare_pi]
        if not row.empty and col in row:
            out[seed] = float(row[col].iloc[0])
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def ordering_paired_delta(root_a, root_b, key="ece", rare_pi=0.001):
    """Paired within-seed delta (A - B) on a rare-class metric; returns (per-seed df, summary dict)."""
    a, b = ordering_seed_metric(root_a, key, rare_pi), ordering_seed_metric(root_b, key, rare_pi)
    seeds = sorted(set(a) & set(b), key=str)
    if not seeds:
        raise ValueError(f"No shared seeds for {key!r} between the two conditions.")
    df = pd.DataFrame({"seed": seeds, "A": [a[s] for s in seeds], "B": [b[s] for s in seeds]})
    df["delta"] = df["A"] - df["B"]
    n = len(df); sd = df["delta"].std(ddof=1) if n > 1 else 0.0
    return df, {"n_seeds": n, "mean_delta": df["delta"].mean(), "se": sd / np.sqrt(n) if n > 1 else float("nan")}


def _load_mrd_allpi(sweep_root):
    """seed × π rare-class metrics under a sweep dir (all π)."""
    import glob, os, json
    rows = []
    for csv in glob.glob(os.path.join(sweep_root, "**", "mrd_lod.csv"), recursive=True):
        run = os.path.dirname(os.path.dirname(csv))
        try:
            seed = json.load(open(os.path.join(run, "metrics.json")))["seed"]
        except Exception:
            continue
        for _, r in pd.read_csv(csv).iterrows():
            rows.append({"seed": seed, "pi": float(r["pi"]), "auprc": float(r["auprc_mean"]),
                         "sens": float(r["sens_at_fpr_mean"]), "ece": float(r["ece_mean"])})
    return pd.DataFrame(rows)


DEFAULT_ORDERING_CONTRASTS = [("rank - random", "lstm_rank", "lstm_random"),
                              ("ascending - descending", "lstm_asc", "lstm_rank"),
                              ("imp-last - imp-first", "lstm_imp_last", "lstm_imp_first"),
                              ("PE - no-PE", "tf_pe", "tf_nope")]


def ordering_rare_contrasts(ablation, contrasts=None, metrics=("auprc", "sens", "ece")):
    """Paired-Δ frame (contrast × metric × π) for the ordering ablation on rare-class metrics.
    `ablation` maps keys -> sweep dirs; `contrasts` is a list of (label, key_a, key_b)."""
    contrasts = contrasts or DEFAULT_ORDERING_CONTRASTS
    cache = {}
    def load(k):
        if k not in cache:
            cache[k] = _load_mrd_allpi(ablation[k])
        return cache[k]
    rows = []
    for name, ka, kb in contrasts:
        A, B = load(ka), load(kb)
        if A.empty or B.empty:
            continue
        for metric in metrics:
            for pi in sorted(set(A.pi) & set(B.pi)):
                ap = A[A.pi == pi].set_index("seed")[metric]; bp = B[B.pi == pi].set_index("seed")[metric]
                s = sorted(set(ap.index) & set(bp.index)); dl = np.array([ap[k] - bp[k] for k in s]); n = len(dl)
                if n > 1:
                    rows.append({"contrast": name, "metric": metric, "pi": pi,
                                 "mean_delta": float(dl.mean()), "se": float(dl.std(ddof=1) / np.sqrt(n)),
                                 "std": float(dl.std(ddof=1))})
    return pd.DataFrame(rows)


# ── notebook 01 · dataset & cohort helpers (loader + Table 1 builder; lifted verbatim) ──
def load_cohort(cache="../data/cache/cached_scpca_2000hvg_full.npz",
                data_dir="../data/SCPCP000008_ann-data/SCPCP000008_single-cell",
                meta_out="../data/cache/scpca_patient_metadata.csv"):
    """Cached corpus (X, y, groups, genes) + provenance manifest + per-patient metadata sidecar.
    Metadata (diagnosis/age/sex/tissue) is built once from each .h5ad `.obs` and cached to CSV.
    Returns a SimpleNamespace with .X .y .groups .genes .manifest .meta .n_hvgs."""
    import json, types
    import anndata as ad
    from pathlib import Path
    CACHE = Path(cache)
    blob = np.load(CACHE, allow_pickle=False)
    X, y, groups, genes = blob["X"], blob["y"], blob["groups"], blob["genes"]
    manifest = json.loads(CACHE.with_suffix(".json").read_text())
    n_hvgs = genes.shape[0]
    print("X:", X.shape, "| genes:", n_hvgs)
    print("cells:", len(y), "| blast rate:", round(float(y.mean()), 3),
          "| patients:", len(set(groups.tolist())))

    DATA_DIR = Path(data_dir)
    META_OUT = Path(meta_out)
    META_COLS = ["participant_id", "diagnosis", "subdiagnosis", "age", "sex",
                 "tissue_location", "disease_timing"]
    if META_OUT.exists():
        meta = pd.read_csv(META_OUT, index_col="participant_id")
    else:
        rows = []
        for f in sorted(DATA_DIR.rglob("*_processed_rna.h5ad")):
            obs = ad.read_h5ad(f, backed="r").obs           # backed = reads obs only, not X
            if "participant_id" not in obs:
                continue
            rows.append({c: (obs[c].iloc[0] if c in obs else None) for c in META_COLS})
        meta = (pd.DataFrame(rows).drop_duplicates("participant_id").set_index("participant_id"))
        meta["age"] = pd.to_numeric(meta["age"], errors="coerce")   # some ages are 'False' -> NaN
        META_OUT.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(META_OUT)
    print(f"metadata: {len(meta)} patients | diagnoses: {meta.diagnosis.nunique()} | age missing: {int(meta.age.isna().sum())}")
    return types.SimpleNamespace(X=X, y=y, groups=groups, genes=genes,
                                 manifest=manifest, meta=meta, n_hvgs=n_hvgs)


def cohort_summary(groups, y):
    """Per-patient cell/blast counts + blast fraction, sorted by prevalence (Table-1 substrate)."""
    df = pd.DataFrame({"patient": groups, "blast": y})
    cohort = (df.groupby("patient")
                .agg(n_cells=("blast", "size"), n_blast=("blast", "sum"))
                .assign(n_normal=lambda d: d.n_cells - d.n_blast,
                        blast_frac=lambda d: (d.n_blast / d.n_cells).round(3))
                .sort_values("blast_frac", ascending=False))
    print("patients:", len(cohort),
          "| blast fraction range:", round(cohort.blast_frac.min(), 3), "-", round(cohort.blast_frac.max(), 3))
    return cohort


def cohort_split_report(X, y, groups, seed=42):
    """Reconstruct + print the leakage-free patient-level train/val/test split sizes (mirrors scripts/train.py)."""
    from sklearn.model_selection import GroupShuffleSplit
    train_idx, temp_idx = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups=groups))
    val_rel, test_rel = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed)
                             .split(X[temp_idx], y[temp_idx], groups=groups[temp_idx]))
    for name, idx in [("train", train_idx), ("val", temp_idx[val_rel]), ("test", temp_idx[test_rel])]:
        g = groups[idx]
        print(f"{name:5s}: {len(idx):6d} cells | {len(set(g.tolist())):2d} patients | blast {y[idx].mean():.3f}")


def build_table1(cohort, per_patient, n_hvgs, out="tables/table1_cohort.tex"):
    """Cohort Table 1 -> (LaTeX string, preview DataFrame); also writes `out`. Lifted from nb01."""
    from pathlib import Path
    N = len(per_patient)
    tot, blast = int(cohort.n_cells.sum()), int(cohort.n_blast.sum())
    def rng(s, dp=0):
        s = s.dropna()
        return f"{s.median():.{dp}f} [{s.min():.{dp}f}--{s.max():.{dp}f}]"
    def npct(k):  return f"{int(k)} ({round(100 * k / N)}%)"
    def cats(col): return [("sub", str(i), npct(v)) for i, v in per_patient[col].value_counts(dropna=False).items()]
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")

    rows = [
        ("stat", "Patients", "n", str(N)),
        ("stat", "Total cells", "n", f"{tot:,}"),
        ("stat", "Blast cells", "n (%)", f"{blast:,} ({round(100*blast/tot)}%)"),
        ("stat", "Normal cells", "n (%)", f"{tot-blast:,} ({round(100*(tot-blast)/tot)}%)"),
        ("stat", "Cells per patient", "median [range]", rng(cohort.n_cells)),
        ("stat", "Genes per Cell(2000 HVGS)", "n", f"{n_hvgs:,}"),
        ("stat", "Blast prevalence (%)", "median [range]", rng(cohort.blast_frac * 100)),
        ("stat", "Age at diagnosis (years)", "median [range]", rng(per_patient.age, 1)),
        ("section", "Sex", "n (%)"),       *cats("sex"),
        ("section", "Diagnosis", "n (%)"), *cats("diagnosis"),
        ("section", "Tissue", "n (%)"),    *cats("tissue_location"),
    ]

    # --- LaTeX ---
    body = []
    for r in rows:
        if r[0] == "stat":
            body.append(f"{esc(r[1])} -- \\textit{{\\footnotesize {esc(r[2])}}} & {esc(r[3])} \\\\")
        elif r[0] == "section":
            body += ["\\addlinespace", f"\\textbf{{{esc(r[1])}}} -- \\textit{{\\footnotesize {esc(r[2])}}} & \\\\"]
        else:
            body.append(f"\\quad {esc(r[1])} & {esc(r[2])} \\\\")
    tex = ("\\begin{table}\n\\centering\n"
           "\\renewcommand{\\arraystretch}{1.4}\n\\setlength{\\tabcolsep}{15pt}\n"
           "\\caption{Cohort characteristics (ScPCA SCPCP000008).}\n\\label{tab:cohort}\n"
           "\\begin{tabular}{ll}\n\\toprule\n"
           "\\textbf{Characteristic} & \\textbf{Value} \\\\\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # --- plain preview DataFrame (same rows) ---
    IND = "   "                       # indent for sub-items (renders in HTML)
    prev = []
    for r in rows:
        if r[0] == "stat":      prev.append((f"{r[1]} ({r[2]})", r[3]))
        elif r[0] == "section": prev.append((f"{r[1]} ({r[2]})", ""))
        else:                   prev.append((IND + r[1], r[2]))
    preview_df = pd.DataFrame(prev, columns=["Characteristic", "Value"])

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(tex)
    print(f"saved -> {out}")
    return tex, preview_df


def plot_paired(df, label_a, label_b, ylabel, title):
    """Per-seed connected (dumbbell) plot — each grey line is one held-out patient split."""
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for _, r in df.iterrows():
        ax.plot([0, 1], [r["A"], r["B"]], "-", color=STYLE_COLORS["CONTROL"], lw=0.9, alpha=0.7, zorder=1)
    ax.scatter([0] * len(df), df["A"], color=STYLE_COLORS["PRIMARY"], s=18, zorder=2)
    ax.scatter([1] * len(df), df["B"], color=STYLE_COLORS["ACCENT"], s=18, zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([label_a, label_b]); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8, fontweight="bold")
    return fig
