# plot_style.py
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Deine feste Farbpalette für das gesamte Paper
STYLE_COLORS = {
    "PRIMARY": "#740707",     # Dein edles Dunkelrot (z.B. für Blasten/Patienten)
    "CONTROL": "#7F8C8D",     # Ein sauberes Mausgrau für Kontrollen/Baselines
    "TEXT": "#1A1A1A",        # Soft-Black für Text und Achsen
    "ACCENT": "#E6A100",       # Optionales gedecktes Gold/Gelb
    "SECONDARY": "#91BCCF",   # Optionales Blau für zusätzliche Akzente
    "TERTIARY": "#ED807A",    # Optionales Orange für weitere Akzente
    "QUATERNARY": "#1F821C"   # Optionales Rot für weitere Akzente
}

# 2. Globale Matplotlib-Einstellungen (Nature-Level)
def apply_paper_style():
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.3),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.edgecolor": STYLE_COLORS["TEXT"],
        "axes.labelcolor": STYLE_COLORS["TEXT"],
        "xtick.color": STYLE_COLORS["TEXT"],
        "ytick.color": STYLE_COLORS["TEXT"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
    })


# 3. Einheitliches Speichern: Vektor-PDF (Paper) + Raster-PNG (Vorschau)
def save_fig(fig, name, figdir="figures"):
    """Save a figure once as vector PDF and once as 300-dpi PNG under figdir/."""
    Path(figdir).mkdir(exist_ok=True)
    fig.savefig(f"{figdir}/{name}.pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(f"{figdir}/{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
