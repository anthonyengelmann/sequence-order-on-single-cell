# What Do Sequence Models Actually Use?

**Dissecting order-sensitivity and permutation-invariance with a non-linguistic token stream.**

Author: **Anthony Engelmann** — HHU Düsseldorf, BSc Computer Science (university ID: `3160882`)
Course: Python for NLP, SS 2026 — Dr. Shutong Feng · License: MIT

Read the full report here: [Engelmann NLP Research Report](./Engelmann_NLP_Research_Report.pdf)

> **Thesis.** Sequence models (LSTM, Transformer) assume *order carries information*, but natural
> language cannot test that assumption cleanly — you cannot remove word order without destroying
> meaning. A single-cell gene-expression profile is an **order-free token set** with an explicit
> *(id, value)* channel and an experimenter-imposed position, so it *can*. We use it as a controlled
> **instrument** to measure what sequence order contributes to the standard NLP architecture ladder
> (FNN → LSTM → Transformer). The biology is the substrate; the contribution is a controlled study of
> **sequence-order inductive bias**.

![Cell Representation Methodology](./notebooks/figures/03_cell_representation.png)

---

## 1 · Research questions

**Q1 — Does sequential inductive bias help when the signal is a set?**
The architecture ladder **FNN** (bag-of-tokens) → **LSTM** (imposed order) → **Transformer**
(permutation-invariant set), over rank-value gene tokens, on a balanced task and a rare-class stress
test.

**Q2 — When we impose an arbitrary order, do order-sensitive models exploit it?**
An **ordering ablation**, analysed *paired within-seed*: LSTM `rank` / `random` / `ascending`,
importance-ordering, and Transformer **positional-encoding on/off**.

Why a cell is a good NLP testbed: it is (i) **order-free** — any ordering is our choice, so we can
ablate order directly; (ii) **position-controllable** — we place known content at any slot to probe
recency/primacy; (iii) **dual-channel** — each token is *(gene-id, expression-value)*, so we can ask
whether position is redundant with an explicit value channel; (iv) a **syntax-free vocabulary** of
~2,000 "words," isolating the bag-of-tokens regime from the sequential-syntax regime text confounds.

---

## 2 · Data

**Single-cell Pediatric Cancer Atlas (ScPCA), project `SCPCP000008`** (pediatric B-ALL, scRNA-seq of
bone marrow; 84 patients). Used purely as the token substrate, no clinical claim is made.

| | |
|---|---|
| **Source / licence** | [ScPCA portal](https://scpca.alexslemonade.org/) (Alex's Lemonade Stand Foundation), released **CC-BY 4.0**. Download project `SCPCP000008` as `*_processed_rna.h5ad` (X = logcounts) into `data/SCPCP000008_ann-data/`. |
| **Tokens** | top **2,000 highly variable genes** (re-selected on the concatenated cohort) = the vocabulary; `PAD_ID = 2000`. |
| **Label** | `submitter_celltype_annotation`: `Blast` → 1; normal lineages → 0; `Submitter-excluded` dropped. Annotation-derived → *mild label noise*. |
| **Class balance** | strong imbalance (cohort blast fraction ~87%). Motivates the **rare-class stress test** and AUPRC/ECE alongside AUROC. |
| **Splits** | **patient-level** 70/15/15 via `GroupShuffleSplit` on `participant_id`, no patient in two splits (leakage-free). Test = unseen patients = the distribution-shift stress source. |

The **rare-class stress test** (code name `mrd`) is a controllable class-imbalance regime, a spike-in
dilution driving the positive rate to 1% / 0.1%, the NLP analogue of rare-intent/entity detection. It
is a **methods knob, not a clinical claim.**

---

## 3 · Models & representation

| Model | Representation | Inductive bias |
|---|---|---|
| **FNN** | dense 2,000-dim VHVG vector | bag-of-features; order-agnostic |
| **LSTM** | rank-value gene tokens (embedded), `top_k=256`, masked | recurrent; order-**sensitive** |
| **Transformer** | rank-value gene tokens, masked, mean-pool | self-attention; order-**agnostic** unless PE added |

Tokenization (`src/lyra_lite/data/representation.py::encode_cells`): each cell → a padded sequence of
*(gene-id, value)* tokens; ordering ∈ `{rank, random, ascending, importance_first/last}`; `PAD_ID` +
attention masking. All models are trained **from scratch** (no pretraining) under one protocol (Adam,
`BCEWithLogitsLoss`, identical split/metrics), so differences are attributable to architecture/order,
not training setup.

---

## 4 · Evaluation

- **Balanced task:** AUROC + AUPRC + **ECE** (calibration) on the held-out-patient test set; model
  selected on validation loss (test touched once); multi-seed with 95% CIs.
- **Rare-class stress test:** AUPRC, sensitivity@FPR, and ECE across a dilution series
  (`scripts/evaluate_mrd.py` → `mrd_lod.csv`).
- **Paired within-seed analysis:** deltas on the *same* patient split (one knob changed), so the
  patient-split variance that dominates cross-seed spread cancels.

  ![MRD Limit of Detection Results](./notebooks/figures/04b_mrd_lod_curves.png)

---

## 5 · Quickstart (no data needed)

```bash
uv sync                     # reproducible env from pyproject.toml / uv.lock
bash scripts/demo.sh        # trains FNN + LSTM + Transformer a few epochs on built-in
                            # SYNTHETIC data, then runs a rare-class eval on the FNN
```

This proves the full pipeline runs end-to-end **without any download**. It is a code check only,
reproducing the *reported numbers* needs the real ScPCA data (below).

---

## 6 · Reproduce the reported results (real data)

```bash
uv sync --extra bio         # adds anndata / scanpy / h5py for the real ScPCA loader
export DATA_DIR=./data       # where SCPCP000008_ann-data/ lives (see §2 for the download)

# one-time cache build (~15 min), then instant on reuse:
uv run python scripts/train.py data=scpca

# the full report sweep (ladder + rare-class eval + ordering ablation + PE):
bash scripts/run_report_experiments.sh
```

All hyperparameters live in `configs/` (Hydra). Every run snapshots its resolved config to
`outputs/.../.hydra/`, and `evaluate_mrd.py` reads that snapshot so evaluation always matches the
checkpoint. Seeds are fixed (`cfg.seed`) and the dataset build is seed-independent. Results land in
`outputs/.../metrics.json` (balanced) and `.../mrd_lod.csv` (rare-class); the notebooks read those and
build the report figures. A subset of these result files (`metrics.json`, `mrd_lod.csv`) is committed
under `outputs/report/` so the notebooks reproduce the figures without retraining.

---

## 7 · Tests

```bash
uv sync --extra dev         # adds pytest and the other dev tools
uv run pytest
```

Unit tests cover the tokenizer (`encode_cells`), the synthetic data generator, the ECE metric, package
imports, and the model forward shapes (`tests/`).

---

## 8 · Repository map

```
sequence-order-4nlp/
├── README.md
├── pyproject.toml · uv.lock         ← environment (uv), pinned versions
├── configs/                         ← Hydra: data/ model/ representation/ training/ eval/
├── scripts/
│   ├── train.py                     ← trains one config → best_model.pt + metrics.json
│   ├── evaluate_mrd.py              ← rare-class stress-test eval → mrd_lod.csv
│   ├── paired_analysis.py           ← paired within-seed analysis (Q1)
│   ├── aggregate.py                 ← multi-seed aggregation
│   ├── run_report_experiments.sh    ← full report sweep (needs real data)
│   ├── smoke_test.sh                ← fast pipeline check on a cached regime
│   └── demo.sh                      ← runnable demo on synthetic data (no download)
├── src/lyra_lite/
│   ├── data/                        ← scpca loader + cache, synthetic, representation (tokenizer)
│   ├── models/                      ← fnn, lstm, transformer
│   ├── analysis/                    ← ece, mrd_eval, eda, multiseed (aggregation)
│   └── training/                    ← trainer loop
├── notebooks/                       ← 00–05 analysis → figures/ + tables/ 
└── tests/                           ← pytest unit tests
```

The 8-page report is submitted separately as a PDF (its LaTeX source is kept out of the repo).

---

## 9 · Code ↔ experiment ↔ report mapping

| Experiment | Command | Artifact | Notebook | Report |
|---|---|---|---|---|
| Q1 ladder (balanced) | `train.py -m model=fnn,lstm,transformer` | `metrics.json` | `04` | Results · Q1 |
| Q1 rare-class | `evaluate_mrd.py --run_dir <sweep>` | `mrd_lod.csv` | `04` | Results · Q1 |
| Q2 ordering (LSTM) | `train.py -m representation.ordering=rank,random,ascending` | `metrics.json` + `mrd_lod.csv` | `05` | Results · Q2 |
| Q2 positional encoding | `train.py -m model.positional_encoding=none,sinusoidal` | `metrics.json` + `mrd_lod.csv` | `05` | Results · Q2 |

---

## 10 · Connection to the course & AI usage

Tokenization/vocabulary (**P2**), padding + embeddings (**P5.1**), attention masking (**P6, P8.1**),
FNN (**P3, P4.2**), LSTM (**P5.2**), Transformer encoder (**P8.1**). The project *extends* the
practicals into a controlled inductive-bias study rather than reproducing a notebook.

A full **Declaration of AI Usage** is included in the report (per the project guidelines).
