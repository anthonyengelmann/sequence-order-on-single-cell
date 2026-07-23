#!/usr/bin/env bash
# ============================================================================
# run_report_experiments.sh — the full run plan for the Python-for-NLP report.
# Run this OVERNIGHT from a REAL terminal (not the VS Code integrated one).
#
#   bash scripts/run_report_experiments.sh
#
# Knobs (override at the front):  SEEDS, MAXCELLS, WORKERS, PY, DEVICE
#   SEEDS=1,2,3,4,5 MAXCELLS=1000 bash scripts/run_report_experiments.sh   # faster
# Deterministic output dirs under outputs/report/<name>/ -> paste straight into nb04/nb05.
# No `set -e`: one failed sweep must not kill the whole night.
# ============================================================================
SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10}"     # 10 seeds (drop to 5 if wall-time is tight)
# MAXCELLS: cells/sample. A subsampled regime is cached PER SEED, so a new number builds 10 caches
# ONCE (during the first sweep) and every later sweep reuses them — fine for an overnight run.
#   500  = already cached (fast; good for the smoke test / a quick preview)
#   2000 = recommended for the report run (builds the 10 caches once, then reuses)
#   full = all cells (may OOM a laptop — avoid unless you know you have the RAM)
MAXCELLS="${MAXCELLS:-2000}"                # overnight default: more data than the 500 preview
WORKERS="${WORKERS:-4}"                     # DataLoader workers (0 is very slow)
PY="${PY:-python}"                          # set PY="uv run python" if needed
DEVICE="${DEVICE:-mps}"                      # mps (Apple Silicon) | cuda | cpu

# omit the override entirely for the full (seed-independent) cache; otherwise pass the cached number
if [ -z "$MAXCELLS" ] || [ "$MAXCELLS" = "full" ]; then MC=""; else MC="data.max_cells_per_sample=${MAXCELLS}"; fi
COMMON="-m data=scpca ${MC} training.num_workers=${WORKERS} device=${DEVICE} seed=${SEEDS}"
LOG=outputs/report/_logs; mkdir -p "$LOG"
run(){ echo -e "\n===== $1 =====  $(date +%H:%M:%S)"; shift; "$@" 2>&1 | tee "$LOG/$(date +%H%M%S).log"; }

echo "SEEDS=$SEEDS  MAXCELLS=$MAXCELLS  WORKERS=$WORKERS  DEVICE=$DEVICE  PY=$PY"
echo "First training run also builds the data cache for this regime (~15 min, once)."

# ---------------------------------------------------------------------------
# PRIORITY 1 — Q1 architecture ladder (balanced task -> metrics.json)   [CORE]
# ---------------------------------------------------------------------------
run "ladder: FNN"         $PY scripts/train.py $COMMON model=fnn         representation=dense                             hydra.sweep.dir=outputs/report/ladder_fnn
run "ladder: LSTM (rank)" $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=rank hydra.sweep.dir=outputs/report/ladder_lstm
run "ladder: Transformer (no PE)" $PY scripts/train.py $COMMON model=transformer representation=tokens representation.ordering=rank model.positional_encoding=none hydra.sweep.dir=outputs/report/ladder_transformer

# ---------------------------------------------------------------------------
# PRIORITY 2 — Q1 rare-class stress test (dilution eval -> mrd_lod.csv)  [CORE]
# ---------------------------------------------------------------------------
for m in ladder_fnn ladder_lstm ladder_transformer; do
  run "rare-class eval: $m" $PY scripts/evaluate_mrd.py --run_dir outputs/report/$m
done

# ---------------------------------------------------------------------------
# PRIORITY 3 — Q2 LSTM ordering ablation (balanced metrics.json; rank reused above)
#   ascending REQUIRES the encode_cells add in docs/RUNBOOK.md §0 (this arm fails otherwise)
# ---------------------------------------------------------------------------
run "order: LSTM alphabetical" $PY scripts/train.py $COMMON model=lstm representation=tokens representation.ordering=alphabetical hydra.sweep.dir=outputs/report/order_lstm_alphabetical
run "order: LSTM random"       $PY scripts/train.py $COMMON model=lstm representation=tokens representation.ordering=random       hydra.sweep.dir=outputs/report/order_lstm_random
run "order: LSTM ascending"    $PY scripts/train.py $COMMON model=lstm representation=tokens representation.ordering=ascending    hydra.sweep.dir=outputs/report/order_lstm_ascending

# rare-class readout for the ordering arms (more sensitive than the saturated balanced metric)
for m in order_lstm_alphabetical order_lstm_random order_lstm_ascending; do
  run "rare-class eval: $m" $PY scripts/evaluate_mrd.py --run_dir outputs/report/$m
done

# ---------------------------------------------------------------------------
# PRIORITY 4 — Q2 Transformer positional-encoding ablation (none reused above)
# ---------------------------------------------------------------------------
run "PE: Transformer sinusoidal" $PY scripts/train.py $COMMON model=transformer representation=tokens representation.ordering=rank model.positional_encoding=sinusoidal hydra.sweep.dir=outputs/report/pe_transformer_sinusoidal

# ---------------------------------------------------------------------------
# PRIORITY 5 — Q2 importance-ordering (implemented; probe fit on TRAIN split, leakage-free)
# ---------------------------------------------------------------------------
run "order: LSTM importance-first" $PY scripts/train.py $COMMON model=lstm representation=tokens representation.ordering=importance_first hydra.sweep.dir=outputs/report/order_lstm_impfirst
run "order: LSTM importance-last"  $PY scripts/train.py $COMMON model=lstm representation=tokens representation.ordering=importance_last  hydra.sweep.dir=outputs/report/order_lstm_implast
for m in order_lstm_impfirst order_lstm_implast; do
  run "rare-class eval: $m" $PY scripts/evaluate_mrd.py --run_dir outputs/report/$m
done

# ---------------------------------------------------------------------------
# PRIORITY 6 — Q1 paired within-seed analysis of the ladder (the airtight step)
#   Pairs FNN/LSTM/Transformer by seed so patient-split variance cancels; prints mean paired
#   delta +/- SE and sign-consistency (e.g. "FNN better calibrated in 9/10 seeds"), + a per-seed
#   figure showing the models move together across splits.
# ---------------------------------------------------------------------------
run "paired within-seed analysis" $PY scripts/paired_analysis.py \
    --fnn outputs/report/ladder_fnn --lstm outputs/report/ladder_lstm --transformer outputs/report/ladder_transformer \
    --out outputs/report/paired

echo -e "\n===== ALL DONE $(date +%H:%M:%S) ====="
echo "Artifacts under outputs/report/*/ (metrics.json + mrd_lod.csv). Paste these dirs into nb04/nb05."
ls -d outputs/report/*/ 2>/dev/null
