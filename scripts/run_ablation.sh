#!/usr/bin/env bash
# ============================================================================
# run_ablation.sh — the Q2 ORDERING ABLATION ONLY (what should have run overnight).
# No FNN, no Q1 ladder retrain. Runs on the cached max500 regime (fast) and reuses your
# existing max500 baselines (LSTM rank, Transformer no-PE) via consolidate_baselines.py.
#
#   SEEDS=1,2,3,4,5 bash scripts/run_ablation.sh
#
# Sweeps (CORE first, so stopping early still leaves the key results):
#   core:    LSTM random, LSTM ascending, Transformer sinusoidal-PE
#   stretch: LSTM alphabetical, LSTM importance_first, LSTM importance_last
# Baselines (rank / no-PE) are REUSED, not retrained.
# ============================================================================
SEEDS="${SEEDS:-1,2,3,4,5,6,7,8,9,10}"   # 10 seeds (overnight) — matches your ladder for tight pairing
MAXCELLS="${MAXCELLS:-500}"       # cached for seeds 1-10; keep 500 (matches your existing ladder)
WORKERS="${WORKERS:-4}"
PY="${PY:-python}"
DEVICE="${DEVICE:-mps}"
DO_MRD="${DO_MRD:-1}"             # 1 = also rare-class eval (sensitive readout, since balanced saturates)

if [ -z "$MAXCELLS" ] || [ "$MAXCELLS" = "full" ]; then MC=""; else MC="data.max_cells_per_sample=${MAXCELLS}"; fi
COMMON="-m data=scpca ${MC} training.num_workers=${WORKERS} device=${DEVICE} seed=${SEEDS}"
LOG=outputs/report/_logs; mkdir -p "$LOG"
run(){ echo -e "\n===== $1 =====  $(date +%H:%M:%S)"; shift; "$@" 2>&1 | tee "$LOG/ablation_$(date +%H%M%S).log"; }

echo "ABLATION (Q2 only): SEEDS=$SEEDS MAXCELLS=$MAXCELLS DEVICE=$DEVICE  — Q1 ladder is NOT retrained"

# 0. bring the existing max500 baselines into outputs/report/ (copy, no training)
$PY scripts/consolidate_baselines.py

# clear any stale ordering dirs from the aborted max2000 run
rm -rf outputs/report/order_lstm_* outputs/report/pe_transformer_sinusoidal

# Ordering sweeps — exactly the 5 requested. descending == rank == your existing baseline (reused).
run "LSTM random"               $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=random                                     hydra.sweep.dir=outputs/report/order_lstm_random
run "LSTM ascending"            $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=ascending                                  hydra.sweep.dir=outputs/report/order_lstm_ascending
run "LSTM importance_first"     $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=importance_first                           hydra.sweep.dir=outputs/report/order_lstm_impfirst
run "LSTM importance_last"      $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=importance_last                            hydra.sweep.dir=outputs/report/order_lstm_implast
run "Transformer sinusoidal-PE" $PY scripts/train.py $COMMON model=transformer representation=tokens representation.ordering=rank model.positional_encoding=sinusoidal   hydra.sweep.dir=outputs/report/pe_transformer_sinusoidal

# 3. rare-class eval (sensitive readout) on every ordering condition + the reused baselines
if [ "$DO_MRD" = "1" ]; then
  for d in ladder_lstm order_lstm_random order_lstm_ascending order_lstm_impfirst order_lstm_implast ladder_transformer pe_transformer_sinusoidal; do
    [ -d "outputs/report/$d" ] && run "rare-class eval: $d" $PY scripts/evaluate_mrd.py --run_dir outputs/report/$d
  done
fi

echo -e "\n===== ABLATION DONE $(date +%H:%M:%S) ====="
echo "nb05 registry already points at these dirs. Paired ordering analysis is ready."
