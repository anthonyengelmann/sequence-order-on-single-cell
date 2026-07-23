#!/usr/bin/env bash
# ============================================================================
# smoke_test.sh — fast end-to-end sanity check BEFORE the overnight run.
# Reuses an ALREADY-BUILT cache (no ETL) and caps epochs, so it runs in minutes.
# Touches every new/risky path: dense, tokens, ascending, importance(+probe), PE,
# rare-class eval, paired analysis. Writes to outputs/smoke/ (never outputs/report/).
#
#   bash scripts/smoke_test.sh
#
# CACHE RULE (this is what made MAXCELLS=200 slow): a subsampled regime is cached
# PER SEED (cached_scpca_2000hvg_max500_seed<N>.npz), so a NEW number rebuilds 10
# caches, each a full raw read. Only use a regime already on disk:
#   * 500   -> cached for seeds 1-10   (default here; smoke uses seed 1)
#   * full  -> one seed-independent cache (set MAXCELLS=full)
# Do NOT pass 200 / 1000 / 2000 — those trigger per-seed rebuilds.
# ============================================================================
MAXCELLS="${MAXCELLS:-500}"      # 500 (cached) or full (cached). Nothing else.
EPOCHS="${EPOCHS:-1}"             # 1 epoch = fast; we only check the pipeline runs, not accuracy
PY="${PY:-python}"
DEVICE="${DEVICE:-mps}"

if [ -z "$MAXCELLS" ] || [ "$MAXCELLS" = "full" ]; then MC=""; else MC="data.max_cells_per_sample=${MAXCELLS}"; fi
COMMON="-m data=scpca ${MC} training.num_workers=0 training.epochs=${EPOCHS} device=${DEVICE} seed=1"
run(){ echo -e "\n### $1  $(date +%H:%M:%S)"; shift; "$@"; }

echo "SMOKE: MAXCELLS=$MAXCELLS EPOCHS=$EPOCHS DEVICE=$DEVICE  (reusing cache; writing to outputs/smoke/)"
rm -rf outputs/smoke
set -e   # stop at the first failure — surfacing a broken arm is the whole point

run "FNN (dense)"                $PY scripts/train.py $COMMON model=fnn         representation=dense                                                                    hydra.sweep.dir=outputs/smoke/fnn
run "LSTM (tokens, rank)"        $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=rank                                        hydra.sweep.dir=outputs/smoke/lstm_rank
run "LSTM (ascending)"           $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=ascending                                   hydra.sweep.dir=outputs/smoke/lstm_asc
run "LSTM (importance_first)"    $PY scripts/train.py $COMMON model=lstm        representation=tokens representation.ordering=importance_first                            hydra.sweep.dir=outputs/smoke/lstm_impfirst
run "Transformer (PE sinusoidal)" $PY scripts/train.py $COMMON model=transformer representation=tokens representation.ordering=rank model.positional_encoding=sinusoidal   hydra.sweep.dir=outputs/smoke/tf_pe
for d in fnn lstm_rank tf_pe; do
  run "rare-class eval ($d)"     $PY scripts/evaluate_mrd.py --run_dir outputs/smoke/$d
done
run "paired analysis"            $PY scripts/paired_analysis.py --fnn outputs/smoke/fnn --lstm outputs/smoke/lstm_rank --transformer outputs/smoke/tf_pe --out outputs/smoke/paired

set +e
echo -e "\n### SMOKE PASSED — every arm ran. Kick off the overnight job (RUNBOOK Quickstart step 2)."
