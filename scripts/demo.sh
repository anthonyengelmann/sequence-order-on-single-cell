#!/usr/bin/env bash
# Quick demo on the built-in SYNTHETIC data -- no ScPCA download needed.
# Trains each model for a few epochs so you can watch the whole pipeline run,
# then does a rare-class eval on the FNN run. This only checks the code runs
# end-to-end; the reported numbers need the real ScPCA data (see README).
#
#   uv sync                 # once, to build the env (see README)
#   bash scripts/demo.sh
set -e
PY="${PY:-uv run python}"
DEVICE="${DEVICE:-cpu}"
OUT=outputs/demo
rm -rf "$OUT"
COMMON="-m data=synthetic training.epochs=3 device=$DEVICE seed=1"

echo "### FNN (dense)"
$PY scripts/train.py $COMMON model=fnn representation=dense hydra.sweep.dir=$OUT/fnn

echo "### LSTM (tokens, rank)"
$PY scripts/train.py $COMMON model=lstm representation=tokens hydra.sweep.dir=$OUT/lstm

echo "### Transformer (tokens, sinusoidal PE)"
$PY scripts/train.py $COMMON model=transformer representation=tokens model.positional_encoding=sinusoidal hydra.sweep.dir=$OUT/tf

echo "### rare-class eval (FNN)"
$PY scripts/evaluate_mrd.py --run_dir $OUT/fnn || echo "(eval skipped on the tiny synthetic split)"

echo
echo "Demo done -> $OUT/. This ran on synthetic data, so there is no real signal to reproduce."
echo "For the reported results: download ScPCA SCPCP000008 (see README), then bash scripts/run_report_experiments.sh"
