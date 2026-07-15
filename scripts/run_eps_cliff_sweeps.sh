#!/bin/bash
# Map certified robustness vs eval epsilon across the seed_variance_sweep.sh models,
# to test the predicted certified-radius distribution shape under gloro training:
# hollow density below eps_train, pile-up + cliff just above it, and cross-seed
# variance exploding exactly at the cliff (Le & Cao's phenomenon relocated).
#
# Grids bracket eps_train (HIGGS 0.1, MNIST 0.45; MNIST certifies at 0.3) densely
# near the predicted cliff, sparsely in the tails. Safe to re-run: eval_eps_sweep.py
# appends + flushes per row (dedupe on (run,eps) when analysing if re-run).
#
# Usage: ./run_eps_cliff_sweeps.sh {higgs|mnist}
set -u
cd "$(dirname "$0")"
PY="./cav2025-artifact-venv/bin/python3"
R="seed_variance_results"

case "${1:-}" in
  higgs)
    HIGGS_N_TRAIN=500000 HIGGS_N_TEST=500000 PYTHONPATH=. "$PY" eval_eps_sweep.py \
      higgs "[512,512,512,512,512]" 1 1000 "$R/eps_sweep_higgs.tsv" \
      0.02,0.05,0.08,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.13,0.15,0.175,0.2,0.25,0.3 \
      "$R"/higgs_w512_d5_s*
    ;;
  mnist)
    # Runs over whichever MNIST seeds have completed so far; re-run as more land.
    PYTHONPATH=. "$PY" eval_eps_sweep.py \
      mnist "[128,128,128,128,128,128,128,128]" 28 1000 "$R/eps_sweep_mnist.tsv" \
      0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.475,0.5,0.525,0.55,0.6,0.65,0.7,0.8,1.0 \
      "$R"/mnist_tobler_s*
    ;;
  *)
    echo "Usage: $0 {higgs|mnist}"; exit 1
    ;;
esac
