#!/usr/bin/env bash
# Natural-model controls for natural_grid_cexs.py over the FULL 10,000-image test sets
# (the first-1000 runs are resumed via the existing .jsonl checkpoints). Same models,
# layer lists, Lipschitz refs and radii as run_natural_grid_cexs.sh, but NO FP_BIAS env,
# so attack_verified_certifier_nat_numpy builds the natural (unbiased) models.
# Detached: nohup caffeinate -i -s ./run_natural_grid_controls.sh > results/natural_grid_cex/run_natural_all.log 2>&1 &
set -uo pipefail
cd ~/git/verified-certified-robustness/scripts
PY=cav2025-artifact-venv/bin/python
PC=~/git/python-certifier
R=$PC/models/precomputed; CAV=../cav2025-models
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"
OUT=results/natural_grid_cex
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset FP_BIAS FP_BIAS_POS BIAS_OUTPUT
echo "### natural MNIST float32, eps 0.3 (full test set)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC $PY natural_grid_cexs.py float32 mnist "$MNIST_L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json \
  --eps 0.3 --out $OUT/natgrid_mnist_natural.json --restarts 5 2>&1 | tee -a $OUT/natgrid_mnist_natural.log
echo "### natural Fashion-MNIST float32, eps 0.25 (full test set)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC $PY natural_grid_cexs.py float32 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 $R/dafny_fashion_gram13.json \
  --eps 0.25 --out $OUT/natgrid_fashion_natural.json --restarts 5 2>&1 | tee -a $OUT/natgrid_fashion_natural.log
echo "### natural CIFAR-10 float32, eps 0.141 (full test set)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC $PY natural_grid_cexs.py float32 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 $R/dafny_cifar10_gram12.json \
  --eps 0.141 --out $OUT/natgrid_cifar_natural.json --restarts 5 2>&1 | tee -a $OUT/natgrid_cifar_natural.log
echo "### done (natural controls)"
