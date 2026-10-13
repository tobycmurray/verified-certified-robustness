#!/usr/bin/env bash
# Natural-image grid counter-examples (natural_grid_cexs.py) on the three
# adversarially-biased float32 models, over their full 10,000-image test sets,
# at the paper's standard radii (MNIST 0.3, Fashion-MNIST 0.25, CIFAR-10 0.141).
# Same model CSVs, layer lists, Lipschitz refs and FP_BIAS env as
# run_grid_search_cexs.sh. Outputs results/natural_grid_cex/natgrid_<tag>.{json,jsonl}
# (+ _points/ npy pairs, .certifier_input.txt) and logs natgrid_<tag>.log.
# Resumable: re-running skips points already in the .jsonl.
#
# Usage: ./run_natural_grid_cexs.sh [mnist|fashion|cifar|all] [extra natural_grid_cexs.py args]
# Detached full run (survives the terminal):
#   nohup caffeinate -i -s ./run_natural_grid_cexs.sh all > results/natural_grid_cex/run_all.log 2>&1 &
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
WHICH=${1:-all}; shift || true
EXTRA=("$@")
S=${BIAS_DIR:-$OUT/biases}
mkdir -p "$OUT" "$S"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

run_mnist() {
  echo "### biased MNIST float32 (1e6), eps 0.3"
  env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=1e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_mnist_1e6.txt \
    $PY natural_grid_cexs.py float32 mnist "$MNIST_L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json \
    --eps 0.3 --out $OUT/natgrid_mnist_biased.json --restarts 5 ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a $OUT/natgrid_mnist_biased.log
}
run_fashion() {
  echo "### biased Fashion-MNIST float32 (3e6), eps 0.25"
  env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=3e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_fashion_3e6.txt \
    $PY natural_grid_cexs.py float32 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 $R/dafny_fashion_gram13.json \
    --eps 0.25 --out $OUT/natgrid_fashion_biased.json --restarts 5 ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a $OUT/natgrid_fashion_biased.log
}
run_cifar() {
  echo "### biased CIFAR-10 float32 (4e6), eps 0.141"
  env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=4e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_cifar_4e6.txt \
    $PY natural_grid_cexs.py float32 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 $R/dafny_cifar10_gram12.json \
    --eps 0.141 --out $OUT/natgrid_cifar_biased.json --restarts 5 ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a $OUT/natgrid_cifar_biased.log
}

case "$WHICH" in
  mnist) run_mnist ;;
  fashion) run_fashion ;;
  cifar) run_cifar ;;
  all) run_mnist; run_fashion; run_cifar ;;
  *) echo "unknown model set: $WHICH"; exit 1 ;;
esac
echo "### done ($WHICH)"
