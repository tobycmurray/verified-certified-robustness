#!/usr/bin/env bash
# Grid-restricted counter-example search (grid_search_cexs.py) on the four
# suites of run_quantised_checks.sh: biased MNIST (1e6), natural MNIST float16,
# biased Fashion-MNIST (3e6), biased CIFAR-10 (4e6). Same model CSVs, layer
# lists, Lipschitz refs and FP_BIAS env as run_quantised_checks.sh /
# generate_numpy_cexs.sh. Outputs results/quantised_cex/gridsearch_<tag>.json
# (+ _points/ npy pairs) and logs results/quantised_cex/gridsearch_<tag>.log.
# Usage: caffeinate -i ./run_grid_search_cexs.sh [extra grid_search_cexs.py args]
#   SUFFIX=_anylabel ./run_grid_search_cexs.sh --allow-label-change   # variant run
set -uo pipefail
cd ~/git/verified-certified-robustness/scripts
PY=cav2025-artifact-venv/bin/python
PC=~/git/python-certifier
T=$PC/tests; R=$PC/models/precomputed; CAV=../cav2025-models
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"
OUT=results/quantised_cex
SFX=${SUFFIX:-}
EXTRA=("$@")
S=${BIAS_DIR:-/private/tmp/claude-502/-Users-tobiasm1-git-float-conservatism-paper/ddf998f9-1707-420c-b03a-1e27b49e17e8/scratchpad}
mkdir -p "$OUT" "$S"

echo "### biased MNIST float32 (1e6)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=1e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_mnist_1e6.txt \
  $PY grid_search_cexs.py float32 mnist "$MNIST_L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json $T/cex_mnist_float32_biased_1e6_end_numpy \
  --out $OUT/gridsearch_mnist_biased$SFX.json --naive $OUT/quant_mnist_biased.json ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee $OUT/gridsearch_mnist_biased$SFX.log
echo "### natural MNIST float16"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC \
  $PY grid_search_cexs.py float16 mnist "$MNIST_L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json $T/cex_mnist_float16_numpy \
  --out $OUT/gridsearch_mnist_float16$SFX.json --naive $OUT/quant_mnist_float16.json ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee $OUT/gridsearch_mnist_float16$SFX.log
echo "### biased Fashion-MNIST float32 (3e6)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=3e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_fashion_3e6.txt \
  $PY grid_search_cexs.py float32 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 $R/dafny_fashion_gram13.json $T/cex_fashion_mnist_float32_biased_3e6_end_numpy \
  --out $OUT/gridsearch_fashion_biased$SFX.json --naive $OUT/quant_fashion_biased.json ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee $OUT/gridsearch_fashion_biased$SFX.log
echo "### biased CIFAR-10 float32 (4e6)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=4e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_cifar_4e6.txt \
  $PY grid_search_cexs.py float32 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 $R/dafny_cifar10_gram12.json $T/cex_cifar10_float32_biased_4e6_end_numpy \
  --out $OUT/gridsearch_cifar_biased$SFX.json --naive $OUT/quant_cifar_biased.json ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee $OUT/gridsearch_cifar_biased$SFX.log
echo "### done"
