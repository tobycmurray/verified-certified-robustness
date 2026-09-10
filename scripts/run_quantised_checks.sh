#!/usr/bin/env bash
set -uo pipefail
cd ~/git/verified-certified-robustness/scripts
PY=cav2025-artifact-venv/bin/python
PC=~/git/python-certifier
T=$PC/tests; R=$PC/models/precomputed; CAV=../cav2025-models
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
L="[128,128,128,128,128,128,128,128]"
S=/private/tmp/claude-502/-Users-tobiasm1-git-float-conservatism-paper/ddf998f9-1707-420c-b03a-1e27b49e17e8/scratchpad
echo "### biased MNIST float32"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=1e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_mnist_1e6.txt \
  $PY check_quantised_cexs.py float32 mnist "$L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json $T/cex_mnist_float32_biased_1e6_end_numpy --out results/quantised_cex/quant_mnist_biased.json
echo "### natural MNIST float16"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC \
  $PY check_quantised_cexs.py float16 mnist "$L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json $T/cex_mnist_float16_numpy --out results/quantised_cex/quant_mnist_float16.json
echo "### natural MNIST float32"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC \
  $PY check_quantised_cexs.py float32 mnist "$L" "$MNIST_CSV" 28 $R/dafny_mnist_gram20.json $T/cex_mnist_float32_numpy --out results/quantised_cex/quant_mnist_float32.json
echo "### done"
echo "### biased Fashion-MNIST float32 (3e6)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=3e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_fashion_3e6.txt \
  $PY check_quantised_cexs.py float32 fashion_mnist "[256,128,128,128,128,128,128,128,128,128,128,128]" "$FASHION_CSV" 28 $R/dafny_fashion_gram13.json $T/cex_fashion_mnist_float32_biased_3e6_end_numpy --out results/quantised_cex/quant_fashion_biased.json
echo "### biased CIFAR-10 float32 (4e6)"
env PYTHONPATH=. PYTHON_CERTIFIER=$PC FP_BIAS=4e6 FP_BIAS_POS=end BIAS_OUTPUT=$S/biases_cifar_4e6.txt \
  $PY check_quantised_cexs.py float32 cifar10 "[512,256,128,128,128,128,128,128]" "$CIFAR_CSV" 32 $R/dafny_cifar10_gram12.json $T/cex_cifar10_float32_biased_4e6_end_numpy --out results/quantised_cex/quant_cifar_biased.json
echo "### done"
