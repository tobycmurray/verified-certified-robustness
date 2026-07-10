#!/usr/bin/env bash
# Generate the numpy-execution counter-example suites for every "cex" test in
# python-certifier/tests/run_tests.sh (except the synthetic z3 one): 30 cexs per
# case via attack_verified_certifier_nat_numpy.py (DeepFool on the Keras model,
# cex search judged by the IEEE-754-compliant numpy execution).
#
# Cases (mirroring run_tests.sh's cex matrix; per-model RQ1 grams):
#   mnist          float32 / float16 / float64   (L ref: dafny_mnist_gram20)
#   fashion_mnist  float32 / float16 / float64   (L ref: dafny_fashion_gram13)
#   cifar10        float32 / float64             (L ref: dafny_cifar10_gram12; no
#                                                 float16 -- n*u >= 1, not certifiable)
#   mnist_biased_1e6_end          float32        (FP_BIAS=1e6, pos=end)
#   fashion_mnist_biased_3e6_end  float32        (FP_BIAS=3e6, pos=end)
#   cifar10_biased_4e6_end        float32        (FP_BIAS=4e6, pos=end)
# No bfloat16: it is not in run_tests.sh's matrix and compliant_forward cannot
# execute it (numpy has no native bfloat16).
#
# The Lipschitz refs are the same Dafny reference JSONs the certifier certifies
# against, so max_eps is calibrated to the exact L bounds used in the cex tests.
# Output dirs are the _numpy siblings of the existing Keras cex dirs. Idempotent:
# a case whose counter_examples.json already exists is skipped (delete it to
# regenerate). Runs under cav2025-artifact-venv (ART/DeepFool + gmpy2).
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # verified-certified-robustness/scripts
VCRS_ROOT="$(cd "$SCRIPTS/.." && pwd)"
PC="${PYTHON_CERTIFIER:-$VCRS_ROOT/../python-certifier}"
TESTS="$PC/tests"
REFS="$PC/models/precomputed"
CAV="$VCRS_ROOT/cav2025-models"
PY="${PY:-$SCRIPTS/cav2025-artifact-venv/bin/python}"
ATTACK="$SCRIPTS/attack_verified_certifier_nat_numpy.py"
N="${N:-30}"
# Output root for the cex_*_numpy dirs (default: python-certifier/tests, where
# run_tests.sh consumes them). Override with CEX_OUT_DIR for a regeneration run
# that must not collide with committed suites (e.g. the artifact container,
# whose baked tests/ already holds them and would trigger skip-if-exists).
OUT="${CEX_OUT_DIR:-$TESTS}"
mkdir -p "$OUT"

MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"

run_cex () {  # fmt dataset layers csv isize lipref outdir [fp_bias]
  local fmt="$1" dataset="$2" layers="$3" csv="$4" isize="$5" lip="$6" outdir="$7" fp_bias="${8:-}"
  if [ -s "$outdir/counter_examples.json" ]; then
    echo "  exists, skip: $(basename "$outdir")"; return 0
  fi
  mkdir -p "$outdir"
  echo ">>> $(basename "$outdir") (fmt=$fmt${fp_bias:+, FP_BIAS=$fp_bias end})"
  ( cd "$outdir" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" \
      ${fp_bias:+FP_BIAS="$fp_bias"} ${fp_bias:+FP_BIAS_POS=end} \
      ${fp_bias:+BIAS_OUTPUT="$outdir/biases.txt"} \
      "$PY" "$ATTACK" "$fmt" "$dataset" "$layers" "$csv" "$isize" "$lip" "$N" )
}

# ---- MNIST natural (gram-20 ref) ----
run_cex float32 mnist "$MNIST_L" "$MNIST_CSV" 28 "$REFS/dafny_mnist_gram20.json" "$OUT/cex_mnist_float32_numpy"
run_cex float16 mnist "$MNIST_L" "$MNIST_CSV" 28 "$REFS/dafny_mnist_gram20.json" "$OUT/cex_mnist_float16_numpy"
run_cex float64 mnist "$MNIST_L" "$MNIST_CSV" 28 "$REFS/dafny_mnist_gram20.json" "$OUT/cex_mnist_float64_numpy"

# ---- Fashion-MNIST natural (gram-13 ref) ----
run_cex float32 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float32_numpy"
run_cex float16 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float16_numpy"
run_cex float64 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float64_numpy"

# ---- CIFAR-10 natural (gram-12 ref; no float16) ----
run_cex float32 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float32_numpy"
run_cex float64 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float64_numpy"

# ---- Adversarially-biased models (float32; FP_BIAS per adversarial-bias-values) ----
run_cex float32 mnist         "$MNIST_L"   "$MNIST_CSV"   28 "$REFS/dafny_mnist_gram20.json"   "$OUT/cex_mnist_float32_biased_1e6_end_numpy"         1e6
run_cex float32 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float32_biased_3e6_end_numpy" 3e6
run_cex float32 cifar10       "$CIFAR_L"   "$CIFAR_CSV"   32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float32_biased_4e6_end_numpy"       4e6

echo "All numpy cex suites generated."
