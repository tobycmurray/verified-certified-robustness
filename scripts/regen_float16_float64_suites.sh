#!/usr/bin/env bash
# Regenerate the five natural float16/float64 continuous-search suites (MNIST and
# Fashion MNIST at float16; MNIST, Fashion MNIST and CIFAR-10 at float64; CIFAR-10 at
# float16, which the certifier by policy refuses (n*u = 1.5 >= 1), is generated separately by
# regen_cifar10_float16_suite.sh for §2.2 only) with the attack script's two fixes: points cast
# to the model's format (3cb2ba8) and weights loaded at the float32 policy and cast to
# the target format (this commit; the July 2026 suites parsed the CSV decimals straight
# to the target format, so the certifier's hybrid passes refused every float64
# instance). Into a fresh directory so the committed python-certifier/tests suites are
# untouched until replaced deliberately. Same invocation as regen_float32_suites.sh.
# Exact invocation (2026-11-05):  cd scripts && nohup bash regen_float16_float64_suites.sh > results/natural_grid_cex/f16f64_suites_regen.log 2>&1 &
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC=~/git/python-certifier; REFS=$PC/models/precomputed; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python; ATTACK=$SCRIPTS/attack_verified_certifier_nat_numpy.py; N=30
OUT=${CEX_OUT_DIR:-$SCRIPTS/results/natural_grid_cex/f16f64_suites_regen}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
MNIST_L="[128,128,128,128,128,128,128,128]"
FASHION_L="[256,128,128,128,128,128,128,128,128,128,128,128]"
CIFAR_L="[512,256,128,128,128,128,128,128]"
run_cex () {  # fmt dataset layers csv isize lipref outdir
  local outdir="$7"; mkdir -p "$outdir"; echo ">>> $(basename "$outdir") ($(date))"
  ( cd "$outdir" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" "$PY" "$ATTACK" "$1" "$2" "$3" "$4" "$5" "$6" "$N" )
}
run_cex float16 mnist         "$MNIST_L"   "$MNIST_CSV"   28 "$REFS/dafny_mnist_gram20.json"   "$OUT/cex_mnist_float16_numpy"
run_cex float16 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float16_numpy"
run_cex float64 mnist         "$MNIST_L"   "$MNIST_CSV"   28 "$REFS/dafny_mnist_gram20.json"   "$OUT/cex_mnist_float64_numpy"
run_cex float64 fashion_mnist "$FASHION_L" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float64_numpy"
run_cex float64 cifar10       "$CIFAR_L"   "$CIFAR_CSV"   32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float64_numpy"
echo "### done (float16/float64 suites regen, $(date))"
