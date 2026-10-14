#!/usr/bin/env bash
# Regenerate ONLY the three natural float32 continuous-search suites, with the
# attack script's model-format cast fix (points stored as float32-representable
# vectors), into a fresh directory so the committed python-certifier/tests suites
# are untouched until replaced deliberately. Same invocation as generate_numpy_cexs.sh.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC=~/git/python-certifier; REFS=$PC/models/precomputed; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python; ATTACK=$SCRIPTS/attack_verified_certifier_nat_numpy.py; N=30
OUT=${CEX_OUT_DIR:-$SCRIPTS/results/natural_grid_cex/float32_suites_regen}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MNIST_CSV="$CAV/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"
FASHION_CSV="$CAV/2025-01-30_10:58:01-fashion_mnist/model_weights_epsilon_0.26_[256,128,128,128,128,128,128,128,128,128,128,128]_500"
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
run_cex () {  # fmt dataset layers csv isize lipref outdir
  local outdir="$7"; mkdir -p "$outdir"; echo ">>> $(basename "$outdir")"
  ( cd "$outdir" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" "$PY" "$ATTACK" "$1" "$2" "$3" "$4" "$5" "$6" "$N" )
}
run_cex float32 mnist "[128,128,128,128,128,128,128,128]" "$MNIST_CSV" 28 "$REFS/dafny_mnist_gram20.json" "$OUT/cex_mnist_float32_numpy"
run_cex float32 fashion_mnist "[256,128,128,128,128,128,128,128,128,128,128,128]" "$FASHION_CSV" 28 "$REFS/dafny_fashion_gram13.json" "$OUT/cex_fashion_mnist_float32_numpy"
run_cex float32 cifar10 "[512,256,128,128,128,128,128,128]" "$CIFAR_CSV" 32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float32_numpy"
echo "### done (float32 suites regen)"
