#!/usr/bin/env bash
# SUPERSEDED (2026-11-05): generate_numpy_cexs.sh is the single continuous-search entry point
# (same command lines, model table from python-certifier/models/registry.tsv; CEX_OUT_DIR for a
# fresh output directory; note generate_numpy_cexs.sh has no CIFAR-10 float16 row, by policy --
# that suite is §2.2's only). Kept unchanged because
# results/natural_grid_cex/cifar10_float16_suite_regen.log references this script as the authors' run.
# Generate the natural CIFAR-10 float16 continuous-search suite, the one suite that
# regen_float16_float64_suites.sh left out. CIFAR-10 at float16 has n*u = 1.5 >= 1 at its
# 3072-wide input layer, where the certifier's bounds are sound but useless, so by policy the
# python certifier refuses it (robust_certifier.check_rounding_preconditions) and the paper
# does not evaluate it; the suite exists so that §2.2 can report the magnitude of the float16
# counterexamples for all three models. It is verified by reverify_cifar10_float16_suite.sh
# and NOT installed into python-certifier/tests. Same attack script, fixes and invocation as
# regen_float16_float64_suites.sh (points cast to float16; weights loaded at the float32
# policy and cast down to float16, the deployment rounding).
# Exact invocation (2026-11-04):  cd scripts && nohup bash regen_cifar10_float16_suite.sh > results/natural_grid_cex/cifar10_float16_suite_regen.log 2>&1 &
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC=~/git/python-certifier; REFS=$PC/models/precomputed; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python; ATTACK=$SCRIPTS/attack_verified_certifier_nat_numpy.py; N=30
OUT=${CEX_OUT_DIR:-$SCRIPTS/results/natural_grid_cex/f16f64_suites_regen}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
CIFAR_L="[512,256,128,128,128,128,128,128]"
run_cex () {  # fmt dataset layers csv isize lipref outdir
  local outdir="$7"; mkdir -p "$outdir"; echo ">>> $(basename "$outdir") ($(date))"
  ( cd "$outdir" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" "$PY" "$ATTACK" "$1" "$2" "$3" "$4" "$5" "$6" "$N" )
}
run_cex float16 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 "$REFS/dafny_cifar10_gram12.json" "$OUT/cex_cifar10_float16_numpy"
echo "### done (CIFAR-10 float16 suite regen, $(date))"
