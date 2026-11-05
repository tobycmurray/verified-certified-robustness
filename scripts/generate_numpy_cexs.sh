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
# The model table (weights CSV, layer list, input size, RQ1 Lipschitz ref, bias) is read
# from python-certifier/models/registry.tsv (registry.sh; REGISTRY overrides the path).
# The Lipschitz refs are the same Dafny reference JSONs the certifier certifies
# against, so max_eps is calibrated to the exact L bounds used in the cex tests.
# Output dirs are the _numpy siblings of the existing Keras cex dirs. Idempotent:
# a case whose counter_examples.json already exists is skipped (delete it to
# regenerate). Runs under cav2025-artifact-venv (ART/DeepFool + gmpy2).
# DRY_RUN=1 prints the exact command line of every row and runs nothing.
#
# This is the single continuous-search entry point (artifact DESIGN §7.2). The authors'
# actual regeneration runs used its per-subset copies regen_float32_suites.sh,
# regen_float16_float64_suites.sh and regen_cifar10_float16_suite.sh (same command lines,
# fresh output directories; now superseded, kept because the logs reference them).
# The SECOND entry point, not merged here, is the grid search on the adversarially biased
# models over their full test sets at the standard radii: run_natural_grid_cexs.sh
# (natural_grid_cexs.py; outputs results/natural_grid_cex/natgrid_<model>_biased.json,
# the 24 / 204 / 370 refuted points of §2.2).
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # verified-certified-robustness/scripts
VCRS_ROOT="$(cd "$SCRIPTS/.." && pwd)"
PC="${PYTHON_CERTIFIER:-$VCRS_ROOT/../python-certifier}"
TESTS="$PC/tests"
REFS="$PC/models/precomputed"
PY="${PY:-$SCRIPTS/cav2025-artifact-venv/bin/python}"
ATTACK="$SCRIPTS/attack_verified_certifier_nat_numpy.py"
N="${N:-30}"
# Output root for the cex_*_numpy dirs (default: python-certifier/tests, where
# run_tests.sh consumes them). Override with CEX_OUT_DIR for a regeneration run
# that must not collide with committed suites (e.g. the artifact container,
# whose baked tests/ already holds them and would trigger skip-if-exists).
OUT="${CEX_OUT_DIR:-$TESTS}"
mkdir -p "$OUT"

source "$SCRIPTS/registry.sh"   # reg KEY COLUMN (python-certifier/models/registry.tsv)
mdl () {  # mdl KEY: sets DATASET LAYERS CSV ISIZE LIP BIAS BIAS_POS from the registry row KEY
  DATASET=$(reg "$1" dataset); LAYERS=$(reg "$1" layers); CSV="$VCRS_ROOT/$(reg "$1" weights_csv)"
  ISIZE=$(reg "$1" isize); LIP="$REFS/$(reg "$1" ref_rq1)"; BIAS=$(reg "$1" bias); BIAS_POS=$(reg "$1" bias_pos)
}

run_cex () {  # fmt dataset layers csv isize lipref outdir [fp_bias [fp_bias_pos]]
  local fmt="$1" dataset="$2" layers="$3" csv="$4" isize="$5" lip="$6" outdir="$7" fp_bias="${8:-}" pos="${9:-end}"
  if [ "${DRY_RUN:-0}" = 1 ]; then   # print the exact command line of this row, run nothing
    printf '%s' "cd $(printf '%q' "$outdir") && env PYTHONPATH=$(printf '%q' "$SCRIPTS") PYTHON_CERTIFIER=$(printf '%q' "$PC")"
    [ -n "$fp_bias" ] && printf ' FP_BIAS=%q FP_BIAS_POS=%q BIAS_OUTPUT=%q' "$fp_bias" "$pos" "$outdir/biases.txt"
    printf ' %q' "$PY" "$ATTACK" "$fmt" "$dataset" "$layers" "$csv" "$isize" "$lip" "$N"; printf '\n'
    return 0
  fi
  if [ -s "$outdir/counter_examples.json" ]; then
    echo "  exists, skip: $(basename "$outdir")"; return 0
  fi
  mkdir -p "$outdir"
  echo ">>> $(basename "$outdir") (fmt=$fmt${fp_bias:+, FP_BIAS=$fp_bias $pos})"
  ( cd "$outdir" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" \
      ${fp_bias:+FP_BIAS="$fp_bias"} ${fp_bias:+FP_BIAS_POS="$pos"} \
      ${fp_bias:+BIAS_OUTPUT="$outdir/biases.txt"} \
      "$PY" "$ATTACK" "$fmt" "$dataset" "$layers" "$csv" "$isize" "$lip" "$N" )
}

# ---- MNIST natural (gram-20 ref) ----
mdl mnist
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_mnist_float32_numpy"
run_cex float16 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_mnist_float16_numpy"
run_cex float64 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_mnist_float64_numpy"

# ---- Fashion-MNIST natural (gram-13 ref) ----
mdl fashion_mnist
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_fashion_mnist_float32_numpy"
run_cex float16 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_fashion_mnist_float16_numpy"
run_cex float64 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_fashion_mnist_float64_numpy"

# ---- CIFAR-10 natural (gram-12 ref; no float16) ----
mdl cifar10
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_cifar10_float32_numpy"
run_cex float64 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_cifar10_float64_numpy"

# ---- Adversarially-biased models (float32; FP_BIAS / FP_BIAS_POS from the registry) ----
mdl mnist_biased_1e6_end
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_mnist_float32_biased_1e6_end_numpy" "$BIAS" "$BIAS_POS"
mdl fashion_mnist_biased_3e6_end
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_fashion_mnist_float32_biased_3e6_end_numpy" "$BIAS" "$BIAS_POS"
mdl cifar10_biased_4e6_end
run_cex float32 "$DATASET" "$LAYERS" "$CSV" "$ISIZE" "$LIP" "$OUT/cex_cifar10_float32_biased_4e6_end_numpy" "$BIAS" "$BIAS_POS"

echo "All numpy cex suites generated."
