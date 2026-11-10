#!/usr/bin/env bash
# Exact re-verification of every shipped continuous-search counterexample suite (artifact
# MANIFEST A2.1/A2.2, DESIGN §7.6): reverify_cex_suite.py on the three image models at
# float32, float16 and float64 (nine suites). Each run checks, in exact rational arithmetic
# on the points cast to the suite's format, that the labels differ under the compliant numpy
# execution, that ||x' - x||_2 <= eps exactly, and that the certifier's real-arithmetic margin
# test passes at eps (so the pair is a counterexample to the real-arithmetic check); a suite
# passes when all 30 pairs verify. The label flips are the host's own observation (recorded);
# the distances and radii are exact.
# Suites: python-certifier/tests/cex_<model>_<fmt>_numpy for eight of them; the CIFAR-10
# float16 suite (not installed in tests/, the certifier refuses CIFAR-10 at float16 by policy)
# is results/natural_grid_cex/f16f64_suites_regen/cex_cifar10_float16_numpy. The command
# lines are those of python-certifier/tests/run_rq1_f16f64.sh:36-40 and
# reverify_cifar10_float16_suite.sh:18, with the model table read from
# python-certifier/models/registry.tsv (registry.sh; REGISTRY overrides the path).
# The three adversarially biased suites (cex_<model>_float32_biased_<B>_end_numpy, registry rows
# <model>_biased_<B>_end) are re-verified the same way with the biased model rebuilt from the registry's
# bias and bias_pos (FP_BIAS / FP_BIAS_POS / BIAS_OUTPUT of the attack script); the rebuilt bias vectors
# are compared with the suite's biases.txt (summary field biases_match_suite). Selection `biased` runs
# them, `natural` the nine natural suites, `all` both (artifact run 1 passes `natural`: the shipped
# biased suites are covered by the certifier runs; run 3-cex-regen re-verifies all of its regenerated suites).
# Usage:  bash reverify_all_suites.sh [float32|float16|float64|biased|natural|all]     (default all)
#   OUT_DIR   where the per-suite JSONs go (default results/reverify_suites/)
#   SUITES_ROOT  where the cex_<model>_<fmt>_numpy suites are (default python-certifier/tests; a
#             regeneration run points it at CEX_OUT_DIR of generate_numpy_cexs.sh)
#   PYTHON_CERTIFIER, PY, REGISTRY as in generate_numpy_cexs.sh
# Writes OUT_DIR/reverify_<suite>.json per suite (summary + per-pair verdicts) and
# OUT_DIR/summary.tsv (suite, n_pairs, n_verified, eps_min, eps_max); exits non-zero if any
# suite verifies fewer than 30/30. Each suite takes seconds to a minute (TensorFlow start-up
# and the keras test-set load dominate).
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VCRS_ROOT="$(cd "$SCRIPTS/.." && pwd)"
PC="${PYTHON_CERTIFIER:-$VCRS_ROOT/../python-certifier}"
TESTS="$PC/tests"
REFS="$PC/models/precomputed"
PY="${PY:-$SCRIPTS/cav2025-artifact-venv/bin/python}"
OUT_DIR="${OUT_DIR:-$SCRIPTS/results/reverify_suites}"
WHICH="${1:-all}"
# SUITES_ROOT: the directory holding the cex_<model>_<fmt>_numpy suites (default: the shipped ones in
# python-certifier/tests). A regeneration run (artifact run 3-cex-regen, CEX_OUT_DIR of
# generate_numpy_cexs.sh) points it at its own output; a suite absent there is skipped.
SUITES_ROOT="${SUITES_ROOT:-$TESTS}"
F16_CIFAR_SUITE="$SCRIPTS/results/natural_grid_cex/f16f64_suites_regen/cex_cifar10_float16_numpy"
[ "$SUITES_ROOT" = "$TESTS" ] || F16_CIFAR_SUITE="$SUITES_ROOT/cex_cifar10_float16_numpy"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
source "$SCRIPTS/registry.sh"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf 'suite\tn_pairs\tn_verified\teps_min\teps_max\n' > "$SUMMARY"
fail=0

reverify () {  # fmt key suite_dir
  local fmt="$1" key="$2" suite="$3" dataset layers csv isize lip name bias pos
  dataset=$(reg "$key" dataset); layers=$(reg "$key" layers); csv="$VCRS_ROOT/$(reg "$key" weights_csv)"
  isize=$(reg "$key" isize); lip="$REFS/$(reg "$key" ref_rq1)"
  bias=$(reg "$key" bias); pos=$(reg "$key" bias_pos)
  name=$(basename "$suite")
  if [ ! -s "$suite/counter_examples.json" ]; then echo "##### $name absent under $SUITES_ROOT, skip"; return 0; fi
  echo "##### reverify $name ($(date))"
  ( cd "$SCRIPTS" && env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" REVERIFY_OUT_DIR="$OUT_DIR" \
      ${bias:+FP_BIAS="$bias"} ${bias:+FP_BIAS_POS="$pos"} ${bias:+BIAS_OUTPUT="$OUT_DIR/biases_rebuilt_$name.txt"} \
      "$PY" reverify_cex_suite.py "$fmt" "$dataset" "$layers" "$csv" "$isize" "$lip" --suite "$suite" ) \
    | grep -v 'DeepFool\|^$' | tail -n 12
  python3 -I - "$OUT_DIR/reverify_$name.json" "$SUMMARY" <<'PYEOF' || fail=1
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
print("SUMMARY", json.dumps(s))
with open(sys.argv[2], "a") as f:
    f.write(f"{s['suite']}\t{s['n_pairs']}\t{s['n_verified_float32']}\t{s['eps_min']!r}\t{s['eps_max']!r}\n")
sys.exit(0 if s["n_verified_float32"] == s["n_pairs"] == 30 else 1)
PYEOF
}

want () {  # all = natural + biased; natural = float32 + float16 + float64
  case "$WHICH" in all) return 0 ;; natural) [ "$1" != biased ] ;; *) [ "$WHICH" = "$1" ] ;; esac
}
if want float32; then
  reverify float32 mnist         "$SUITES_ROOT/cex_mnist_float32_numpy"
  reverify float32 fashion_mnist "$SUITES_ROOT/cex_fashion_mnist_float32_numpy"
  reverify float32 cifar10       "$SUITES_ROOT/cex_cifar10_float32_numpy"
fi
if want float16; then
  reverify float16 mnist         "$SUITES_ROOT/cex_mnist_float16_numpy"
  reverify float16 fashion_mnist "$SUITES_ROOT/cex_fashion_mnist_float16_numpy"
  reverify float16 cifar10       "$F16_CIFAR_SUITE"
fi
if want float64; then
  reverify float64 mnist         "$SUITES_ROOT/cex_mnist_float64_numpy"
  reverify float64 fashion_mnist "$SUITES_ROOT/cex_fashion_mnist_float64_numpy"
  reverify float64 cifar10       "$SUITES_ROOT/cex_cifar10_float64_numpy"
fi
if want biased; then
  reverify float32 mnist_biased_1e6_end         "$SUITES_ROOT/cex_mnist_float32_biased_1e6_end_numpy"
  reverify float32 fashion_mnist_biased_3e6_end "$SUITES_ROOT/cex_fashion_mnist_float32_biased_3e6_end_numpy"
  reverify float32 cifar10_biased_4e6_end       "$SUITES_ROOT/cex_cifar10_float32_biased_4e6_end_numpy"
fi
case "$WHICH" in float32|float16|float64|biased|natural|all) ;; *) echo "unknown selection: $WHICH" >&2; exit 2 ;; esac
echo "##### done ($WHICH, $(date)); summary in $SUMMARY"
cat "$SUMMARY"
exit $fail
