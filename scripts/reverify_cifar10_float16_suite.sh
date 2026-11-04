#!/usr/bin/env bash
# Verify the natural CIFAR-10 float16 suite from regen_cifar10_float16_suite.sh with
# reverify_cex_suite.py (exact rational checks on the points cast to float16; must verify
# 30/30), as run_rq1_f16f64.sh in python-certifier did for the other five float16/float64
# suites. This suite is NOT installed into python-certifier/tests or certified: CIFAR-10 at
# float16 has n*u = 1.5 >= 1 at its input layer, where by policy the certifier is not
# evaluated (robust_certifier.check_rounding_preconditions); the paper reports the suite's
# radii in §2.2 only. Output: results/natural_grid_cex/reverify_cex_cifar10_float16_numpy.json.
# Exact invocation (2026-11-04): cd scripts && bash reverify_cifar10_float16_suite.sh 2>&1 | tee results/natural_grid_cex/reverify_cifar10_float16_suite.log
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPTS"
PC=~/git/python-certifier; REFS=$PC/models/precomputed; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python
SUITE=$SCRIPTS/results/natural_grid_cex/f16f64_suites_regen/cex_cifar10_float16_numpy
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
CIFAR_L="[512,256,128,128,128,128,128,128]"
env PYTHONPATH="$SCRIPTS" PYTHON_CERTIFIER="$PC" "$PY" reverify_cex_suite.py float16 cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 "$REFS/dafny_cifar10_gram12.json" --suite "$SUITE" | grep -v 'DeepFool\|^$' | tail -n 12
python3 -I -c "
import json,sys; s=json.load(open('$SCRIPTS/results/natural_grid_cex/reverify_cex_cifar10_float16_numpy.json'))['summary']
print('SUMMARY', json.dumps(s)); sys.exit(0 if s['n_verified_float32']==s['n_pairs']==30 else 1)"
