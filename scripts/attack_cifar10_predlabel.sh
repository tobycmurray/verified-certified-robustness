#!/usr/bin/env bash
# CIFAR-10 l2 attacks over ALL 10,000 test points against the numpy execution's predicted
# label (pgd_l2_robust_accuracy.py), for the paper's RQ2 partition of the
# test set into certified / attacked / neither. Same model, eps, steps and restarts as the
# POPL-response runs (results/pgd_l2_cifar10_eps0.141.json, apgd_...): PGD-l2 then the two
# AutoAttack gradient components. ART attacks at eps*(1-1e-5) and every adversarial is
# judged by the EXACT distance (see pgd_l2_robust_accuracy.py). Outputs
# results/{pgd,apgd}_l2_cifar10_eps0.141_predlabel_exact.json (+ _adv.npz adversarials).
# Exact invocation (2026-11-05, second run; the first run of the same day used the float64
# shrink-back verdict, results/*_predlabel.json, superseded):
#   cd scripts && nohup bash attack_cifar10_predlabel.sh > results/attack_cifar10_predlabel_exact.log 2>&1 &
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC=~/git/python-certifier; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
CIFAR_L="[512,256,128,128,128,128,128,128]"
cd "$SCRIPTS"
for atk in pgd apgd; do
  echo ">>> $atk ($(date))"
  env PYTHON_CERTIFIER="$PC" "$PY" pgd_l2_robust_accuracy.py cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 0.141 \
      --steps 100 --restarts 5 --attack "$atk" --inset 1e-5
done
echo "### done ($(date))"
