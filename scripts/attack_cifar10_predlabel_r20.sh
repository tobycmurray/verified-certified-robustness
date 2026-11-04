#!/usr/bin/env bash
# Stage 2 of the CIFAR-10 all-points predicted-label attack: PGD-l2 again with 20 random restarts
# (the 5-restart PGD and APGD runs of attack_cifar10_predlabel.sh overlapped only partly, so many
# broken points are marginal and more restarts should break more). The union over all runs is
# the reported "broken" set; every adversarial is re-judged individually, so the union is sound.
# Output results/pgd_l2_cifar10_eps0.141_predlabel_r20.json.
# Exact invocation (2026-11-05):  cd scripts && nohup bash attack_cifar10_predlabel_r20.sh > results/attack_cifar10_predlabel_r20.log 2>&1 &
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PC=~/git/python-certifier; CAV="$SCRIPTS/../cav2025-models"
PY=$SCRIPTS/cav2025-artifact-venv/bin/python
CIFAR_CSV="$CAV/2025-01-28_20:39:32-cifar10/model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800"
CIFAR_L="[512,256,128,128,128,128,128,128]"
cd "$SCRIPTS"
env PYTHON_CERTIFIER="$PC" "$PY" pgd_l2_robust_accuracy.py cifar10 "$CIFAR_L" "$CIFAR_CSV" 32 0.141 \
    --steps 100 --restarts 20 --attack pgd --out results/pgd_l2_cifar10_eps0.141_predlabel_r20.json
echo "### done ($(date))"
