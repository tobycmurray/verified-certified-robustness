#!/bin/bash
# Seed-variance sweep for gloro training (response to arXiv:2601.13303, Le & Cao 2026).
#
# Le & Cao report extreme certified-robustness variance across training seeds, but
# measure it on STANDARD-trained models at knife-edge epsilons chosen to span the
# certified/uncertified transition. This sweep measures the same quantity in the
# regime their survey targets: certified (gloro) training, evaluated at the
# trained-for epsilon, on this paper's RQ4 configs, with 10 seeds (10..100 step 10,
# mirroring their design). Records per-seed clean accuracy, gloro VRA and
# robustness on the FULL test set (not 100 inputs). Each run also saves
# model_weights_csv, so the FP-sound certifier can be run per seed as a separate
# downstream step (same layering as sweep_higgs_emnist.sh).
#
# Usage:
#   ./seed_variance_sweep.sh higgs    # 10 seeds x HIGGS [512]x5      (~12 min each, ~2h)
#   ./seed_variance_sweep.sh emnist   # 10 seeds x EMNIST-bal [512]x8 (~19 min each, ~3h)
#   ./seed_variance_sweep.sh stats    # per-config n/mean/stddev/min/max from summary.tsv
#
# Do NOT run concurrently with sweep_higgs_emnist.sh (both use scripts/ as scratch
# for model_weights_csv etc.). Completed (tag,seed) runs are skipped, so the script
# is safe to kill and re-run.

set -u
cd "$(dirname "$0")"
PY="./cav2025-artifact-venv/bin/python3"
RESULTS="seed_variance_results"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.tsv"
if [ ! -f "$SUMMARY" ]; then
  printf "tag\tseed\tdataset\tlayers\ttrain_eps\teval_eps\tepochs\tbatch\tn_train\tclean_acc\tvra\trobustness\texit\tseconds\n" > "$SUMMARY"
fi

SEEDS="10 20 30 40 50 60 70 80 90 100"

run_cfg () {
  # tag seed dataset train_eps layers epochs batch eval_eps input_size
  local tag="$1" seed="$2" dataset="$3" train_eps="$4" layers="$5" epochs="$6" batch="$7" eval_eps="$8" input_size="$9"
  local outdir="$RESULTS/${tag}_s${seed}"
  if [ -f "$outdir/gloro_model_results.json" ]; then
    echo ">>> SKIP ${tag}_s${seed} (already done)"; return
  fi
  echo ">>> $(date '+%H:%M:%S') RUN ${tag}_s${seed} : $dataset $layers eps=$train_eps/$eval_eps ep=$epochs bs=$batch ntrain=${HIGGS_N_TRAIN:-NA}"
  rm -rf model_weights_csv model.keras gloro.summary layer_*_weights.npz 2>/dev/null
  mkdir -p "$outdir"
  local t0; t0=$(date +%s)
  GLORO_SEED="$seed" PYTHONPATH=. "$PY" train_gloro.py "$dataset" "$train_eps" "$layers" "$epochs" "$batch" "$eval_eps" "$input_size" > "$outdir/train.log" 2>&1
  local ec=$?
  local t1; t1=$(date +%s); local secs=$((t1 - t0))
  [ -d model_weights_csv ] && mv model_weights_csv "$outdir/"
  [ -f "$outdir/model_weights_csv/gloro_model_results.json" ] && cp "$outdir/model_weights_csv/gloro_model_results.json" "$outdir/"
  [ -f higgs_standardization.npz ] && cp higgs_standardization.npz "$outdir/" 2>/dev/null
  local acc vra rob
  acc=$(grep -a '"accuracy"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  vra=$(grep -a '"vra"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  rob=$(grep -a '"robustness"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$tag" "$seed" "$dataset" "$layers" "$train_eps" "$eval_eps" "$epochs" "$batch" "${HIGGS_N_TRAIN:-NA}" "$acc" "$vra" "$rob" "$ec" "$secs" >> "$SUMMARY"
  echo ">>> $(date '+%H:%M:%S') DONE ${tag}_s${seed} ec=$ec ${secs}s acc=$acc vra=$vra"
}

MODE="${1:-}"

case "$MODE" in
  higgs)
    # Same config as sweep_higgs_emnist.sh higgs_w512_d5 (the RQ4 HIGGS-512 point).
    export HIGGS_N_TRAIN=500000 HIGGS_N_TEST=500000
    for s in $SEEDS; do
      run_cfg higgs_w512_d5 "$s" higgs 0.1 "[512,512,512,512,512]" 40 512 0.1 1
    done
    unset HIGGS_N_TRAIN HIGGS_N_TEST
    ;;
  emnist)
    # Same config as sweep_higgs_emnist.sh emnistbal_w512_d8 (the RQ4 EMNIST point).
    for s in $SEEDS; do
      run_cfg emnistbal_w512_d8 "$s" emnist/balanced 0.4 "[512,512,512,512,512,512,512,512]" 100 256 0.3 28
    done
    ;;
  stats)
    "$PY" - "$SUMMARY" <<'EOF'
import csv, math, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
by_tag = defaultdict(list)
for r in rows:
    if r['exit'] == '0':
        by_tag[r['tag']].append(r)
for tag, rs in by_tag.items():
    print(f"{tag} (n={len(rs)}, seeds={','.join(r['seed'] for r in rs)})")
    for metric in ('clean_acc', 'vra', 'robustness'):
        vals = [float(r[metric]) * 100 for r in rs]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        print(f"  {metric:10s} mean={mean:6.2f}pp  stddev={std:5.2f}pp  min={min(vals):6.2f}  max={max(vals):6.2f}")
EOF
    ;;
  *)
    echo "Usage: $0 {higgs|emnist|stats}"; exit 1
    ;;
esac
