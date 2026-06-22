#!/bin/bash
# Width/depth sweep for HIGGS + EMNIST gloro models (branch higgs-gloro).
#
# Trains a series of gloro models, saving each model's weights + gloro VRA so they
# can later be run through the FP-sound certifier. NOTE: certification (norm compute
# + cert time, the actual scalability evidence) is a SEPARATE downstream step; this
# script only TRAINS and records the gloro-computed VRA ceiling.
#
# Usage:
#   ./sweep_higgs_emnist.sh now        # ~3h CPU subset (fits an AC-power window)
#   ./sweep_higgs_emnist.sh overnight  # heavy configs: EMNIST 1024-wide, byclass,
#                                       # HIGGS 1024-wide on the full 11M, flagships
#
# Output per config: sweep_results/<tag>/{model_weights_csv/, train.log,
#                    gloro_model_results.json, higgs_standardization.npz}
# Summary row per config appended to sweep_results/summary.tsv.
#
# Each config is independent and skipped if already done, so the script is safe to
# kill (e.g. AC power ends) and re-run; completed configs are preserved.

set -u
cd "$(dirname "$0")"
PY="./cav2025-artifact-venv/bin/python3"
RESULTS="sweep_results"
mkdir -p "$RESULTS"
SUMMARY="$RESULTS/summary.tsv"
if [ ! -f "$SUMMARY" ]; then
  printf "tag\tdataset\tlayers\ttrain_eps\teval_eps\tepochs\tbatch\tn_train\tclean_acc\tvra\trobustness\texit\tseconds\n" > "$SUMMARY"
fi

run_cfg () {
  # tag dataset train_eps layers epochs batch eval_eps input_size
  local tag="$1" dataset="$2" train_eps="$3" layers="$4" epochs="$5" batch="$6" eval_eps="$7" input_size="$8"
  local outdir="$RESULTS/$tag"
  if [ -f "$outdir/gloro_model_results.json" ]; then
    echo ">>> SKIP $tag (already done)"; return
  fi
  echo ">>> $(date '+%H:%M:%S') RUN $tag : $dataset $layers eps=$train_eps/$eval_eps ep=$epochs bs=$batch ntrain=${HIGGS_N_TRAIN:-NA}"
  rm -rf model_weights_csv model.keras gloro.summary layer_*_weights.npz 2>/dev/null
  mkdir -p "$outdir"
  local t0; t0=$(date +%s)
  PYTHONPATH=. "$PY" train_gloro.py "$dataset" "$train_eps" "$layers" "$epochs" "$batch" "$eval_eps" "$input_size" > "$outdir/train.log" 2>&1
  local ec=$?
  local t1; t1=$(date +%s); local secs=$((t1 - t0))
  # The CSV is the canonical weights artifact used by all downstream tooling
  # (make_certifier_format*, load_and_set_weights, get_all_test_inputs.py).
  [ -d model_weights_csv ] && mv model_weights_csv "$outdir/"
  [ -f "$outdir/model_weights_csv/gloro_model_results.json" ] && cp "$outdir/model_weights_csv/gloro_model_results.json" "$outdir/"
  [ -f higgs_standardization.npz ] && cp higgs_standardization.npz "$outdir/" 2>/dev/null
  local acc vra rob
  acc=$(grep -a '"accuracy"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  vra=$(grep -a '"vra"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  rob=$(grep -a '"robustness"' "$outdir/gloro_model_results.json" 2>/dev/null | head -1 | grep -oE '[0-9.]+' | head -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$tag" "$dataset" "$layers" "$train_eps" "$eval_eps" "$epochs" "$batch" "${HIGGS_N_TRAIN:-NA}" "$acc" "$vra" "$rob" "$ec" "$secs" >> "$SUMMARY"
  echo ">>> $(date '+%H:%M:%S') DONE $tag ec=$ec ${secs}s acc=$acc vra=$vra"
}

MODE="${1:-now}"

if [ "$MODE" = "now" ]; then
  # Only models whose first internal layer >= 512 (the "beyond CIFAR-10's 512"
  # frontier); sub-512 widths add no scalability novelty and are covered by pilots.
  # Ordered headline-first so the only past-512 config (HIGGS 1024-wide) is
  # guaranteed to complete even if CPU times run long.
  export HIGGS_N_TRAIN=500000 HIGGS_N_TEST=500000
  run_cfg higgs_w1024_d5 higgs 0.1 "[1024,1024,1024,1024,1024]" 40 512 0.1 1
  unset HIGGS_N_TRAIN HIGGS_N_TEST
  run_cfg emnistbal_w512_d8 emnist/balanced 0.4 "[512,512,512,512,512,512,512,512]" 100 256 0.3 28
  export HIGGS_N_TRAIN=500000 HIGGS_N_TEST=500000
  run_cfg higgs_w512_d5  higgs 0.1 "[512,512,512,512,512]"      40 512 0.1 1
  unset HIGGS_N_TRAIN HIGGS_N_TEST
fi

if [ "$MODE" = "overnight" ]; then
  # EMNIST balanced 1024-wide (>512 point, full epochs)
  run_cfg emnistbal_w1024_d8 emnist/balanced 0.4 "[1024,1024,1024,1024,1024,1024,1024,1024]" 300 256 0.3 28
  # EMNIST byclass headline (62 classes, CIFAR-style widths)
  run_cfg emnistbyc_cifar   emnist/byclass  0.4 "[512,256,128,128,128,128,128,128]"        300 256 0.3 28
  # HIGGS 1024-wide on the full 11M (10.5M train / 500k test)
  export HIGGS_N_TRAIN=10500000 HIGGS_N_TEST=500000
  run_cfg higgs_w1024_d5_full higgs 0.1 "[1024,1024,1024,1024,1024]" 30 1024 0.1 1
  unset HIGGS_N_TRAIN HIGGS_N_TEST
  # Optional flagship (very slow on CPU; uncomment to include):
  # run_cfg emnistbyc_w2048 emnist/byclass 0.4 "[2048,1024,1024,512,512,256,256,128]" 300 256 0.3 28
fi

echo "ALL DONE ($MODE). Summary:"; column -t -s $'\t' "$SUMMARY"
