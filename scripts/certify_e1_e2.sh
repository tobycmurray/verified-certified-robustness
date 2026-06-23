#!/bin/bash
# Morning certification for the HIGGS/EMNIST revision experiments. float32 only.
#   E1: FP-cost-vs-width on HIGGS (full-data models {128,256,512,1024}, 10k test).
#   E2: breadth (HIGGS-512 full; EMNIST balanced full test; EMNIST byclass full test).
# For each model: exact .txt (make_certifier_format_from_model) -> float32 cex with
# labels (get_all_test_inputs) -> certify standard/hybrid-only/hybrid-meas (fp64
# norms, no Dafny ref -> L_real real baseline). Idempotent (skips existing outputs).
# Output: cert_morning/<tag>/{net.txt, cex.json, results_<mode>.json, *.log}
set -u
cd "$(dirname "$0")"
VCR="$(pwd)"
PC="$HOME/git/python-certifier"
PY="$VCR/cav2025-artifact-venv/bin/python3"
GRAM=20
OUT="$VCR/cert_morning"
mkdir -p "$OUT"

# certify_model tag dataset layers input_size eval_eps n_cex [HIGGS_N_TRAIN HIGGS_N_TEST]
certify_model () {
  local tag="$1" dataset="$2" layers="$3" isize="$4" eps="$5" ncex="$6" modes="$7" htr="${8:-}" hte="${9:-}"
  local csv="$VCR/sweep_results/$tag/model_weights_csv"
  local d="$OUT/$tag"; mkdir -p "$d"
  if [ ! -f "$csv/layer_0_weights.csv" ]; then echo "!! no weights for $tag"; return; fi
  if [ ! -f "$d/net.txt" ]; then
    echo ">>> $(date '+%H:%M:%S') [$tag] make .txt"
    (cd "$PC" && "$PY" make_certifier_format_from_model.py "$dataset" "$layers" "$isize" "$csv" "$d/net.txt") >"$d/make_txt.log" 2>&1 \
      || { echo "!! .txt failed for $tag"; tail -3 "$d/make_txt.log"; return; }
  fi
  if [ ! -f "$d/cex.json" ]; then
    echo ">>> $(date '+%H:%M:%S') [$tag] make cex (n=$ncex)"
    (cd "$d" && env PYTHONPATH="$VCR" ${htr:+HIGGS_N_TRAIN="$htr"} ${hte:+HIGGS_N_TEST="$hte"} \
       "$PY" "$VCR/get_all_test_inputs.py" float32 "$dataset" "$layers" "$csv" "$isize" "$d/cex.json" "$eps" "$ncex") >"$d/make_cex.log" 2>&1 \
      || { echo "!! cex failed for $tag"; tail -3 "$d/make_cex.log"; return; }
  fi
  for mode in $modes; do
    [ -f "$d/results_$mode.json" ] && continue
    echo ">>> $(date '+%H:%M:%S') [$tag] certify $mode"
    (cd "$PC" && "$PY" robust_certifier.py float32 "$d/net.txt" "$GRAM" "$d/cex.json" \
       --norm-method fp64 --mode "$mode" --json-output "$d/results_$mode.json") >"$d/cert_$mode.log" 2>&1 \
      || { echo "!! cert $mode failed for $tag"; tail -3 "$d/cert_$mode.log"; }
  done
  echo ">>> $(date '+%H:%M:%S') [$tag] DONE"
}

H="10500000"; HT="500000"
ALL="standard hybrid-only hybrid-meas"
# EMNIST: hybrid-meas is marginal on images (+0.2pp over hybrid-only, daytime) and
# pathologically slow per-input on many-class/deep nets (>4h on 18.8k) -> skip it.
STDHYB="standard hybrid-only"
# E1: HIGGS width sweep (full-data models), fixed 10k test subset, eps 0.1, all 3 modes
certify_model higgs_w128_d5_full  higgs "[128,128,128,128,128]"      1 0.1 10000 "$ALL" "$H" "$HT"
certify_model higgs_w256_d5_full  higgs "[256,256,256,256,256]"      1 0.1 10000 "$ALL" "$H" "$HT"
certify_model higgs_w512_d5_full  higgs "[512,512,512,512,512]"      1 0.1 10000 "$ALL" "$H" "$HT"
certify_model higgs_w1024_d5_full higgs "[1024,1024,1024,1024,1024]" 1 0.1 10000 "$ALL" "$H" "$HT"
# E2: EMNIST balanced (full ~18.8k) and byclass (10k subset; per-input keras hybrid is slow)
certify_model emnistbal_w512_d8_ep500 emnist/balanced "[512,512,512,512,512,512,512,512]" 28 0.3 20000 "$STDHYB"
certify_model emnistbyc_cifar         emnist/byclass  "[512,256,128,128,128,128,128,128]"  28 0.3 10000  "$STDHYB"

echo ">>> $(date '+%H:%M:%S') ALL CERT DONE"
"$PY" "$VCR/aggregate_cert.py" "$OUT"
