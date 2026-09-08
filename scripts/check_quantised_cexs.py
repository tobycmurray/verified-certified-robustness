#!/usr/bin/env python3
"""Re-judge an existing counter-example suite after snapping both points to the
8-bit pixel grid (k/255), i.e. restricting to inputs that are valid images.

Motivation (POPL'27 reviewer C, Q3): image pixels are integers 0..255, so a
counter-example (x0, x1) found by the continuous search may not correspond to
valid images. This script quantises both points of each stored counter-example
to the k/255 grid (rounded to nearest, clipped to [0,1], cast to the deployment
format) and re-runs the *same* judgement used to produce the suite --
attack_verified_certifier_nat_numpy.check_counter_example, i.e. classification
under the IEEE-754-compliant numpy execution and the certifier oracle against
the same Lipschitz reference -- on the quantised pair.

The model, format, bias amplification (FP_BIAS / FP_BIAS_POS / BIAS_OUTPUT env
vars) and Lipschitz reference are configured exactly as for the attack script,
so the CLI is the attack script's CLI plus the cex directory:

  python check_quantised_cexs.py float_format dataset INTERNAL_LAYER_SIZES \
      model_weights_csv_dir input_size lipschitz_json cex_dir [--out out.json]

Conventions inherited from the suite: in counter_examples.json, x1_file is the
CERTIFIED point (the paper's x_0) and x0_file the differently-classified point
within eps of it (the paper's x_1); check_counter_example(x0, x1) certifies x1's
label at radius ||x1 - x0||_2 and, if certified, grows the radius to the largest
certifiable one (max_eps). Run under cav2025-artifact-venv.
"""
import json
import os
import sys

import numpy as np

# ---- CLI: strip our own arguments, hand the attack script's CLI to the module ----
if len(sys.argv) < 8:
    print(__doc__)
    sys.exit(1)
out_path = None
if "--out" in sys.argv:
    i = sys.argv.index("--out")
    out_path = sys.argv[i + 1]
    del sys.argv[i:i + 2]
cex_dir = sys.argv.pop(7)
if out_path is None:
    out_path = os.path.join(cex_dir, "quantised_check.json")

# The attack module parses sys.argv at import (fmt dataset layers csv isize ref),
# builds the (possibly bias-amplified) Keras model, the compliant numpy model and
# the certifier oracle. Nothing else runs at import (main() is __main__-guarded).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_verified_certifier_nat_numpy as A  # noqa: E402

LEVELS = 255


def quantise(x):
    """Snap to the k/255 grid: nearest level, clipped to [0,1], deployment dtype."""
    x64 = np.asarray(x, dtype=np.float64)
    q = np.clip(np.rint(x64 * LEVELS), 0, LEVELS) / LEVELS
    return q.astype(A.NP_DT)


def on_grid(x):
    x64 = np.asarray(x, dtype=np.float64)
    return bool(np.all(np.abs(np.rint(x64 * LEVELS) / LEVELS - x64) <= 1e-6))


def l2(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()
                                - np.asarray(b, dtype=np.float64).ravel()))


def judge(x0, x1):
    ok, max_eps = A.check_counter_example(x0, x1)
    return bool(ok), (float(max_eps) if ok else None)


with open(os.path.join(cex_dir, "counter_examples.json")) as f:
    entries = [e for e in json.load(f) if "_metadata" not in e]

records = []
for e in entries:
    x0 = np.load(e["x0_file"])
    x1 = np.load(e["x1_file"])
    orig_ok, orig_eps = judge(x0, x1)
    q0, q1 = quantise(x0), quantise(x1)
    q_ok, q_eps = judge(q0, q1)
    rec = {
        "index": e.get("index"),
        "stored_max_eps": float(e["max_eps"]),
        "stored_dist": float(e["dist"]),
        "recheck_is_cex": orig_ok,
        "recheck_max_eps": orig_eps,
        "x0_on_grid": on_grid(x0),
        "x1_on_grid": on_grid(x1),
        "x0_pixels_moved": int(np.sum(q0 != np.asarray(x0, dtype=A.NP_DT).reshape(q0.shape))),
        "x1_pixels_moved": int(np.sum(q1 != np.asarray(x1, dtype=A.NP_DT).reshape(q1.shape))),
        "dist_quantised": l2(q0, q1),
        "labels_differ_quantised": int(np.argmax(A.np_logits(q0))) != int(np.argmax(A.np_logits(q1))),
        "quantised_is_cex": q_ok,
        "quantised_max_eps": q_eps,
    }
    records.append(rec)
    print(f"idx {rec['index']}: stored eps={rec['stored_max_eps']:.4g} recheck={orig_ok} "
          f"| quantised: dist={rec['dist_quantised']:.4g} labels_differ={rec['labels_differ_quantised']} "
          f"cex={q_ok} max_eps={q_eps}", flush=True)

n = len(records)
summary = {
    "cex_dir": cex_dir,
    "format": A.fmt,
    "n": n,
    "recheck_cex": sum(r["recheck_is_cex"] for r in records),
    "quantised_labels_differ": sum(r["labels_differ_quantised"] for r in records),
    "quantised_cex": sum(r["quantised_is_cex"] for r in records),
    "quantised_max_eps_max": max([r["quantised_max_eps"] for r in records if r["quantised_max_eps"]] or [None]),
    "quantised_max_eps_min": min([r["quantised_max_eps"] for r in records if r["quantised_max_eps"]] or [None]),
    "quantised_cex_gt_0.1": sum(1 for r in records if r["quantised_max_eps"] and r["quantised_max_eps"] > 0.1),
}
print("\nSUMMARY:", json.dumps(summary, indent=1))
with open(out_path, "w") as f:
    json.dump({"summary": summary, "records": records}, f, indent=1)
print(f"Wrote {out_path}")
