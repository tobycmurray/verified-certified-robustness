#!/usr/bin/env python3
"""Exact re-verification of the saved CIFAR-10 adversarials (artifact MANIFEST B8.2).

pgd_l2_robust_accuracy.py saves, beside each results JSON, an `_adv.npz` with the
adversarial point of every test point whose label flipped (`idx`: test-set indices,
`x_adv`: the float32 adversarials, flattened). This script re-checks each saved
adversarial independently of the attack:
  * exact l2 distance: ||x_adv - x||_2^2 <= eps^2 in exact rational arithmetic
    (fractions.Fraction over the float32 values; eps the exact decimal string, as
    sq_dist_exact in pgd_l2_robust_accuracy.py and reverify_cex_suite.py judge);
  * every coordinate of x_adv lies in [0, 1];
  * (recorded, host-specific) whether the label flips under the IEEE-754-compliant
    numpy float32 execution (python-certifier/compliant_forward.py) relative to the
    numpy-predicted label of the clean point, the label the attack ran against;
    the clean label is also compared with the `attack_label_numpy` the results JSON
    recorded, which checks that the model and the test set are the ones attacked.
The weights are the certifier-format CSVs (float64 decimals cast to float32, as
Keras holds them); the clean points are the keras CIFAR-10 test set at float32 /255,
exactly as doitlib.load_test_data gives them to the attack.

Usage (under cav2025-artifact-venv, PYTHON_CERTIFIER pointing at python-certifier):
  python reverify_cifar10_adversarials.py [--csv CSV_DIR] [--eps 0.141]
      [--results-dir results] [--out results/reverify_cifar10_adversarials.json]
Writes a JSON {attack: {n, n_distance_ok, n_in_range, n_flip, max_distance, ...}}.
"""
import json
import os
import sys
from fractions import Fraction

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
PYTHON_CERTIFIER = os.path.expanduser(os.environ.get("PYTHON_CERTIFIER", "~/git/python-certifier"))
sys.path.insert(0, PYTHON_CERTIFIER)
import compliant_forward  # noqa: E402
import doitlib  # noqa: E402

CIFAR_LAYERS = [512, 256, 128, 128, 128, 128, 128, 128]
CIFAR_CSV = os.path.join(SCRIPTS, "..", "cav2025-models", "2025-01-28_20:39:32-cifar10",
                         "model_weights_epsilon_0.1551_[512,256,128,128,128,128,128,128]_800")


def opt(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


csv_loc = opt("--csv", CIFAR_CSV).rstrip("/") + "/"
eps_str = opt("--eps", "0.141")
eps_q = Fraction(eps_str)
results_dir = opt("--results-dir", os.path.join(SCRIPTS, "results"))
out_path = opt("--out", os.path.join(results_dir, "reverify_cifar10_adversarials.json"))
attacks = opt("--attacks", "pgd,apgd").split(",")

# the compliant numpy model: certifier orientation (out, in), float32 (the trained model's format)
Ws = [np.ascontiguousarray(np.loadtxt(csv_loc + f"layer_{i}_weights.csv", delimiter=",").T).astype(np.float32)
      for i in range(len(CIFAR_LAYERS) + 1)]
NP_MODEL = (Ws, None, np.float32)


def np_pred(x):
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return int(np.argmax(np.asarray(compliant_forward.forward_logits(NP_MODEL, flat), dtype=np.float64)))


def sq_dist_exact(a, b):
    """Exact squared l2 distance of two float32 vectors (every float32 value is a rational)."""
    return sum((Fraction(float(u)) - Fraction(float(v))) ** 2 for u, v in zip(a.ravel().tolist(), b.ravel().tolist()))


x_test, _ = doitlib.load_test_data(input_size=32, dataset="cifar10")
x_test = np.asarray(x_test, dtype=np.float32).reshape(len(x_test), -1)

out = {}
for atk in attacks:
    stem = os.path.join(results_dir, f"{atk}_l2_cifar10_eps{eps_str}_predlabel_exact")
    z = np.load(stem + "_adv.npz")
    idx, x_adv = z["idx"], z["x_adv"]
    assert x_adv.dtype == np.float32, x_adv.dtype
    recorded = {r["idx"]: r for r in json.load(open(stem + ".json"))["records"]}
    n = len(idx)
    n_dist_ok = n_in_range = n_flip = n_label_match = 0
    max_sq = Fraction(0)
    bad = []
    for k in range(n):
        i = int(idx[k])
        xa = x_adv[k]
        xi = x_test[i]
        sq = sq_dist_exact(xi, xa)
        dist_ok = sq <= eps_q * eps_q
        in_range = bool(np.all(xa >= 0.0) and np.all(xa <= 1.0))
        clean = np_pred(xi)
        adv = np_pred(xa)
        flip = adv != clean
        label_match = clean == recorded[i]["attack_label_numpy"]
        n_dist_ok += dist_ok
        n_in_range += in_range
        n_flip += flip
        n_label_match += label_match
        max_sq = max(max_sq, sq)
        if not (dist_ok and in_range and flip and label_match):
            bad.append({"idx": i, "dist_ok": dist_ok, "in_range": in_range, "flip": flip,
                        "clean_label_matches_record": label_match, "clean": clean, "adv": adv})
    summary = {"n": n, "n_distance_ok": n_dist_ok, "n_in_range": n_in_range, "n_flip": n_flip,
               "n_clean_label_matches_record": n_label_match,
               "max_distance": float(np.sqrt(float(max_sq))),  # sqrt of the exact maximum, for reading
               "max_sq_distance_exact": str(max_sq), "eps": eps_str, "n_unique_idx": len(set(idx.tolist())),
               "failures": bad}
    out[atk] = summary
    print(atk, json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)
    for b in bad[:20]:
        print("   ", b)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print("Wrote", out_path)
