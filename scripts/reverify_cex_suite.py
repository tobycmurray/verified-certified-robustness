#!/usr/bin/env python3
"""Exact re-verification of a stored counter-example suite from the continuous
(DeepFool + bisection) search, python-certifier/tests/cex_<model>_<format>_numpy.
Per record (x1_file = the certified point x, x0_file = the differently-classified
point x', max_eps = the radius eps as a decimal string), checks in exact rational
arithmetic (Fraction) under the IEEE-754-compliant numpy execution:
  * both stored points are vectors of the suite's format (float16/float32/float64, the
    attack script's first argument), so their coordinates are exact rationals;
  * the compliant labels of x and x' at that format differ;
  * ||x' - x||_2^2 <= eps^2 exactly (no square root, no rounding);
  * the certifier's margin test y_lab - y_j > L_{lab,j} * eps holds exactly for x's
    compliant logits y, with L parsed from the Dafny-verified JSON as exact decimals.
The stored points are float64 arrays; the deployed execution casts them to the
model's format (compliant_forward casts x), so the checks that count are made on
those casts, and the suite's own float64 distance is reported alongside. The JSON
keys keep their "f32cast"/"float32" names for every format: they mean "cast to the
suite's format".
Writes results/natural_grid_cex/reverify_<suite>.json (per-record verdicts + summary);
REVERIFY_OUT_DIR overrides the directory. The summary's eps range is over every stored pair; the
verified count and the natural-image distances are this host's (see main).
CLI: the attack script's positional args plus --suite DIR, e.g.
  PYTHONPATH=. PYTHON_CERTIFIER=~/git/python-certifier cav2025-artifact-venv/bin/python \
    reverify_cex_suite.py float32 mnist "[128,...]" CSV_DIR 28 L.json --suite ~/git/python-certifier/tests/cex_mnist_float32_numpy
"""
import os
import sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json  # noqa: E402
from fractions import Fraction  # noqa: E402
import numpy as np  # noqa: E402

if "--suite" not in sys.argv:
    sys.exit(__doc__)
_i = sys.argv.index("--suite")
SUITE = os.path.expanduser(sys.argv[_i + 1])
del sys.argv[_i:_i + 2]
OUT_DIR = os.environ.get("REVERIFY_OUT_DIR") or os.path.join(   # the runner reverify_all_suites.sh sets it
    os.path.dirname(os.path.abspath(__file__)), "results", "natural_grid_cex")
import attack_verified_certifier_nat_numpy as A  # noqa: E402  (parses argv, builds models)

# An adversarially biased suite (cex_<model>_float32_biased_<B>_<pos>_numpy) carries the bias vectors its
# search used (biases.txt, the attack script's BIAS_OUTPUT). The attack module rebuilds the biased model
# from FP_BIAS / FP_BIAS_POS at import (deterministic given the weights: a flat bias B and its float64
# compensation, cast to float32) and writes the rebuilt vectors to BIAS_OUTPUT; they must equal the suite's
# for this run to re-verify the suite's own model. Recorded in the summary (biases_match_suite; None for a
# natural suite); the runner reverify_all_suites.sh sets the three variables from the registry.
_SUITE_BIASES = os.path.join(SUITE, "biases.txt")
_FP_BIAS = float(os.environ.get("FP_BIAS", "0")) > 0
if os.path.exists(_SUITE_BIASES) != _FP_BIAS:
    sys.exit(f"{SUITE}: {'has' if not _FP_BIAS else 'has no'} biases.txt but FP_BIAS is {'un' if not _FP_BIAS else ''}set"
             " (a biased suite is re-verified with FP_BIAS, FP_BIAS_POS and BIAS_OUTPUT set, as the runner does)")
BIASES_MATCH = None
if _FP_BIAS:
    BIASES_MATCH = open(_SUITE_BIASES).read() == open(os.environ["BIAS_OUTPUT"]).read()
    print(f"bias vectors rebuilt from FP_BIAS={os.environ['FP_BIAS']} {os.environ.get('FP_BIAS_POS')}: "
          f"{'equal to' if BIASES_MATCH else 'DIFFER FROM'} the suite's biases.txt")


def load_L_exact(path):
    with open(path) as f:
        data = json.load(f, parse_float=Fraction)
    for obj in data:
        if isinstance(obj, dict) and "lipschitz_bounds" in obj:
            return [[Fraction(v) for v in row] for row in obj["lipschitz_bounds"]]
    raise ValueError("no lipschitz_bounds in " + path)


L_Q = load_L_exact(A.lipschitz_json)


DT = A.NP_DT   # the suite's format (float16/float32/float64), from the attack script's argv


def load_pt(path):
    """Returns (stored float64 vector, its cast to the suite's format, whether the cast is lossless)."""
    x = np.asarray(np.load(os.path.join(SUITE, os.path.basename(path))), dtype=np.float64).reshape(-1)
    x32 = x.astype(DT)
    return x, x32, bool(np.array_equal(x32.astype(np.float64), x))


def sq_dist_exact(a, b):
    return sum((Fraction(float(u)) - Fraction(float(v))) ** 2 for u, v in zip(a, b))


def main():
    recs = [r for r in json.load(open(os.path.join(SUITE, "counter_examples.json"))) if "x1_file" in r]
    bad = 0
    n_nonrepr = 0
    out = []
    for r in recs:
        xc, xc32, rc = load_pt(r["x1_file"])      # certified point x
        xa, xa32, ra = load_pt(r["x0_file"])      # differently-classified point x'
        eps_q = Fraction(r["max_eps"])
        yc, ya = A.np_logits(xc), A.np_logits(xa)              # as the suite judged them (float64 input)
        yc32, ya32 = A.np_logits(xc32), A.np_logits(xa32)      # the float32 casts
        lab, laba = int(np.argmax(yc)), int(np.argmax(ya))
        yq = [Fraction(float(v)) for v in np.asarray(yc32, dtype=np.float64).ravel()]
        det = {
            "both_float32_representable": rc and ra,
            "labels_differ": lab != laba,
            "labels_differ_f32cast": int(np.argmax(yc32)) != int(np.argmax(ya32)),
            "logits_equal_f32cast": bool(np.array_equal(yc, yc32) and np.array_equal(ya, ya32)),
            "labels_as_recorded": lab == r["argmax_y1"] and laba == r["argmax_y0"],
            "dist_le_eps_stored": sq_dist_exact(xc, xa) <= eps_q * eps_q,
            "dist_le_eps_f32cast": sq_dist_exact(xc32, xa32) <= eps_q * eps_q,
            "certified_at_eps_f32cast": all(yq[lab] - yq[j] > L_Q[lab][j] * eps_q for j in range(len(yq)) if j != lab),
        }
        if not (rc and ra):
            n_nonrepr += 1
        # A pair is a verified float32-vector counterexample iff the float32 casts (what the
        # deployed execution sees) satisfy every check; the stored float64 distance is advisory.
        ok = (det["labels_differ_f32cast"] and det["dist_le_eps_f32cast"] and det["certified_at_eps_f32cast"])
        xnat = np.asarray(A.x_test[r["index"]], dtype=np.float64).reshape(-1)
        out.append({"index": r["index"], "verified_float32": ok, "eps": r["max_eps"],
                    "dist_nat": float(np.sqrt(float(sq_dist_exact(xc32, xnat)))),
                    "label_x": lab, "label_xadv": laba, "checks": det})
        if not all(det.values()):
            bad += 1
            print(f"index {r['index']}: {'ok as float32 vectors' if ok else 'NOT VERIFIED'} {det}")
    good = [o for o in out if o["verified_float32"]]
    # The radii are a property of the stored suite (every pair); the distances to the natural image are
    # summarised over the pairs this host verifies, which can be none: the labels of points this close
    # to the decision boundary depend on the host's float32 summation order.
    eps = sorted(float(o["eps"]) for o in out)
    dn = sorted(o["dist_nat"] for o in good)
    summary = {"suite": os.path.basename(SUITE), "n_pairs": len(recs), "n_nonrepresentable": n_nonrepr,
               "n_verified_float32": len(good), "eps_min": eps[0] if eps else None, "eps_max": eps[-1] if eps else None,
               "dist_nat_median": (None if not dn else dn[len(dn) // 2] if len(dn) % 2 else (dn[len(dn) // 2 - 1] + dn[len(dn) // 2]) / 2),
               "dist_nat_min": dn[0] if dn else None, "dist_nat_max": dn[-1] if dn else None,
               "biases_match_suite": BIASES_MATCH}
    print(json.dumps(summary))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"reverify_{os.path.basename(SUITE)}.json"), "w") as f:
        json.dump({"summary": summary, "records": out}, f, indent=1)


if __name__ == "__main__":
    main()
