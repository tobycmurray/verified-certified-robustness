#!/usr/bin/env python3
"""The CIFAR-10 partition of the test set (artifact MANIFEST B8.1, exact part):
certified (real-arithmetic Lipschitz check, Standard mode at float32) / broken (a
point within eps that the execution classifies differently, the union of the PGD
and APGD attacks judged by the exact distance) / neither.

Inputs: the broken-union JSON pgd_l2_robust_accuracy.py's runs were merged into
(field `broken_idx`) and a certifier per-instance results JSON (a list: header dict
then one record per instance with `certified_real`).
Usage:
  python3 cifar10_partition.py <certifier_results.json> [<broken_union.json>] [--out out.json]
Prints and writes the four counts as integers and percentages of the instance count.
"""
import json
import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
out_path = None
if "--out" in argv:
    k = argv.index("--out")
    out_path = argv[k + 1]
    del argv[k:k + 2]
args = argv
if not args:
    sys.exit(__doc__)
cert_path = args[0]
broken_path = args[1] if len(args) > 1 else os.path.join(
    SCRIPTS, "results", "broken_union_cifar10_eps0.141_predlabel_exact.json")

cert = json.load(open(cert_path))
header, recs = cert[0], cert[1:]
n = len(recs)
certified = {i for i, r in enumerate(recs) if r["certified_real"]}
bu = json.load(open(broken_path))
broken = set(int(i) for i in bu["broken_idx"])
assert len(broken) == bu["n_broken"] == len(bu["broken_idx"]), "broken_idx inconsistent with n_broken"
assert all(0 <= i < n for i in broken)
both = certified & broken
neither = set(range(n)) - certified - broken
res = {"n": n, "certifier": header, "broken_source": os.path.basename(broken_path),
       "certified_real": len(certified), "broken": len(broken), "certified_and_broken": len(both),
       "neither": len(neither)}
for k in ("certified_real", "broken", "certified_and_broken", "neither"):
    res[k + "_pct"] = 100.0 * res[k] / n
    print(f"{k:22s} {res[k]:6d}  {res[k + '_pct']:6.2f} %")
if both:
    res["certified_and_broken_idx"] = sorted(both)
    print("certified AND broken (soundness violation if the verdicts are both right):", sorted(both)[:50])
if out_path:
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("Wrote", out_path)
