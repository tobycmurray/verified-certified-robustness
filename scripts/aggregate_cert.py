"""Aggregate cert_morning/<tag>/ results into the E1 (FP-cost-vs-width) and E2
(breadth) tables. float32, all modes. Metrics:
  robustness rate = proportion certified robust (the pure certifier-conservatism
                    measure; FP cost = real - FP);
  VRA            = certified AND correctly-classified;
  balanced acc   = mean per-class accuracy (matters for the imbalanced byclass).
"""
import sys, os, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "cert_morning"
MODES = ["standard", "hybrid-only", "hybrid-meas"]


def model_metrics(d):
    cex = json.load(open(os.path.join(d, "cex.json")))
    labels = np.array([c["label"] for c in cex])
    out = {}
    for mode in MODES:
        p = os.path.join(d, f"results_{mode}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))[1:]  # drop header
        n = len(r)
        lab = labels[:n]
        pred = np.array([int(np.argmax(x["output"])) for x in r])
        cert = np.array([bool(x["certified"]) for x in r])
        certr = np.array([bool(x["certified_real"]) for x in r])
        correct = pred == lab
        cls = [c for c in np.unique(lab) if (lab == c).any()]
        bacc = np.mean([correct[lab == c].mean() for c in cls])
        mvra_fp = np.mean([(cert & correct)[lab == c].mean() for c in cls])   # macro VRA (FP)
        mvra_re = np.mean([(certr & correct)[lab == c].mean() for c in cls])  # macro VRA (real)
        out[mode] = dict(
            n=n, acc=correct.mean(), bacc=bacc, mvra_fp=mvra_fp, mvra_re=mvra_re,
            rob_fp=cert.mean(), rob_re=certr.mean(),
            vra_fp=(cert & correct).mean(), vra_re=(certr & correct).mean(),
        )
    return out


tags = sorted(t for t in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, t)))
data = {t: model_metrics(os.path.join(OUT, t)) for t in tags}

print("\n================ E1: HIGGS FP-cost-vs-width (10k test, float32) ================")
print(f"{'width':>6} | {'real rob':>8} | {'FP rob (std/hyb/meas)':>26} | {'FP cost pp (std/hyb/meas)':>26} | {'real VRA':>8}")
for w in (128, 256, 512, 1024):
    t = f"higgs_w{w}_d5_full"
    if t not in data or "standard" not in data[t]:
        continue
    m = data[t]
    rr = m["standard"]["rob_re"] * 100
    rob = [m[k]["rob_fp"] * 100 if k in m else float('nan') for k in MODES]
    cost = [rr - x for x in rob]
    print(f"{w:>6} | {rr:7.2f}% | {rob[0]:7.2f}/{rob[1]:6.2f}/{rob[2]:6.2f} | "
          f"{cost[0]:7.2f}/{cost[1]:6.2f}/{cost[2]:6.2f} | {m['standard']['vra_re']*100:7.2f}%")

print("\n================ E2: breadth (full test, float32) ================")
e2 = [("higgs_w512_d5_full", "HIGGS-512 (full 11M)"),
      ("emnistbal_w512_d8_ep500", "EMNIST-bal 47 (500ep)"),
      ("emnistbyc_cifar", "EMNIST-byclass 62")]
print(f"{'model':>24} | {'n':>7} | {'acc':>6} {'bacc':>6} | {'VRA std/hyb/meas':>22} | {'real VRA':>8} | {'rob std/hyb/meas':>22} | {'real rob':>8}")
for t, name in e2:
    if t not in data or "standard" not in data[t]:
        continue
    m = data[t]
    vra = [m[k]["vra_fp"] * 100 if k in m else float('nan') for k in MODES]
    rob = [m[k]["rob_fp"] * 100 if k in m else float('nan') for k in MODES]
    s = m["standard"]
    print(f"{name:>24} | {s['n']:>7} | {s['acc']*100:5.1f}% {s['bacc']*100:5.1f}% | "
          f"{vra[0]:6.2f}/{vra[1]:5.2f}/{vra[2]:5.2f} | {s['vra_re']*100:7.2f}% | "
          f"{rob[0]:6.2f}/{rob[1]:5.2f}/{rob[2]:5.2f} | {s['rob_re']*100:7.2f}%")

print("\n  macro (per-class-averaged) VRA — the honest metric for imbalanced byclass:")
for t, name in e2:
    if t not in data or "standard" not in data[t]:
        continue
    m = data[t]
    best = "hybrid-meas" if "hybrid-meas" in m else "hybrid-only"
    print(f"    {name:>24}: top-1 VRA({best})={m[best]['vra_fp']*100:5.1f}%  "
          f"macro VRA({best})={m[best]['mvra_fp']*100:5.1f}%  macro real={m[best]['mvra_re']*100:5.1f}%")
print()
