# Render the eps-cliff sweep figures from the seed-variance experiment (response to
# arXiv:2601.13303, Le & Cao 2026). Third stage of the pipeline:
#   1. seed_variance_sweep.sh   -- train 10 seeds per config
#   2. run_eps_cliff_sweeps.sh  -- evaluate certified robustness across eval eps
#                                  -> seed_variance_results/eps_sweep_{mnist,higgs}.tsv
#   3. this script              -- mean line + min-max band across seeds, full-axis
#                                  and zoomed variants, PNG/SVG/PDF each
# Run: ./cav2025-artifact-venv/bin/python3 plot_eps_sweeps.py
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_variance_results")

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#3B5FC0"

def load(path):
    by_eps = defaultdict(dict)
    for r in csv.DictReader(open(path), delimiter="\t"):
        by_eps[float(r["eps"])][r["run"]] = float(r["robustness"]) * 100  # last wins on re-runs
    eps = sorted(by_eps)
    mean = [sum(by_eps[e].values()) / len(by_eps[e]) for e in eps]
    lo = [min(by_eps[e].values()) for e in eps]
    hi = [max(by_eps[e].values()) for e in eps]
    ns = sorted({len(by_eps[e]) for e in eps})
    return eps, mean, lo, hi, ns

def draw(fname, title, eps, mean, lo, hi, ns, marks, zoom=False, sub=None,
         color=None, marker=None, xlabel="evaluation ε (L2)"):
    color = color or BLUE
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    ax.fill_between(eps, lo, hi, color=color, alpha=0.3, linewidth=0)
    ax.plot(eps, mean, color=color, linewidth=1.2, solid_capstyle="round",
            marker=marker, markersize=5)

    if zoom:
        pad = 0.02 * (max(hi) - min(lo))
        y0, y1 = min(lo) - pad, max(hi) + pad
    else:
        y0, y1 = 0, 100

    for x, label in marks:
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x, y0 + 0.03 * (y1 - y0), label, color=MUTED, fontsize=8.5,
                ha="left", va="bottom", rotation=90)

    band = max(h - l for h, l in zip(hi, lo))
    n = ns[-1] if len(ns) == 1 else f"{ns[0]}-{ns[-1]}"
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    if sub is None:
        sub = f"mean over {n} seeds; band = min-max per ε (widest: {band:.2f}pp)"
        if zoom:
            sub += " — y-axis zoomed to data range"
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")

    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel("certified robust (% of test set)", color=INK, fontsize=10)
    ax.set_ylim(y0, y1)
    ax.set_xlim(min(eps), max(eps))
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{fname}.{ext}", facecolor="white")
    print(fname, f"widest band {band:.3f}pp, n={n}")

eps, mean, lo, hi, ns = load(f"{R}/eps_sweep_mnist.tsv")
mnist_marks = [(0.3, "ε certified (0.3)"), (0.45, "ε trained (0.45)")]
draw(f"{R}/eps_sweep_mnist",
     "MNIST gloro [128]×8: certified robustness vs evaluation ε",
     eps, mean, lo, hi, ns, mnist_marks)
draw(f"{R}/eps_sweep_mnist_zoom",
     "MNIST gloro [128]×8: certified robustness vs evaluation ε (zoom)",
     eps, mean, lo, hi, ns, mnist_marks, zoom=True)

eps, mean, lo, hi, ns = load(f"{R}/eps_sweep_higgs.tsv")
higgs_marks = [(0.1, "ε trained = certified (0.1)")]
draw(f"{R}/eps_sweep_higgs",
     "HIGGS gloro [512]×5: certified robustness vs evaluation ε",
     eps, mean, lo, hi, ns, higgs_marks)
draw(f"{R}/eps_sweep_higgs_zoom",
     "HIGGS gloro [512]×5: certified robustness vs evaluation ε (zoom)",
     eps, mean, lo, hi, ns, higgs_marks, zoom=True)

# Le & Cao's own reported numbers (arXiv:2601.13303 Table III, MNIST): standard-trained
# ResNet-4, alpha-beta-CROWN, first 100 test inputs, L-inf. They report mean and stddev
# (not min-max), so this band is +/-1 stddev -- which if anything UNDERSTATES their
# spread relative to the min-max bands above.
LC_ORANGE = "#B4531F"
lc_eps = [0.006, 0.007, 0.008]
lc_mean = [81.6, 54.3, 22.3]
lc_std = [22.3, 28.9, 20.5]
draw(f"{R}/lecao_mnist",
     "Le & Cao's standard-trained MNIST models: certified robustness vs ε",
     lc_eps, lc_mean,
     [m - s for m, s in zip(lc_mean, lc_std)],
     [m + s for m, s in zip(lc_mean, lc_std)],
     [10], [],
     sub="mean over 10 seeds; band = ±1 standard deviation (data: their Table III)",
     color=LC_ORANGE, marker="o", xlabel="evaluation ε (L∞)")
