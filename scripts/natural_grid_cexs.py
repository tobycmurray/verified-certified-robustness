#!/usr/bin/env python3
"""Counter-examples between valid images, with the certified image a NATURAL one.

For each test image x -- a valid image, every pixel on the k/255 grid -- that
Tobler et al.'s real-arithmetic certification certifies at the standard radius
EPS (the deployed execution's float32 logits y(x) satisfy y_i - y_j > EPS * L_ij
for every j, with the verified Lipschitz constants L of lipschitz_json), search
for a valid image x' with ||x' - x||_2 <= EPS that the deployed execution
classifies differently from x. Each such pair refutes the certificate for x.

Search (per certified x): projected gradient descent in the EPS-ball around x on
the real-arithmetic model (float64 forward/backward through the deployed
weights; loss = margin of x's label over the runner-up), judging every iterate
AFTER rounding it to the pixel grid, under the deployed IEEE-754-compliant
float32 execution (python-certifier's compliant_forward). Rounding can leave an
iterate just outside the ball; the pixels furthest from x are then moved back
one grey level at a time until it is inside. The first in-ball grid image whose
float32 label differs from x's is returned.

Verification of each success, independent of the search, in EXACT rational
arithmetic (Python Fraction; float32 pixel values, float32 logits, the decimal
EPS and the certifier's decimal Lipschitz constants are all exact rationals):
both images on the grid; sum of squared pixel differences <= EPS^2 (no square
root, no rounding); the compliant float32 labels differ; the certifier's margin
test certifies x at EPS. The float32 logits of x and the radius EPS are also
written in Tobler et al.'s certifier input format (one line per instance) so
that their Dafny binary can be run on the instances.

CLI is the attack script's CLI (the model, data, biased-model env FP_BIAS /
FP_BIAS_POS / BIAS_OUTPUT and the certifier oracle are all built by importing
attack_verified_certifier_nat_numpy) plus:

  python natural_grid_cexs.py float_format dataset INTERNAL_LAYER_SIZES \
      model_weights_csv_dir input_size lipschitz_json --eps EPS --out OUT.json \
      [--steps 200] [--restarts 5] [--alpha-frac 0.1] [--seed 0] \
      [--limit N] [--indices i,j,k] [--summarise]

Resumable: per-point records are appended to OUT.jsonl as they complete and
points already there are skipped on restart; --summarise only rebuilds OUT.json
(summary + records) from OUT.jsonl. Found pairs are saved as float32 .npy under
OUT_points/ (idx<I>_x.npy, idx<I>_xadv.npy) and OUT.certifier_input.txt.

Run under cav2025-artifact-venv with PYTHONPATH=. and PYTHON_CERTIFIER set.
"""
import json
import os
import sys
import time
from fractions import Fraction

# The float64 matvecs are tiny; OpenBLAS's thread pool costs ~2 ms per call
# here (200x the single-threaded time), so pin BLAS to one thread.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

if len(sys.argv) < 7:
    print(__doc__)
    sys.exit(1)


def _pop_opt(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


def _pop_flag(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


EPS_STR = _pop_opt("--eps")
out_path = _pop_opt("--out")
STEPS = int(_pop_opt("--steps", "200"))
RESTARTS = int(_pop_opt("--restarts", "5"))
ALPHA_FRAC = float(_pop_opt("--alpha-frac", "0.1"))
seed = int(_pop_opt("--seed", "0"))
limit = _pop_opt("--limit")
indices_opt = _pop_opt("--indices")
summarise_only = _pop_flag("--summarise")
reverify_only = _pop_flag("--reverify")
if EPS_STR is None or out_path is None:
    print("--eps and --out are required")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_verified_certifier_nat_numpy as A  # noqa: E402  (parses argv, builds models)
from mpmath import mp, mpf  # noqa: E402

LEVELS = 255
EPS = mpf(EPS_STR)          # exact decimal, as the certifier reads it
EPS_F = float(EPS)
rng = np.random.default_rng(seed)

jsonl_path = os.path.splitext(out_path)[0] + ".jsonl"
points_dir = os.path.splitext(out_path)[0] + "_points"
certin_path = os.path.splitext(out_path)[0] + ".certifier_input.txt"

# ---- the real-arithmetic model (float64) for gradients ---------------------------
Ws64 = [np.asarray(W, dtype=np.float64) for W in A.NP_MODEL[0]]
bs64 = ([np.asarray(b, dtype=np.float64) for b in A.NP_MODEL[1]]
        if A.NP_MODEL[1] is not None else [np.zeros(W.shape[0]) for W in Ws64])
NL = len(Ws64)


def forward64(x):
    """Pre-activations of every layer (ReLU hidden, identity output)."""
    pre = []
    z = x
    for l in range(NL):
        a = Ws64[l] @ z + bs64[l]
        pre.append(a)
        z = np.maximum(a, 0.0) if l < NL - 1 else a
    return pre


def margin_and_grad(x, i):
    """Real-arithmetic margin y_i - max_{j != i} y_j at x and its gradient wrt x."""
    pre = forward64(x)
    y = pre[-1]
    yo = y.copy()
    yo[i] = -np.inf
    j = int(np.argmax(yo))
    g = np.zeros_like(y)
    g[i] = 1.0
    g[j] = -1.0
    for l in range(NL - 1, -1, -1):
        g = Ws64[l].T @ g
        if l > 0:
            g = g * (pre[l - 1] > 0)
    return float(y[i] - y[j]), g


# ---- the pixel grid ---------------------------------------------------------------
def grid_value(k):
    """The float32 pixel value of grey level k, as the data loader computes it."""
    return (np.asarray(k, dtype=np.float32) / np.float32(LEVELS))


def snap(x64):
    """Nearest grid image to x64 (clipped to [0,1]), as float32 pixel values."""
    k = np.clip(np.rint(np.asarray(x64, dtype=np.float64) * LEVELS), 0, LEVELS)
    return grid_value(k)


def levels(q32):
    return np.rint(np.asarray(q32, dtype=np.float64) * LEVELS).astype(np.int64)


def on_grid(q32):
    q32 = np.asarray(q32, dtype=np.float32).reshape(-1)
    return bool(np.array_equal(q32, grid_value(levels(q32))))


def f64(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def l2(a, b):
    return float(np.linalg.norm(f64(a) - f64(b)))


def exact_dist(a32, b32):
    return A.l2_norm_mph(A.vector_to_mph(f64(a32)), A.vector_to_mph(f64(b32)))


EPS_Q = Fraction(EPS_STR)        # the radius as an exact rational


def _load_L_exact(path):
    with open(path) as f:
        data = json.load(f, parse_float=Fraction)
    for obj in data:
        if isinstance(obj, dict) and "lipschitz_bounds" in obj:
            return [[Fraction(v) for v in row] for row in obj["lipschitz_bounds"]]
    raise ValueError("no lipschitz_bounds in " + path)


L_Q = _load_L_exact(A.lipschitz_json)


def certified_at_exact(y, lab):
    """The certifier's margin test y_lab - y_j > L_{lab,j} * EPS, in exact rationals."""
    yq = [Fraction(float(v)) for v in np.asarray(y, dtype=np.float64).ravel()]
    return all(yq[lab] - yq[j] > L_Q[lab][j] * EPS_Q for j in range(len(yq)) if j != lab)


def exact_sq_dist(a32, b32):
    """Squared distance as an exact rational: float32 pixel values are exact
    rationals, so no rounding occurs anywhere (and no square root is needed)."""
    a = np.asarray(a32, dtype=np.float32).reshape(-1)
    b = np.asarray(b32, dtype=np.float32).reshape(-1)
    return sum((Fraction(float(u)) - Fraction(float(v))) ** 2 for u, v in zip(a, b))


def label32(q32):
    return int(np.argmax(A.np_logits(q32)))


def shrink_into_ball(q32, x32):
    """Move the pixels of q furthest from x back towards x, one grey level each,
    until ||q - x||_2 <= EPS (float64 check; the exact check comes at
    verification). Each round moves the fewest furthest pixels whose moves
    together cover the excess (pixels are moved again in later rounds if they
    are still the furthest). Returns (q, moves) or (None, moves)."""
    kq = levels(q32)
    kx = levels(x32)
    moves = 0
    while True:
        q = grid_value(kq)
        excess = l2(q, x32) ** 2 - EPS_F ** 2
        if excess <= 0:
            return q, moves
        d = kq - kx
        ad = np.abs(d)
        order = np.argsort(-ad, kind="stable")
        # moving pixel i one level towards x lowers the squared distance (in grid
        # units) by 2|d_i| - 1; take the furthest pixels until that covers the excess
        red = np.cumsum((2 * ad[order] - 1) / float(LEVELS) ** 2)
        m = int(np.searchsorted(red, excess)) + 1
        m = min(m, int(np.sum(ad > 0)))
        if m <= 0:
            return None, moves
        sel = order[:m]
        kq[sel] -= np.sign(d[sel]).astype(np.int64)
        moves += m


def certified_at(y, lab, eps):
    ok, _ = A.certifier_oracle_logits_mp(y, eps, lab)
    return bool(ok)


def certified_radius(y, lab):
    """Largest radius (to 1e-6 relative) the certifier certifies; None if none."""
    eps = EPS_F
    if not certified_at(y, lab, mpf(eps)):
        while eps > 1e-9 and not certified_at(y, lab, mpf(eps)):
            eps /= 1.5
        if eps <= 1e-9:
            return None
    lo, hi = eps, eps * 1.5
    while certified_at(y, lab, mpf(hi)):
        lo, hi = hi, hi * 1.5
    while hi - lo > 1e-6 * lo:
        mid = 0.5 * (lo + hi)
        if certified_at(y, lab, mpf(mid)):
            lo = mid
        else:
            hi = mid
    return lo


# ---- the search ---------------------------------------------------------------------
def search(x32, lab):
    """PGD-l2 on the real-arithmetic margin, judging the grid-rounded iterate under
    the deployed float32 execution. Returns (xadv32, info) or (None, info)."""
    x = f64(x32)
    d = x.size
    alpha = ALPHA_FRAC * EPS_F
    n_eval = 0
    last_q = None
    min_margin = np.inf
    for r in range(RESTARTS):
        if r == 0:
            xk = x.copy()
        else:
            delta = rng.standard_normal(d)
            delta *= EPS_F * rng.uniform(0.5, 1.0) / np.linalg.norm(delta)
            xk = np.clip(x + delta, 0.0, 1.0)
        for k in range(STEPS + 1):
            q = snap(xk)
            moves = 0
            if l2(q, x32) > EPS_F:
                q, moves = shrink_into_ball(q, x32)
            if q is not None and (last_q is None or not np.array_equal(q, last_q)):
                last_q = q
                n_eval += 1
                if label32(q) != lab:
                    return q, {"restart": r, "step": k, "shrink_moves": moves,
                               "grid_evals": n_eval, "min_real_margin": min_margin}
            if k == STEPS:
                break
            m, g = margin_and_grad(xk, lab)
            min_margin = min(min_margin, m)
            gn = np.linalg.norm(g)
            if gn == 0.0:
                break
            xk = xk - alpha * g / gn
            dv = xk - x
            nv = np.linalg.norm(dv)
            if nv > EPS_F:
                dv *= EPS_F / nv
            xk = np.clip(x + dv, 0.0, 1.0)
    return None, {"grid_evals": n_eval, "min_real_margin": min_margin}


def verify(x32, xadv32, lab):
    """Independent re-check of a found pair; returns (ok, details)."""
    det = {}
    det["x_on_grid"] = on_grid(x32)
    det["xadv_on_grid"] = on_grid(xadv32)
    dist = exact_dist(x32, xadv32)
    det["dist"] = float(dist)                      # for reporting only
    det["dist_le_eps"] = exact_sq_dist(x32, xadv32) <= EPS_Q * EPS_Q   # exact rational test
    y = A.np_logits(x32)
    det["label_x"] = int(np.argmax(y))
    det["label_xadv"] = label32(xadv32)
    det["labels_differ"] = det["label_x"] != det["label_xadv"]
    det["label_x_as_given"] = det["label_x"] == lab
    det["certified_at_eps"] = certified_at_exact(y, lab)
    ok = (det["x_on_grid"] and det["xadv_on_grid"] and det["dist_le_eps"]
          and det["labels_differ"] and det["label_x_as_given"] and det["certified_at_eps"])
    return ok, det


def write_certin(y, idx):
    with open(certin_path, "a") as f:
        f.write(",".join("{:.150f}".format(float(v)) for v in y) + " " + EPS_STR + "\n")
        f.flush()


# ---- driver ---------------------------------------------------------------------------
def load_done():
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["index"]] = rec
    return done


def summarise(records):
    recs = sorted(records.values(), key=lambda r: r["index"])
    cert = [r for r in recs if r["certified_at_eps"]]
    found = [r for r in cert if r.get("found")]
    dists = [r["dist"] for r in found]
    summary = {
        "format": A.fmt,
        "eps": EPS_STR,
        "steps": STEPS, "restarts": RESTARTS, "alpha_frac": ALPHA_FRAC, "seed": seed,
        "n_attempted": len(recs),
        "n_correct": sum(1 for r in recs if r["correct"]),
        "n_certified_at_eps": len(cert),
        "n_certified_and_correct": sum(1 for r in cert if r["correct"]),
        "n_found": len(found),
        "n_found_correct": sum(1 for r in found if r["correct"]),
        "found_indices": [r["index"] for r in found],
        "dist_max": max(dists) if dists else None,
        "dist_min": min(dists) if dists else None,
        "pixels_changed_median": float(np.median([r["pixels_changed"] for r in found])) if found else None,
        "pixels_changed_max": max(r["pixels_changed"] for r in found) if found else None,
        "R_of_found": {str(r["index"]): r["R"] for r in found},
        "verify_failed": sum(1 for r in cert if r.get("verify_failed")),
        "total_secs": round(sum(r["secs"] for r in recs), 1),
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=1)
    return summary


def main():
    done = load_done()
    if summarise_only:
        print(json.dumps(summarise(done), indent=1))
        return
    if reverify_only:
        bad = 0
        n = 0
        for idx, rec in sorted(done.items()):
            if not rec.get("found"):
                continue
            n += 1
            x32 = np.load(os.path.join(points_dir, f"idx{idx}_x.npy")).reshape(-1)
            xadv32 = np.load(os.path.join(points_dir, f"idx{idx}_xadv.npy")).reshape(-1)
            assert x32.dtype == np.float32 and xadv32.dtype == np.float32
            assert np.array_equal(x32, np.asarray(A.x_test[idx], dtype=np.float32).reshape(-1))
            ok, det = verify(x32, xadv32, rec["label"])
            if not ok:
                bad += 1
                print(f"idx {idx}: RE-VERIFICATION FAILED {det}")
        print(f"re-verified {n} stored pairs with the exact rational distance test: {bad} failures")
        return
    total = A.x_test.shape[0]
    idxs = list(range(total))
    if indices_opt:
        idxs = [int(s) for s in indices_opt.split(",")]
    if limit:
        idxs = idxs[:int(limit)]
    os.makedirs(points_dir, exist_ok=True)
    # the test images must be grid images, as float32 pixel values
    x0 = np.asarray(A.x_test[0], dtype=np.float32).reshape(-1)
    assert on_grid(x0), "test images are not float32 k/255 grid values"
    print(f"eps={EPS_STR} steps={STEPS} restarts={RESTARTS} alpha={ALPHA_FRAC}*eps; "
          f"{len(idxs)} points, {len(done)} already done", flush=True)
    t_all = time.time()
    for idx in idxs:
        if idx in done:
            continue
        t0 = time.time()
        x32 = np.asarray(A.x_test[idx], dtype=np.float32).reshape(-1)
        true = int(np.argmax(A.y_test[idx]))
        y = A.np_logits(x32)
        lab = int(np.argmax(y))
        rec = {"index": idx, "true_label": true, "label": lab, "correct": lab == true}
        rec["certified_at_eps"] = certified_at(y, lab, EPS)
        if rec["certified_at_eps"]:
            rec["R"] = certified_radius(y, lab)
            xadv, info = search(x32, lab)
            rec.update(info)
            rec["found"] = xadv is not None
            if xadv is not None:
                ok, det = verify(x32, xadv, lab)
                rec.update(det)
                rec["pixels_changed"] = int(np.sum(xadv != x32))
                if ok:
                    np.save(os.path.join(points_dir, f"idx{idx}_x.npy"), x32.reshape(A.x_test[idx].shape))
                    np.save(os.path.join(points_dir, f"idx{idx}_xadv.npy"), xadv.reshape(A.x_test[idx].shape))
                    write_certin(y, idx)
                else:
                    rec["found"] = False
                    rec["verify_failed"] = True
        rec["secs"] = round(time.time() - t0, 2)
        done[idx] = rec
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        if rec.get("found"):
            print(f"idx {idx}: FOUND dist={rec['dist']:.5f} R={rec['R']:.4g} labels {lab}->{rec['label_xadv']} "
                  f"pixels={rec['pixels_changed']} restart={rec['restart']} step={rec['step']} "
                  f"shrink={rec['shrink_moves']} ({rec['secs']}s)", flush=True)
        elif rec["certified_at_eps"]:
            print(f"idx {idx}: not found (R={rec['R']:.4g}, min real margin {rec['min_real_margin']:.3g}, "
                  f"{rec['grid_evals']} grid evals, {rec['secs']}s)", flush=True)
        else:
            print(f"idx {idx}: not certified at eps ({rec['secs']}s)", flush=True)
    summary = summarise(done)
    print("\nSUMMARY:", json.dumps(summary, indent=1))
    print(f"Wrote {out_path} ({round(time.time() - t_all, 1)}s this run)")


if __name__ == "__main__":
    main()
