#!/usr/bin/env python3
"""Grid-restricted counter-example search seeded from an existing suite.

Motivation (POPL'27 reviewer C, Q3): image pixels are integers 0..255 (inputs
are k/255), so counter-examples (x0, x1) found by the continuous search need not
be valid images. check_quantised_cexs.py merely rounds both stored points to
the grid and re-judges; this script instead SEARCHES on the grid, seeded from
each stored counter-example, for a pair of valid images (qc, qa) such that

  * the certifier (same oracle, same Lipschitz reference) certifies qc's label
    at radius R(qc) under the IEEE-754-compliant numpy execution, and
  * qa is on the k/255 grid, ||qa - qc||_2 <= R(qc), and the numpy execution
    classifies qa differently from qc.

Per stored record (c = x1_file = certified point; a = x0_file = the
differently-classified point):
  1. snap(x) = clip(rint(x*255), 0, 255)/255 cast to the deployment dtype;
     dir = (a - c)/||a - c||.
  2. Candidate certified points qc_t = snap(c + t*(c - a)) for t in T_STEPS
     (t > 0 moves the certified point AWAY from the adversarial one, like the
     original search's extension step). Require label(qc_t) == label(c) under
     np_logits; compute its certified radius R(qc_t) (grow by 1.1x, then bisect
     to 1e-6 relative).
  3. Walk s upward along the ray qc_t + s*dir (snapped), step 1/255/4, up to
     s_max = min(1.5 R, 3 ||c - a||) and at most MAX_STEPS steps; success is
     the first snapped point whose label differs from label(qc_t) and whose
     float64 distance to qc_t is <= R. At flipped-but-too-far points, try a
     local repair: up to REPAIR_MOVES random single-pixel moves of one grey
     level toward qc_t (always distance-reducing), keeping only moves that
     preserve a differing label, until the distance is <= R.
  3b. If the ray (and repairs) fail and R >= 1/255 (the grid's minimum
     non-zero L2 distance), enumerate the grid NEIGHBOURHOOD of qc_t: every
     single-pixel one-level move (2*d candidates), then, if R >= sqrt(2)/255
     (sqrt(3)/255), all pairs (triples) among the NEIGH_K single moves with the
     smallest certified-class margin. Needed at float16, where certified radii
     are of the order of one grey level.
  4. Verify a success with A.check_counter_example(qa, qc_t) (must be True with
     max_eps >= ||qa - qc_t||). First success per record wins.

--allow-label-change: also use candidates qc_t whose numpy label differs from
label(c) (snapping moved c across the boundary); the certifier certifies
whatever label the model gives qc_t, so such pairs are still valid grid
counter-examples, but they are reported separately (default: skip, as spec'd).

The judge is the compliant numpy execution throughout (never Keras). CLI is
the attack script's CLI plus the cex directory:

  python grid_search_cexs.py float_format dataset INTERNAL_LAYER_SIZES \
      model_weights_csv_dir input_size lipschitz_json cex_dir \
      [--out out.json] [--naive quant_<tag>.json] [--seed N] [--limit N] \
      [--allow-label-change]

Run under cav2025-artifact-venv with the same env (PYTHONPATH, PYTHON_CERTIFIER,
FP_BIAS / FP_BIAS_POS / BIAS_OUTPUT) as check_quantised_cexs.py.
"""
import json
import os
import sys
import time

import numpy as np

# ---- CLI: strip our own arguments, hand the attack script's CLI to the module ----
if len(sys.argv) < 8:
    print(__doc__)
    sys.exit(1)


def _pop_opt(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


out_path = _pop_opt("--out")
naive_path = _pop_opt("--naive")
seed = int(_pop_opt("--seed", "0"))
limit = _pop_opt("--limit")
allow_label_change = "--allow-label-change" in sys.argv
if allow_label_change:
    sys.argv.remove("--allow-label-change")
cex_dir = sys.argv.pop(7)
if out_path is None:
    out_path = os.path.join(cex_dir, "grid_search.json")

# The attack module parses sys.argv at import (fmt dataset layers csv isize ref),
# builds the (possibly bias-amplified) Keras model, the compliant numpy model and
# the certifier oracle. Nothing else runs at import (main() is __main__-guarded).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attack_verified_certifier_nat_numpy as A  # noqa: E402

LEVELS = 255
T_STEPS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
RAY_STEP = 1.0 / LEVELS / 4
MAX_STEPS = 4000
REPAIR_MOVES = 200
MAX_REPAIR_SITES = 8          # distinct flipped-but-too-far ray points to try repairing per t
NEIGH_K = 30                  # single moves (by smallest margin) combined into pairs/triples
NEIGH_K3 = 20
rng = np.random.default_rng(seed)


def snap(x):
    """Snap to the k/255 grid: nearest level, clipped to [0,1], deployment dtype."""
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    q = np.clip(np.rint(x64 * LEVELS), 0, LEVELS) / LEVELS
    return q.astype(A.NP_DT)


def on_grid(x):
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    # deployment-dtype cast of k/255 (e.g. float16) is not exactly k/255
    tol = 1e-3 if A.NP_DT is np.float16 else 1e-6
    return bool(np.all(np.abs(np.rint(x64 * LEVELS) / LEVELS - x64) <= tol))


def f64(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def l2(a, b):
    return float(np.linalg.norm(f64(a) - f64(b)))


def label(x):
    return int(np.argmax(A.np_logits(x)))


def certified_radius(y, lab, start):
    """Largest eps (to 1e-6 relative) with certifier_oracle_logits_mp(y, eps, lab) True.
    Returns None if the certifier certifies nothing even at tiny radii."""
    eps = max(float(start), 1e-4)
    ok, _ = A.certifier_oracle_logits_mp(y, eps, lab)
    if not ok:
        while not ok and eps > 1e-9:
            eps /= 1.1
            ok, _ = A.certifier_oracle_logits_mp(y, eps, lab)
        if not ok:
            return None
    lo = eps
    hi = eps * 1.1
    ok, _ = A.certifier_oracle_logits_mp(y, hi, lab)
    while ok:
        lo = hi
        hi *= 1.1
        ok, _ = A.certifier_oracle_logits_mp(y, hi, lab)
    while hi - lo > 1e-6 * lo:
        mid = 0.5 * (lo + hi)
        ok, _ = A.certifier_oracle_logits_mp(y, mid, lab)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


def repair(qa, qc, lab_c, R):
    """Random single-pixel moves of one grey level toward qc, keeping only those
    that preserve label(qa) != lab_c; stop when ||qa - qc|| <= R."""
    qa = qa.copy()
    qc64 = f64(qc)
    d = l2(qa, qc)
    accepted = 0
    for _ in range(REPAIR_MOVES):
        if d <= R:
            break
        diff = f64(qa) - qc64
        idx = np.flatnonzero(np.abs(diff) > 0.5 / LEVELS)
        if idx.size == 0:
            break
        i = int(rng.choice(idx))
        old = qa[i]
        k = np.rint(float(old) * LEVELS) - np.sign(diff[i])
        qa[i] = A.NP_DT(np.clip(k, 0, LEVELS) / LEVELS)
        if label(qa) != lab_c:
            accepted += 1
            d = l2(qa, qc)
        else:
            qa[i] = old
    return qa, d, accepted


def margin(y, lab):
    y = np.asarray(y, dtype=np.float64).ravel()
    others = np.delete(y, lab)
    return float(y[lab] - others.max())


def neighbourhood(qc, lab_c, R):
    """Enumerate one-level single-pixel moves of qc (then pairs/triples among the
    lowest-margin singles, radius permitting). Returns (qa, dist, kind) or None."""
    qc64 = f64(qc)
    d = qc64.size
    lvl = np.rint(qc64 * LEVELS)
    singles = []                       # (margin, i, delta)
    for i in range(d):
        for delta in (1, -1):
            k = lvl[i] + delta
            if k < 0 or k > LEVELS:
                continue
            qa = qc.copy()
            qa[i] = A.NP_DT(k / LEVELS)
            y = A.np_logits(qa)
            if int(np.argmax(y)) != lab_c:
                return qa, l2(qa, qc), "neigh1"
            singles.append((margin(y, lab_c), i, delta))
    if R < np.sqrt(2) / LEVELS or not singles:
        return None
    singles.sort()
    top = singles[:NEIGH_K]
    for x in range(len(top)):
        for yy in range(x + 1, len(top)):
            _, i, di = top[x]
            _, j, dj = top[yy]
            if i == j:
                continue
            qa = qc.copy()
            qa[i] = A.NP_DT((lvl[i] + di) / LEVELS)
            qa[j] = A.NP_DT((lvl[j] + dj) / LEVELS)
            if label(qa) != lab_c:
                return qa, l2(qa, qc), "neigh2"
    if R < np.sqrt(3) / LEVELS:
        return None
    top = singles[:NEIGH_K3]
    n = len(top)
    for x in range(n):
        for yy in range(x + 1, n):
            for z in range(yy + 1, n):
                (_, i, di), (_, j, dj), (_, k_, dk) = top[x], top[yy], top[z]
                if len({i, j, k_}) < 3:
                    continue
                qa = qc.copy()
                qa[i] = A.NP_DT((lvl[i] + di) / LEVELS)
                qa[j] = A.NP_DT((lvl[j] + dj) / LEVELS)
                qa[k_] = A.NP_DT((lvl[k_] + dk) / LEVELS)
                if label(qa) != lab_c:
                    return qa, l2(qa, qc), "neigh3"
    return None


def search_record(c, a):
    """Returns (found_dict_or_None, per_t_diagnostics)."""
    c64, a64 = f64(c), f64(a)
    ca = float(np.linalg.norm(a64 - c64))
    direction = (a64 - c64) / ca
    lab_c = label(c)
    diags = []
    n_logits = 0
    for t in T_STEPS:
        qc = snap(c64 + t * (c64 - a64))
        y_qc = A.np_logits(qc)
        lab_qc = int(np.argmax(y_qc))
        diag = {"t": t, "label_c": lab_c, "label_qc": lab_qc}
        if lab_qc != lab_c and not allow_label_change:
            diag["skipped"] = "label(qc_t) != label(c)"
            diags.append(diag)
            continue
        qc_a = l2(qc, a)
        R = certified_radius(y_qc, lab_qc, qc_a / 4)
        diag["R"] = R
        diag["dist_qc_a"] = qc_a
        if R is None:
            diag["skipped"] = "certifier certifies nothing at qc_t"
            diags.append(diag)
            continue
        s_max = min(1.5 * R, 3 * ca)
        n_steps = min(MAX_STEPS, int(np.floor(s_max / RAY_STEP)))
        diag["s_max"] = s_max
        diag["n_steps"] = n_steps
        qc64 = f64(qc)
        last = None
        first_flip = None
        repair_sites = 0
        best_repair = None
        for k in range(1, n_steps + 1):
            s = k * RAY_STEP
            qa = snap(qc64 + s * direction)
            if last is not None and np.array_equal(qa, last):
                continue
            last = qa
            n_logits += 1
            lab_qa = label(qa)
            if lab_qa == lab_qc:
                continue
            d = l2(qa, qc)
            if first_flip is None:
                first_flip = {"s": s, "dist": d, "label_qa": lab_qa}
                diag["first_flip"] = first_flip
            if d <= R:
                return ({"t": t, "s": s, "how": "ray", "dist": d, "R": R,
                         "label_qc": lab_qc, "label_qa": lab_qa, "repair_moves": 0},
                        qc, qa, diags + [diag])
            if repair_sites < MAX_REPAIR_SITES:
                repair_sites += 1
                qa_r, d_r, acc = repair(qa, qc, lab_qc, R)
                n_logits += REPAIR_MOVES
                if best_repair is None or d_r < best_repair["dist"]:
                    best_repair = {"s": s, "dist_before": d, "dist": d_r, "accepted_moves": acc}
                    diag["best_repair"] = best_repair
                if d_r <= R:
                    lab_r = label(qa_r)
                    return ({"t": t, "s": s, "how": "ray+repair", "dist": d_r, "R": R,
                             "label_qc": lab_qc, "label_qa": lab_r, "repair_moves": acc},
                            qc, qa_r, diags + [diag])
        diag["repair_sites"] = repair_sites
        if R >= 1.0 / LEVELS:
            nb = neighbourhood(qc, lab_qc, R)
            diag["neighbourhood"] = "tried"
            if nb is not None:
                qa_n, d_n, kind = nb
                if d_n <= R:
                    return ({"t": t, "s": None, "how": kind, "dist": d_n, "R": R,
                             "label_qc": lab_qc, "label_qa": label(qa_n), "repair_moves": 0},
                            qc, qa_n, diags + [diag])
                diag["neighbourhood"] = f"{kind} flip at {d_n:.3g} > R"
        else:
            diag["neighbourhood"] = "R < 1/255: no grid point within R"
        diags.append(diag)
    return None, None, None, diags


if naive_path and not os.path.exists(naive_path):
    print(f"(naive-rounding result {naive_path} not found; skipping comparison)")
    naive_path = None
naive = None
if naive_path:
    with open(naive_path) as f:
        naive = json.load(f)

with open(os.path.join(cex_dir, "counter_examples.json")) as f:
    entries = [e for e in json.load(f) if "_metadata" not in e]
if limit:
    entries = entries[:int(limit)]

points_dir = os.path.splitext(out_path)[0] + "_points"
os.makedirs(points_dir, exist_ok=True)

records = []
t_start = time.time()
for e in entries:
    a = np.load(e["x0_file"])   # other-class point
    c = np.load(e["x1_file"])   # certified point
    t0 = time.time()
    found, qc, qa, diags = search_record(c, a)
    rec = {
        "index": e.get("index"),
        "stored_max_eps": float(e["max_eps"]),
        "stored_dist": float(e["dist"]),
        "found": found is not None,
        "per_t": diags,
    }
    if found is not None:
        assert on_grid(qa) and on_grid(qc)
        ok, max_eps = A.check_counter_example(qa, qc)
        d = l2(qa, qc)
        verified = bool(ok) and max_eps is not None and float(max_eps) >= d
        rec.update({
            "verified": verified,
            "max_eps": float(max_eps) if ok else None,
            "dist": d,
            "t": found["t"], "s": found["s"], "how": found["how"],
            "R": found["R"], "label_qc": found["label_qc"], "label_qa": found["label_qa"],
            "label_c": label(c),
            "repair_moves": found["repair_moves"],
            "pixels_differ": int(np.sum(qa != qc)),
        })
        if verified:
            idx = rec["index"]
            np.save(os.path.join(points_dir, f"idx{idx}_qc.npy"), f64(qc).reshape(c.shape))
            np.save(os.path.join(points_dir, f"idx{idx}_qa.npy"), f64(qa).reshape(a.shape))
        else:
            rec["found"] = False
            rec["verify_failed"] = True
    rec["secs"] = round(time.time() - t0, 1)
    records.append(rec)
    if rec["found"]:
        print(f"idx {rec['index']}: stored eps={rec['stored_max_eps']:.4g} | FOUND t={rec['t']} "
              f"how={rec['how']} dist={rec['dist']:.4g} R={rec['R']:.4g} max_eps={rec['max_eps']:.4g} "
              f"labels {rec['label_qc']}->{rec['label_qa']} ({rec['secs']}s)", flush=True)
    else:
        why = "; ".join(f"t={d['t']}:" + (d.get("skipped") or
                        (f"R={d['R']:.3g} flip@{d['first_flip']['dist']:.3g}" if d.get("first_flip")
                         else f"R={d['R']:.3g} no flip") +
                        (f" repair->{d['best_repair']['dist']:.3g}" if d.get("best_repair") else "") +
                        (f" nb:{d['neighbourhood']}" if d.get("neighbourhood") else ""))
                        for d in diags)
        print(f"idx {rec['index']}: stored eps={rec['stored_max_eps']:.4g} | not found ({rec['secs']}s) [{why}]",
              flush=True)

n = len(records)
found_eps = [r["max_eps"] for r in records if r["found"]]
summary = {
    "cex_dir": cex_dir,
    "format": A.fmt,
    "seed": seed,
    "n": n,
    "found": len(found_eps),
    "max_eps_max": max(found_eps) if found_eps else None,
    "max_eps_min": min(found_eps) if found_eps else None,
    "max_eps_median": float(np.median(found_eps)) if found_eps else None,
    "found_gt_0.1": sum(1 for v in found_eps if v > 0.1),
    "found_by_how": {h: sum(1 for r in records if r["found"] and r["how"] == h)
                     for h in ("ray", "ray+repair", "neigh1", "neigh2", "neigh3")},
    "allow_label_change": allow_label_change,
    "found_with_label_change": sum(1 for r in records if r["found"] and r["label_qc"] != r["label_c"]),
    "records_all_t_label_mismatch": sum(1 for r in records if r["per_t"] and
                                        all(d["label_qc"] != d["label_c"] for d in r["per_t"])),
    "records_no_R_ge_1_over_255": sum(1 for r in records if r["per_t"] and
                                      not any((d.get("R") or 0) >= 1.0 / LEVELS for d in r["per_t"])),
    "found_by_t": {str(t): sum(1 for r in records if r["found"] and r["t"] == t) for t in T_STEPS},
    "records_with_label_mismatch_at_t0": sum(1 for r in records if r["per_t"] and r["per_t"][0].get("skipped")),
    "verify_failed": sum(1 for r in records if r.get("verify_failed")),
    "total_secs": round(time.time() - t_start, 1),
}
if naive is not None:
    ns = naive["summary"]
    summary["naive_rounding"] = {
        "path": naive_path,
        "quantised_cex": ns.get("quantised_cex"),
        "quantised_max_eps_max": ns.get("quantised_max_eps_max"),
        "quantised_max_eps_min": ns.get("quantised_max_eps_min"),
        "quantised_cex_gt_0.1": ns.get("quantised_cex_gt_0.1"),
    }
print("\nSUMMARY:", json.dumps(summary, indent=1))
with open(out_path, "w") as f:
    json.dump({"summary": summary, "records": records}, f, indent=1)
print(f"Wrote {out_path} (points in {points_dir})")
