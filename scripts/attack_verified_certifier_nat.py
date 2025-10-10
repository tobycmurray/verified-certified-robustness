 
import doitlib
import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import os
import json
import sys
import signal
import random
import tempfile
      
 
from itertools import combinations
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer
from rational_dense_net import *

from tensorflow.keras import mixed_precision

# set precision here: FIXME make this a command line argument
mixed_precision.set_global_policy("bfloat16")

print(f"Running models with precision: {mixed_precision.global_policy()}")

# arbitrary precision math
from mpmath import mp, mpf, sqrt, nstr, floor

mp.dps = 60  # massive precision

def round_down(x, decimals=0):
    """
    Round down (toward -∞) an mpf to a fixed number of decimal places.
    """
    factor = mp.mpf(10) ** decimals
    return floor(x * factor) / factor


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
  
def vector_to_mph(v):
    return list(map(mpf, np.asarray(v).flatten().tolist()))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

if len(sys.argv) != 6:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES model_weights_csv_dir input_size lipschitz_json\n")
    sys.exit(1)
dataset = sys.argv[1]
INTERNAL_LAYER_SIZES = eval(sys.argv[2])
csv_loc = sys.argv[3] + "/"
input_size = int(sys.argv[4])
lipschitz_json = sys.argv[5]
inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size,
                                      dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)
doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)
num_classes = int(model.output_shape[-1])
x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)

def load_lipschitz_matrix(path):
    with open(path, 'r') as f:
        data = json.load(f, parse_float=mp.mpf)
    for obj in data:
        if isinstance(obj, dict) and 'lipschitz_bounds' in obj:
            L = np.array(obj['lipschitz_bounds'])
            if L.shape != (num_classes, num_classes):
                print("Warning: L shape mismatch", L.shape)
            return L
    raise ValueError("Could not find 'lipschitz_bounds' in JSON")

L_matrix = load_lipschitz_matrix(lipschitz_json)
print("Loaded L matrix shape:", L_matrix.shape)

def certifier_oracle_logits_mp(y_np, eps_val, i_star):
    y = np.asarray(y_np).ravel()
    n = y.shape[0]
    if not (0 <= i_star < n):
        raise ValueError("i_star out of range")
    mp_eps = eps_val
    y_mp = vector_to_mph(y)
    y_i_mp = y_mp[i_star]
    min_slack = mp.mpf('inf')
    for j in range(n):
        if j == i_star:
            continue
        Lij = L_matrix[i_star, j]
        y_j_mp = y_mp[j]
        slack = y_i_mp - y_j_mp - Lij * mp_eps
        if slack < min_slack:
            min_slack = slack
        if slack <= mp.mpf('0'):
            return False, min_slack
    return True, min_slack

def check_counter_example(x0, x1, model, verbose=False):
    y0 = model(x0, training=False).numpy()[0]
    y1 = model(x1, training=False).numpy()[0]
    y0_label = int(np.argmax(y0))
    y1_label = int(np.argmax(y1))
    if y0_label == y1_label:
        if verbose:
            print("x0 and x1 don't have different labels!")
            print(f"y0: {y0}")
            print(f"y1: {y1}")
            print(f"y0_label: {y0_label}")
            print(f"y1_label: {y1_label}")
        return False, None
    else:
        x0_mph = vector_to_mph(x0)
        x1_mph = vector_to_mph(x1)
        dist = l2_norm_mph(x0_mph, x1_mph)
        eps_val = dist
        if verbose:
            print(f"||x1-x0|| = {dist}")
        cert, slack = certifier_oracle_logits_mp(y1, eps_val, y1_label)
        if cert:
            # grow eps until certification fails
            while cert:
                max_eps = eps_val
                eps_val = eps_val * mp.mpf('1.1')
                cert, slack = certifier_oracle_logits_mp(y1, eps_val, y1_label)
            # bisection to refine boundary
            high = eps_val
            low = max_eps
            while high > low + (mp.mpf('1e-20')):
                mid = low + (high - low) / 2
                cert, _ = certifier_oracle_logits_mp(y1, mid, y1_label)
                if cert:
                    low = mid
                else:
                    high = mid
            max_eps = low
            return True, max_eps
        else:
            return False, None
        


# version-safe import
try:
    from art.estimators.classification import TensorFlowV2Classifier    
    from art.attacks.evasion import DeepFool as _DeepFool
except Exception as e:
    raise ImportError("DeepFool not available in this ART version") from e

def _ensure_batched(x, model_input_shape):
    x = np.asarray(x)
    if x.ndim == len(model_input_shape) - 1:
        x = x[np.newaxis, ...]
    return x

def optimize_to_competitor_deepfool_art(
    art_classifier,
    x_nat,
    *,
    steps=100,            # DeepFool iterations
    overshoot=1e-3,       # ART's 'epsilon'
    verbose=False,
):
    """
    ART DeepFool wrapper which returns:
      returns (x_adv, logits_adv, j_star)

    art_classifier: an ART classifier (e.g., TensorFlowV2Classifier)
    x_nat:          np.ndarray (H,W,C) or (1,H,W,C), float32 in [0,1]
    """
    # batch x
    # We don't need the raw TF model here; ART handles predict() for us.
    x_nat_b = _ensure_batched(x_nat, art_classifier.input_shape)

    # configure ART DeepFool
    params = dict(
        classifier=art_classifier,
        max_iter=int(steps),
        epsilon=float(overshoot),
        verbose=bool(verbose),
    )

    attack = _DeepFool(**params)

    # run DeepFool
    x_adv_b = attack.generate(x=x_nat_b.copy())
    x_adv = np.asarray(x_adv_b)[0]

    # logits via ART (probabilities or logits; argmax works either way)
    if hasattr(art_classifier, "predict_logits"):
        logits_adv = art_classifier.predict_logits(x_adv[None, ...])[0]
    else:
        logits_adv = art_classifier.predict(x_adv[None, ...])[0]
    j_star = int(np.argmax(logits_adv))

    # match return types: x_adv batched, logits 1D
    return x_adv[None, ...], np.asarray(logits_adv), j_star

def save_x_to_image(x_final):
    # Normalize x_final to be in the valid image range [0, 255]
    x_final = (x_final - x_final.min()) / (x_final.max() - x_final.min())  # Normalize to [0,1]
    x_final = (x_final * 255).astype(np.uint8)  # Scale to [0,255]
    # Check the shape and adjust for grayscale images (MNIST)
    if x_final.shape[0] == 1:  # Remove batch dimension if present
        x_final = x_final[0]
      
    if x_final.shape[-1] == 1:  # MNIST has an extra channel dimension (28,28,1)
        x_final = x_final.squeeze(-1)  # Remove channel dimension to get (28,28)
    # Convert to PIL image
    image_mode = "L" if x_final.ndim == 2 else "RGB"  # 'L' for grayscale, 'RGB' for color
    image = Image.fromarray(x_final, mode=image_mode)
    # Create a unique temporary file in the current directory
    with tempfile.NamedTemporaryFile(prefix="x_", suffix=".png", dir=".", delete=False) as f:
        output_file = f.name
    image.save(output_file)
    print(f"Image saved {output_file}")
    return output_file
 
def save_x_to_file(x):
    # Create a unique temporary file in the current directory
    with tempfile.NamedTemporaryFile(prefix="x_", suffix=".npy", dir=".", delete=False) as f:
        output_file = f.name
    np.save(output_file, x)
    return output_file

def search_cex(model, x_nat, x_adv, verbose=False):
    x_nat = _ensure_batched(x_nat, model.input_shape)
    x_adv = _ensure_batched(x_adv, model.input_shape)
    y_nat = model(x_nat, training=False).numpy()[0]
    y_adv = model(x_adv, training=False).numpy()[0]
    i_star = np.argmax(y_nat)
    j_star = np.argmax(y_adv)

    low  = np.float64(0.0)
    high = np.float64(1.0)
    x_nat64 = tf.cast(x_nat, tf.float64)
    x_adv64 = tf.cast(x_adv, tf.float64)
    vec64   = x_adv64 - x_nat64

    x_low=x_nat64
    x_high=x_adv64

    def to_model_dtype(x64):
        compute_dtype = getattr(getattr(model, "dtype_policy", None), "compute_dtype", None)
        if compute_dtype is None:
            return x64  # many models accept float64; otherwise Keras will auto-cast
        return tf.cast(x64, compute_dtype)


    cex, max_eps = check_counter_example(x_low, x_high, model)
    steps=0
    # invariant argmax(y_low) == i_star and argmax(y_high) == j_star
    while high>=low and not cex:
        if verbose and steps%10==0:
            print(f"[search cex] steps {steps}, remaining search width {high-low}")
        mid = np.float64(0.5) * (low + high)
        # multiply in float64: cast 'mid' to a tf.float64 scalar
        mid_tf = tf.constant(mid, dtype=tf.float64)
        x_mid64 = x_nat64 + mid_tf * vec64                     # all float64 math here
        x_mid = to_model_dtype(x_mid64)        
        y_mid = model(x_mid, training=False).numpy()[0]
        argmax = np.argmax(y_mid)

        if argmax == i_star:
            x_low=x_mid
            low=np.nextafter(mid, np.float64(np.inf))
        elif argmax == j_star:
            x_high=x_mid
            high=np.nextafter(mid, np.float64(-np.inf))
        else:
            # found something unexpected here: search towards x_adv
            low=np.nextafter(mid, np.float64(np.inf))
            
        steps += 1
        cex, max_eps = check_counter_example(x_low, x_high, model)        
        
    y_low = model(x_low, training=False).numpy()[0]
    y_high = model(x_high, training=False).numpy()[0]
    assert i_star == np.argmax(y_low)
    assert j_star == np.argmax(y_high)

    # probably unnecssary to do this again
    cex, max_eps = check_counter_example(x_low, x_high, model)
    
    if verbose:
        print("search_cex returning: ")
        print(f"  y_low  (argmax {i_star}: {y_low}")
        print(f"  y_high (argmax {j_star}: {y_high}")
    return x_low.numpy(), x_high.numpy(), cex, max_eps

def try_to_improve_cex_by_extension(x0, x1, model):
    cex, max_eps = check_counter_example(x0, x1, model)
    assert cex
 
       
    # we extend the line from x0 to x1 looking for the largest counter-example we can
    follow = (x1-x0)
 
    def check_for_steps(steps):
        a = follow * steps       
        steps_to_find=0
        cex=False
        max_noisy_search=2000
        while steps_to_find<max_noisy_search and not cex:              
            noise = np.random.normal(loc=0.0, scale=np.max(np.abs(a))*0.1, size=a.shape)
            x1=np.clip(x0+a+noise, 0.0, 1.0)
            cex, max_eps = check_counter_example(x0, x1, model)
            steps_to_find=steps_to_find+1
        return cex, max_eps, x1
   
    max_max_eps = 0
    steps=0
    last_cex_found=-1
    max_max_eps = max_eps
    best_x1 = x1
    best_step = 0
    steps = 1
    while cex:
        print(f"[cex extend] steps {steps}, max_max_eps: {max_max_eps}")
        steps *= 2
 
        cex, max_eps, x1 = check_for_steps(steps)
        if cex:
            if max_eps > max_max_eps:
                max_max_eps = max_eps
                best_x1 = x1
                best_step = steps
 
    low=steps/2+1
    high=steps-1
    while high >= low:
        mid = int(low + (high - low)/2)
        print(f"[cex extend] mid {mid}, low {low}, high {high}, max_max_eps: {max_max_eps}")       
        cex, max_eps, x1 = check_for_steps(mid)
        if cex:
            if max_eps > max_max_eps:
                max_max_eps = max_eps
                best_x1 = x1
                best_step = mid
            low=mid+1
        else:
            high=mid-1
   
    x_final=best_x1
    y_final=model(x_final, training=False).numpy()[0]
    eps_final=max_max_eps
    print(f"Best counter-example found on step: {best_step}")
    return x_final, best_step


#################################################################################
#
# Warning: The following function was vibe-coded with ChatGPT
#
#################################################################################
def extend_cex_multi_ray(
    model,
    x_a, x_b,
    L_matrix,                         # (C,C) numpy float (used in float64)
    *,
    # ---- outer loop / geometry ----
    outer_rounds=120,
    clip_min=0.0, clip_max=1.0,
    alpha_headroom=0.75,
    radial_bisect_iters=40,
    eps_accept=1e-13,                 # minimal outward gain to accept
    adapt_label=False,                # if True, i* can change to current argmax

    # ---- Phase A (tangential, limiting facet) ----
    phaseA_steps=300,
    phaseA_lr=7e-3,
    phaseA_mode="limiting",           # accepted for compat; only "limiting" is used
    soft_tau=0.08,                    # used in micro-polish soft-min across near-limiting set
    hybrid_switch=3.0,                # compat no-op

    # ---- keep-label regularizer ----
    gamma_keep=1e-3, m_keep=0.0,

    # ---- certification floor ----
    kappa=1e-8,                       # min slack we require at the current radius

    # ---- reseeding / rays (single-stage) ----
    jitter_frac=0.35,
    stagnation_patience=5,            # rounds without outward accept before reseed
    max_reseeds=30,

    # ---- reseed warmup (single-stage only) ----
    rays_stage1=50, pre_steps=20, pre_lr=6e-4,

    verbose=True,
):
    """
    Certified counter-example extension in an L2 ball around x_a:
      • Phase A (tangential): maximize limiting margin (z_i - z_{j_lim}) on the L2 sphere
            - [NEW] LR decay on backtracks + diagnostics
      • Phase B (radial): expand (feasibility) from a guardrailed seed, then bisect in [lo,hi]
            - Seed and cap come from the old L+dir-deriv bound, but search is feasibility-led
            - Use a top-K near-limiting set to build a more reliable cap
      • After accept: micro-polish (short tangential soft-min) — [OFF by default in this drop-in]
      • Reseeding: single-stage multi-ray warmup when stagnated

    Returns:
      x_b_prime (np.float32, batched like model input),
      info = {"eps": float64 radius, "slack": float64 min_slack, "class": i_star}
    """
    import numpy as np
    import tensorflow as tf

    # ---------------- helpers ----------------
    def _ensure_batched(x):
        x = np.asarray(x)
        want_nd = len(getattr(model, "input_shape", x.shape))  # includes batch dim if available
        if x.ndim == max(1, want_nd - 1):
            x = x[None, ...]
        return x.astype(np.float32)

    def _flat(x):
        return tf.reshape(x, (tf.shape(x)[0], -1))

    def eps64(x_tf):
        # exact L2 radius from x_a (float64)
        d = tf.cast(_flat(x_tf - x_a_tf), tf.float64)
        return tf.norm(d, ord='euclidean', axis=1)

    def unit_radial(x_tf):
        # u = (x - x_a) / ||x - x_a||_2 (per-example), reshaped to x_tf
        d = _flat(x_tf - x_a_tf)
        n = tf.norm(d, axis=1, keepdims=True) + 1e-20
        u = d / n
        return tf.reshape(u, tf.shape(x_tf))

    def project_to_radius(x_tf, r_target):
        # exact L2 projection to the sphere of radius r_target around x_a, then clip box
        u = unit_radial(x_tf)
        x_proj = x_a_tf + tf.cast(r_target, x_tf.dtype) * u
        return tf.clip_by_value(x_proj, clip_min, clip_max)

    def keep_label_penalty(logits, i_star):
        # softplus(max_others - z_i + m_keep) encourages keeping i_star on top
        z = logits[0]
        C = tf.shape(z)[0]
        zi = z[i_star]
        idx = tf.concat([tf.range(0, i_star), tf.range(i_star + 1, C)], axis=0)
        z_others = tf.gather(z, idx)
        gap = tf.reduce_max(z_others) - zi + m_keep
        return tf.nn.softplus(gap)

    # certifier-style min slack and limiting j* (numpy float64), at radius r
    def min_slack_np(z_vec, r, i_star):
        z = np.asarray(z_vec, dtype=np.float64)
        C = z.shape[0]
        idxs = np.concatenate([np.arange(0, i_star), np.arange(i_star + 1, C)], axis=0)
        Lrow = np.asarray(L_matrix, np.float64)[i_star, idxs]
        vals = (z[i_star] - z[idxs]) - Lrow * float(r)   # larger is better; require ≥ kappa
        j_rel = int(np.argmin(vals))
        return float(vals[j_rel]), int(idxs[j_rel])

    # near-limiting set (top-K by slack s_j = (z_i - z_j) - L_{i,j} r) -> absolute indices, slacks, Lvals
    def near_limiting_set(z_vec, r, i_star, K=4):
        z = np.asarray(z_vec, dtype=np.float64)
        C = z.shape[0]
        idxs = np.concatenate([np.arange(0, i_star), np.arange(i_star + 1, C)], axis=0)
        Lrow_all = np.asarray(L_matrix, np.float64)[i_star, idxs]
        s_vals = (z[i_star] - z[idxs]) - Lrow_all * float(r)
        order = np.argsort(s_vals)  # ascending: smaller slack = more limiting
        k = int(min(K, len(order)))
        sel_rel = order[:k]
        return idxs[sel_rel].astype(int), s_vals[sel_rel], Lrow_all[sel_rel]

    # directional derivative of (z_i - z_j) along u at x (scalar)
    def dir_deriv_margin(x, i_star, j_idx, u):
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = model(x, training=False)[0]
            m = z[i_star] - z[j_idx]
        g = tape.gradient(m, x)
        return tf.reduce_sum(g * u)

    # --- single-stage reseed (wide & cheap) ---
    def _warmup_single_stage(x_center, radius, i_star, z_center, ms_center,
                             rays_stage1=rays_stage1, pre_steps=pre_steps, pre_lr=pre_lr, jitter=jitter_frac):
        if verbose:
            print(f"[reseed] S1 {rays_stage1} rays (pre {pre_steps}) @ r={radius:.3e}, base_slack={ms_center:.3e}")

        def warmup_once(cand, steps, lr):
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
            for _ in range(steps):
                # recompute limiting j each step for robustness
                zc = model(cand, training=False).numpy()[0]
                ms_c, j_lim_c = min_slack_np(zc, radius, i_star)
                with tf.GradientTape() as tape:
                    tape.watch(cand)
                    logits = model(cand, training=False)
                    z = logits[0]
                    mA = z[i_star] - z[j_lim_c]
                    keep_pen = keep_label_penalty(logits, i_star)
                    lossW = -mA + gamma_keep * keep_pen
                g = tape.gradient(lossW, cand)
                if g is None:
                    break
                u_loc = unit_radial(cand)
                g_tan = g - tf.reduce_sum(g * u_loc) * u_loc
                opt.apply_gradients([(g_tan, cand)])
                cand.assign(project_to_radius(cand, radius))
            zc = model(cand, training=False).numpy()[0]
            ms_c, _ = min_slack_np(zc, radius, i_star)
            return ms_c, cand

        best_x = x_center.numpy()
        best_ms = ms_center

        # current candidate included
        u0 = unit_radial(x_center)
        s1 = []
        for _ in range(rays_stage1):
            n = tf.random.normal(tf.shape(x_center), dtype=x_center.dtype)
            n_tan = n - tf.reduce_sum(n * u0) * u0
            n_tan /= (tf.norm(_flat(n_tan)) + 1e-20)
            d = u0 + jitter * n_tan
            d /= (tf.norm(_flat(d)) + 1e-20)
            cand = tf.Variable(tf.cast(tf.clip_by_value(x_a_tf + radius * d, clip_min, clip_max), tf.float32))
            ms_c, cand = warmup_once(cand, pre_steps, pre_lr)
            s1.append((ms_c, cand))
        s1.append((ms_center, tf.Variable(x_center.numpy())))

        # pick best
        s1.sort(key=lambda t: t[0], reverse=True)
        best_ms, best_cand = s1[0]
        if best_ms >= kappa and best_ms > ms_center:
            best_x = best_cand.numpy()

        if verbose:
            print(f"[reseed] best_slack={best_ms:.3e} (Δ={best_ms - ms_center:.3e})")
        return best_x, best_ms

    # ---------------- setup ----------------
    x_a = _ensure_batched(x_a)
    x_b = _ensure_batched(x_b)
    x_a_tf = tf.constant(x_a, dtype=tf.float32)
    xb_var = tf.Variable(x_b, dtype=tf.float32)

    # current class (optionally adaptive)
    z0 = model(xb_var, training=False).numpy()[0]
    i_star = int(np.argmax(z0))
    r = float(eps64(xb_var)[0].numpy())
    ms, j_lim = min_slack_np(z0, r, i_star)
    best = {"eps": r, "x": xb_var.numpy(), "slack": ms, "class": i_star}

    if verbose:
        print(f"[init] eps_dist={best['eps']:.12g} min_slack={best['slack']:.6g} (class={i_star})")

    optA = tf.keras.optimizers.legacy.Adam(learning_rate=phaseA_lr)
    rounds_since_accept = 0
    reseeds_done = 0
    accepted_count = 0

    # ---- constants used inside Phase-B (no signature change) ----
    _PB_TOPK = 4                   # near-limiting set size
    _PB_CAP_GUARD = 0.10           # allow +10% past cap before stopping expansion

    # [NEW] Micro-polish OFF by default to reproduce the better plateau baseline.
    #       Set this to 20 (or similar) to re-enable polish.
    _PB_MICRO_STEPS = 0
    _PB_MICRO_LR = float(phaseA_lr) * 0.5

    # [NEW] Phase-A LR decay on backtracks (simple, robust policy)
    _PA_DECAY = 0.5                # multiply lr by this when a step needed any backtracking
    _PA_RECOVER = 1.05             # gentle recovery multiplier when no backtracks
    _PA_LR_MIN = 1e-4              # lower bound on lr_scale * phaseA_lr
    _PA_LR_MAX = 1.0               # upper bound on lr_scale (× phaseA_lr)

    for round_idx in range(outer_rounds):
        if adapt_label:
            i_star = int(np.argmax(model(xb_var, training=False).numpy()[0]))

        if verbose:
            print(f"\n[round {round_idx+1}/{outer_rounds}] start r={r:.12g}, slack={ms:.6g}, i*={i_star}")

        # ---------- Phase A: limiting facet at fixed radius r ----------
        # [NEW] diagnostics accumulators
        pa_bt_sum = 0
        pa_bt_max = 0
        pa_steps = 0
        # [NEW] per-round adaptive lr scale (starts at 1.0 each round)
        lr_scale = 1.0

        for t in range(phaseA_steps):
            # refresh j_lim periodically
            if t == 0 or (t % 10 == 0):
                z_now = model(xb_var, training=False).numpy()[0]
                ms_tmp, j_lim = min_slack_np(z_now, r, i_star)

            with tf.GradientTape() as tape:
                tape.watch(xb_var)
                logits = model(xb_var, training=False)
                z = logits[0]
                mA = z[i_star] - z[j_lim]
                keep_pen = keep_label_penalty(logits, i_star)
                lossA = -mA + gamma_keep * keep_pen

            g = tape.gradient(lossA, xb_var)
            if g is None:
                break

            u = unit_radial(xb_var)
            g_tan = g - tf.reduce_sum(g * u) * u

            # [NEW] apply per-round LR scaling
            lr_eff = float(np.clip(lr_scale, _PA_LR_MIN / max(phaseA_lr, 1e-20), _PA_LR_MAX))
            if hasattr(optA, "learning_rate") and isinstance(optA.learning_rate, tf.Variable):
                optA.learning_rate.assign(phaseA_lr * lr_eff)

            prev = tf.identity(xb_var)
            optA.apply_gradients([(g_tan, xb_var)])
            xb_var.assign(project_to_radius(xb_var, r))

            # feasibility (slack ≥ kappa); simple backtrack if violated
            z_now = model(xb_var, training=False).numpy()[0]
            ms_now, _ = min_slack_np(z_now, r, i_star)
            backtracks = 0
            while ms_now < kappa and backtracks < 6:
                xb_var.assign(0.5 * (prev + xb_var))
                xb_var.assign(project_to_radius(xb_var, r))
                z_now = model(xb_var, training=False).numpy()[0]
                ms_now, _ = min_slack_np(z_now, r, i_star)
                backtracks += 1

            # [NEW] update LR scale based on backtracks
            if backtracks > 0:
                lr_scale = max(lr_scale * _PA_DECAY, _PA_LR_MIN / max(phaseA_lr, 1e-20))
            else:
                lr_scale = min(lr_scale * _PA_RECOVER, _PA_LR_MAX)

            # [DIAG] track totals
            pa_bt_sum += backtracks
            pa_bt_max = max(pa_bt_max, backtracks)
            pa_steps += 1

        # compute current ms, limiting j at r (after Phase A)
        z_now = model(xb_var, training=False).numpy()[0]
        ms, j_lim = min_slack_np(z_now, r, i_star)
        if verbose:
            print(f"[phase A] slack at r={r:.12g}: {ms:.6g} (limiting j={j_lim})")
            # [DIAG] report Phase-A stability
            print(f"[diag A] steps={pa_steps}, backtracks(sum/max)={pa_bt_sum}/{pa_bt_max}, "
                  f"lr_scale_end≈{lr_scale:.3f}")

        # ---------- Phase B: feasibility-driven expand & bisect with guardrailed seed/cap ----------
        u = unit_radial(xb_var)

        # Build near-limiting set and compute (s_j, L_ij, gdrop_j)
        js, s_js, L_js = near_limiting_set(z_now, r, i_star, K=_PB_TOPK)
        eps_denom = 1e-12

        gdrop_js = []
        for j_abs in js:
            try:
                gdir_j = float(dir_deriv_margin(xb_var, tf.constant(i_star, tf.int32),
                                                tf.constant(int(j_abs), tf.int32), u).numpy())
            except Exception:
                gdir_j = 0.0
            gdrop_js.append(max(0.0, -gdir_j))
        gdrop_js = np.asarray(gdrop_js, dtype=np.float64)

        # Seed/cap from the guardrail: Δr_cap = α * min_j (s_j - κ)_+ / (L_ij + gdrop_j)
        denom = np.maximum(L_js + gdrop_js, eps_denom)
        headroom = np.maximum(s_js - float(kappa), 0.0)
        feasible_mask = np.isfinite(denom) & (denom > 0) & (headroom > 0)
        if np.any(feasible_mask):
            drs = headroom[feasible_mask] / denom[feasible_mask]
            dr_cap = float(alpha_headroom) * float(np.min(drs))
            dr_seed = 0.5 * dr_cap
        else:
            # fallback: conservative small relative step
            dr_cap = float(alpha_headroom) * max(1e-12, 0.05 * max(r, 1e-12))
            dr_seed = 0.5 * dr_cap

        dr_seed = float(max(dr_seed, 1e-14))
        dr_max = dr_cap * (1.0 + _PB_CAP_GUARD)

        expand_budget = max(8, radial_bisect_iters // 2)
        lo = 0.0
        hi = None
        dr = dr_seed
        r_best, ms_best = r, ms
        x_best = None

        def _feasible_delta(delta_r):
            cand = project_to_radius(xb_var + delta_r * u, r + delta_r)
            z_c = model(cand, training=False).numpy()[0]
            r_c = float(eps64(cand)[0].numpy())
            ms_c, _ = min_slack_np(z_c, r_c, i_star)
            return (ms_c >= kappa), cand, r_c, ms_c

        # Expand by feasibility (doubling) within cap guard
        expansions = 0  # [DIAG]
        for _ in range(expand_budget):
            if dr > dr_max:
                hi = dr
                break
            ok, cand, r_c, ms_c = _feasible_delta(dr)
            if ok:
                lo = dr
                x_best, r_best, ms_best = cand, r_c, ms_c
                dr *= 2.0
                expansions += 1
            else:
                hi = dr
                break

        accepted = False
        # If we never found an infeasible 'hi', but have a feasible 'lo', accept lo
        if hi is None and lo > 0.0 and (r_best > r + eps_accept):
            accepted = True
            xb_var.assign(x_best)
            r, ms = r_best, ms_best
            rounds_since_accept = 0
            accepted_count += 1
            if r > best["eps"]:
                best.update({"eps": r, "x": xb_var.numpy(), "slack": ms, "class": i_star})
            if verbose:
                print(f"[phase B] accepted r={r:.12g}, slack={ms:.6g} (Δr≥{lo:.3e}, cap≈{dr_cap:.3e})")
                # [DIAG] summarize guardrail & expansion
                s0 = float(s_js[0]) if len(s_js) > 0 else float("nan")
                s1 = float(s_js[1]) if len(s_js) > 1 else float("nan")
                print(f"[diag B] K={len(js)}, s0={s0:.3e}, s1={s1:.3e}, "
                      f"dr_seed={dr_seed:.3e}, dr_cap={dr_cap:.3e}, expand_steps={expansions}, bracket=False")

        # Otherwise bisect in [lo,hi] to the largest feasible radius
        elif hi is not None:
            for _ in range(radial_bisect_iters):
                mid = 0.5 * (lo + hi)
                ok, cand, r_c, ms_c = _feasible_delta(mid)
                if ok:
                    lo = mid
                    x_best, r_best, ms_best = cand, r_c, ms_c
                else:
                    hi = mid
            if (r_best > r + eps_accept):
                accepted = True
                xb_var.assign(x_best)
                r, ms = r_best, ms_best
                rounds_since_accept = 0
                accepted_count += 1
                if r > best["eps"]:
                    best.update({"eps": r, "x": xb_var.numpy(), "slack": ms, "class": i_star})
                if verbose:
                    print(f"[phase B] accepted r={r:.12g}, slack={ms:.6g} (Δr≈{lo:.3e}, cap≈{dr_cap:.3e})")
                    # [DIAG] summarize guardrail & expansion
                    s0 = float(s_js[0]) if len(s_js) > 0 else float("nan")
                    s1 = float(s_js[1]) if len(s_js) > 1 else float("nan")
                    print(f"[diag B] K={len(js)}, s0={s0:.3e}, s1={s1:.3e}, "
                          f"dr_seed={dr_seed:.3e}, dr_cap={dr_cap:.3e}, expand_steps={expansions}, bracket=True")

        if accepted and _PB_MICRO_STEPS > 0:
            # ---- micro-polish: short tangential soft-min over current near-limiting set ----
            try:
                optM = tf.keras.optimizers.legacy.Adam(learning_rate=_PB_MICRO_LR)
                revert = False
                z_before = model(xb_var, training=False).numpy()[0]
                ms_before, _ = min_slack_np(z_before, r, i_star)

                for _ in range(_PB_MICRO_STEPS):
                    # recompute near-limiting set on the new sphere
                    z_pol = model(xb_var, training=False).numpy()[0]
                    js_pol, _, _ = near_limiting_set(z_pol, r, i_star, K=_PB_TOPK)
                    with tf.GradientTape() as tape:
                        tape.watch(xb_var)
                        logits = model(xb_var, training=False)
                        z = logits[0]
                        # smooth min over m_j = z_i - z_j for j in near-limiting set
                        m_list = [z[i_star] - z[int(jp)] for jp in js_pol]
                        m_stack = tf.stack(m_list)
                        # soft-min: -tau * logsumexp(-m/tau)
                        mA = -float(soft_tau) * tf.reduce_logsumexp(-m_stack / float(soft_tau))
                        keep_pen = keep_label_penalty(logits, i_star)
                        lossM = -mA + gamma_keep * keep_pen
                    gM = tape.gradient(lossM, xb_var)
                    if gM is None:
                        break
                    u_now = unit_radial(xb_var)
                    g_tan = gM - tf.reduce_sum(gM * u_now) * u_now
                    prev = tf.identity(xb_var)
                    optM.apply_gradients([(g_tan, xb_var)])
                    xb_var.assign(project_to_radius(xb_var, r))
                    # ensure feasibility
                    z_now = model(xb_var, training=False).numpy()[0]
                    ms_now, _ = min_slack_np(z_now, r, i_star)
                    b = 0
                    while ms_now < kappa and b < 3:
                        xb_var.assign(0.5 * (prev + xb_var))
                        xb_var.assign(project_to_radius(xb_var, r))
                        z_now = model(xb_var, training=False).numpy()[0]
                        ms_now, _ = min_slack_np(z_now, r, i_star)
                        b += 1

                # finalize ms/j_lim after micro-polish
                z_after = model(xb_var, training=False).numpy()[0]
                ms_after, j_lim = min_slack_np(z_after, r, i_star)

                if verbose:
                    if ms_after >= ms_before:
                        print(f"[phase B] micro-polish → slack={ms_after:.6g}, j_lim={j_lim}")
                    else:
                        print(f"[phase B] micro-polish reverted (Δslack={ms_after - ms_before:.3e})")

                # revert if worse
                if ms_after < ms_before:
                    revert = True
                if revert:
                    xb_var.assign(project_to_radius(tf.constant(best["x"], dtype=xb_var.dtype), r))
                    z_now = model(xb_var, training=False).numpy()[0]
                    ms, j_lim = min_slack_np(z_now, r, i_star)
                else:
                    ms = ms_after

            except Exception:
                # polishing is opportunistic; ignore failures
                z_now = model(xb_var, training=False).numpy()[0]
                ms, j_lim = min_slack_np(z_now, r, i_star)
        else:
            if not accepted:
                if verbose:
                    print("[phase B] no feasible radial increase; will try another round.")
                rounds_since_accept += 1
            else:
                # [DIAG] signal that polish was skipped (it's OFF by default here)
                if verbose and _PB_MICRO_STEPS == 0:
                    print("[phase B] micro-polish skipped (disabled)")

        # ---------- reseed on stagnation ----------
        if (rounds_since_accept >= stagnation_patience) and (reseeds_done < max_reseeds):
            if verbose:
                print(f"[reseed] try rays at r={r:.3e}, base_slack={ms:.3e}")
            x_new, ms_new = _warmup_single_stage(xb_var, r, i_star, z_now, ms)
            if ms_new > ms + 1e-12 and ms_new >= kappa:
                xb_var.assign(x_new); ms = ms_new
                if verbose:
                    print(f"[reseed] accepted new seed at same r={r:.3e} with slack={ms:.3e}")
            rounds_since_accept = 0
            reseeds_done += 1

    return best["x"], {"eps": best["eps"], "slack": best["slack"], "class": best["class"]}


# =====================================================================================
# Main loop over dataset
# =====================================================================================
log_file="counter_examples.json"

certifier_input="certifier_input.txt"


def main():
    # assuming x_test in [0,1]; adjust if your preprocessing differs
    total = x_test.shape[0]
    print(f"Total test points: {total}")
    found = 0
    results = []  # collect summaries
    # infer channel shape from model
    want_shape = model.input_shape[1:]
 

    if os.path.exists(certifier_input):
        # ask user
        answer = input(f"{certifier_input} already exists. Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborting, not overwriting.")
            exit(1)
        os.remove(certifier_input)
    
    if os.path.exists(log_file):
        # ask user
        answer = input(f"{log_file} already exists. Overwrite? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborting, not overwriting.")
            exit(1)
        os.remove(log_file)
    with open(log_file, "w", buffering=1) as f:  # line-buffered mode
        f.write("[\n")
        f.flush()  # force flush, though buffering=1 already flushes on newline
    num_logs_written=0
    for idx in range(total):
        x_nat0 = x_test[idx]
        y_true = int(np.argmax(y_test[idx]))
 
        # Ensure shape matches model input (e.g., add channel)
        x_nat = _ensure_batched(x_nat0, model.input_shape)

        # Model logits & correctness check
        y_nat = model(x_nat, training=False).numpy()[0]
        i_star = int(np.argmax(y_nat))
        if i_star != y_true:
            continue  # only consider correctly classified points
        print(f"\nRunning with index {idx}. Optimising to competitor...")

        # Build the ART TensorFlowV2Classifier exactly as you do now...
        loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        input_shape = tuple(model.input_shape[1:])  # or x_nat.shape
        classifier = TensorFlowV2Classifier(
            model=model,
            loss_object=loss_obj,
            nb_classes=int(num_classes),
            input_shape=input_shape,
            clip_values=(0.0, 1.0),
        )

        # DeepFool step
        x_adv, y_adv, j_star = optimize_to_competitor_deepfool_art(
            classifier,
            x_nat,
            steps=100,          # try 50–150
            overshoot=1e-3,     # 1e-3 to 5e-3 often works well
            verbose=True,
        )

        # If argmax hasn't changed, still try to refine; otherwise proceed
        argmax_adv = int(np.argmax(y_adv))
        if argmax_adv == i_star:

            # try a few restarts with more steps and increased threshold
            success = False
            for steps_try in [200, 500, 1000]:
                print(f"Optimising to competitor failed. Re-trying with larger steps {steps_try}...")
                x_adv, y_adv, j_star = optimize_to_competitor_deepfool_art(
                    classifier,
                    x_nat,
                    steps=steps_try,
                    overshoot=5e-3, 
                    verbose=True,
                )

                if int(np.argmax(y_adv)) != i_star:
                    success = True
                    break
            if not success:
                # give up on this point
                print(f"Failed to optimise to competitor for index {idx}. Skipping")
                continue

        print("Searching for counter-example...")
        
        # Refine to an (almost) exact tie along the segment
        x0, x1, cex, max_eps = search_cex(model, x_nat, x_adv, verbose=False)
        if not cex:
            print("Search did not return a counter-example. Checking the other way around...")
            cex, max_eps = check_counter_example(x1, x0, model)
            if not cex:
                print("Other way around also wasn't a counter-example. Skipping...")
                continue
            print("Other way around is a counter-example. Swapping x0 and x1.")
            # swap x0 and x1
            temp=x1
            x1=x0
            x0=temp
        
 
        print(f"Got counter-example with max_eps {max_eps}!")

        # use vibe-coded whitebox search method to try to extend counter-example
        fast_dev = dict(
            outer_rounds=100,
            phaseA_steps=500, phaseA_lr=2e-4,
            phaseA_mode="hybrid", soft_tau=0.08, hybrid_switch=3.0,
            kappa=1e-8, gamma_keep=1e-3, m_keep=0.0,
            alpha_headroom=0.3, radial_bisect_iters=80, eps_accept=1e-13,
            adapt_label=False,
            jitter_frac=0.35, stagnation_patience=5, max_reseeds=30,
            rays_stage1=50, pre_steps=20, pre_lr=6e-4,
            clip_min=0.0, clip_max=1.0,
            verbose=True,
        )
        
        # --- (1) set seeds for repeatability ---
        np.random.seed(0)
        tf.random.set_seed(0)

        params = dict(**fast_dev)

        # --- (2) run the extension ---
        x_b_prime, info = extend_cex_multi_ray(
            model,
            x0, x1,
            L_matrix,
            **params
        )

        # check we got a counter-example
        cex, max_eps = check_counter_example(x0, x_b_prime, model, verbose=True)
        print("certified?", cex, "eps*", max_eps)
        if not cex:
            print("Whitebox didn't give a counter-example! Moving on...")
            continue
        x1 = x_b_prime
        
        # mutation doesn't seem to help improve counter-examples
        print(f"Trying to improve counter-example by extension...")
        x1, best_step = try_to_improve_cex_by_extension(x0, x1, model)
        cex, max_eps = check_counter_example(x0, x1, model)
        assert cex
        print(f"Got counter-example with max_eps {max_eps}!")
        y0 = model(x0, training=False).numpy()[0]      
        y1 = model(x1, training=False).numpy()[0]
        argmax_y0 = np.argmax(y0)
        argmax_y1 = np.argmax(y1)
        
        img_file = save_x_to_image(x1)
        x1_file = save_x_to_file(x1)
        x0_file = save_x_to_file(x0)
      
        x0_mph = vector_to_mph(x0)
        x1_mph = vector_to_mph(x1)
        dist = l2_norm_mph(x0_mph, x1_mph)          
        print(f"Manually confirmed x0 and x1 such that ||x1-x0||=={dist}, F(x0)=={argmax_y0} but F(x1)=={argmax_y1}")
        summary = {
            "index": int(idx),
            "true_label": int(y_true),
            "argmax_y0": int(argmax_y0),
            "argmax_y1": int(argmax_y1),
            "is_counter_example": bool(cex),
            "max_eps": str(max_eps),
            "dist": str(dist),
            "img_file": str(img_file),
            "x0_file": str(x0_file),
            "x1_file": str(x1_file),
            "y0": y0,
            "y1": y1,
            "extension_best_step": int(best_step),
        }      
 
        with open(certifier_input, "a", buffering=1) as f:
            dists = [max_eps, dist, (dist + max_eps)*mp.mpf("0.5")]
            for r in dists:
                radius = round_down(r, 20)
                for i in range(len(y1)):
                    try:
                        s = "{:.150f}".format(y1[i])
                    except Exception:
                        s = str(y1[i])
                    f.write(s)
                    if i<len(y1)-1:
                        f.write(",")
                f.write(" ")
                s = str(radius)
                f.write(s)
                f.write("\n")
                f.flush()
            
        # log to file
        with open(log_file, "a", buffering=1) as f:  # line-buffered mode
            if num_logs_written > 0:
                f.write(",\n")
            f.write(json.dumps(summary, indent=2, cls=NumpyEncoder) + "\n")
            f.flush()
            num_logs_written += 1
                              
        continue  # for now run for a while so we can see what sorts max_epses we get
              
 
def handle_interrupt(sig, frame):
    print("Caught CTRL-C.")
    # close out the log
    with open(log_file, "a", buffering=1) as f:  # line-buffered mode
        f.write("\n]\n")
        f.flush()  # force flush, though buffering=1 already flushes on newline
    sys.exit(1)
  
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)  
    main()
 
