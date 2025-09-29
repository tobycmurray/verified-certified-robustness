 
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

# arbitrary precision math
from mpmath import mp, mpf, sqrt, nstr
mp.dps = 150  # massive precision

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
  
# ----------------------------
# Helpers for high-precision distances
# ----------------------------
def vector_to_mph(v):
    return list(map(mpf, np.asarray(v).flatten().tolist()))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

# ----------------------------
# CLI & model/dataset loading (kept the same)
# ----------------------------
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

# ----------------------------
# Load Lipschitz matrix (kept the same)
# ----------------------------
def load_lipschitz_matrix(path):
    with open(path, 'r') as f:
        data = json.load(f)
    for obj in data:
        if isinstance(obj, dict) and 'lipschitz_bounds' in obj:
            L = np.array(obj['lipschitz_bounds'], dtype=float)
            if L.shape != (num_classes, num_classes):
                print("Warning: L shape mismatch", L.shape)
            return L
    raise ValueError("Could not find 'lipschitz_bounds' in JSON")

L_matrix = load_lipschitz_matrix(lipschitz_json)
print("Loaded L matrix shape:", L_matrix.shape)

# ----------------------------
# Exact certifier (mp)  [PRESERVED]
# ----------------------------
def certifier_oracle_logits_mp(y_np, eps_val, i_star):
    y = np.asarray(y_np).ravel()
    n = y.shape[0]
    if not (0 <= i_star < n):
        raise ValueError("i_star out of range")
    mp_eps = mp.mpf(str(eps_val))
    y_i_mp = mp.mpf(str(float(y[i_star])))
    min_slack = mp.mpf('inf')
    for j in range(n):
        if j == i_star:
            continue
        Lij = mp.mpf(str(float(L_matrix[i_star, j])))
        y_j_mp = mp.mpf(str(float(y[j])))
        slack = y_i_mp - y_j_mp - Lij * mp_eps
        if slack < min_slack:
            min_slack = slack
        if slack <= mp.mpf('0'):
            return False, min_slack
    return True, min_slack

# ----------------------------
# Counter-example check  [PRESERVED, with minor variable fix]
# ----------------------------
def check_counter_example(x0, x1, model):
    """
    Returns (is_counterexample: bool, max_eps: mp.mpf or None)
    """
    y0 = model(x0, training=False).numpy()[0]
    y1 = model(x1, training=False).numpy()[0]
    y0_label = int(np.argmax(y0))
    y1_label = int(np.argmax(y1))
    if y0_label == y1_label:
        #print("x0 and x1 don't have different labels!")
        #print(f"y0: {y0}")
        #print(f"y1: {y1}")
        #print(f"y0_label: {y0_label}")
        #print(f"y1_label: {y1_label}")
        return False, None
    else:
        x0_mph = vector_to_mph(x0)
        x1_mph = vector_to_mph(x1)
        dist = l2_norm_mph(x0_mph, x1_mph)
        eps_val = dist
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
            while high > low + (mp.mpf('1e-80')):
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
        
# =====================================================================================
# New code: tie-search objective and pipeline
# =====================================================================================
def _ensure_batched(x, model_input_shape):
    """Make x -> (1, ...)."""
    x = np.asarray(x)
    if x.ndim == len(model_input_shape) - 1:
        x = x[np.newaxis, ...]
    return x

def _choose_competitor_idx(logits, i_star):
    """argmax over j != i_star."""
    logits = np.asarray(logits).ravel()
    idxs = np.arange(logits.shape[0])
    mask = idxs != i_star
    j_star = int(np.argmax(logits[mask]))
    # map back to original index
    return int(idxs[mask][j_star])

@tf.function(jit_compile=False)
def _smooth_max_excl_i(logits, i_star, temperature=0.0):
    """Return b = smooth/hard max over j != i_star."""
    C = tf.shape(logits)[-1]
    mask_i = tf.one_hot(i_star, C, on_value=tf.constant(-1e9, logits.dtype),
                        off_value=tf.constant(0., logits.dtype))
    z_wo_i = logits + mask_i
    if temperature and temperature > 0:
        return temperature * tf.reduce_logsumexp(z_wo_i / temperature, axis=-1)
    else:
        return tf.reduce_max(z_wo_i, axis=-1)
    
def tie_loss(logits, i_star, margin=0.0, beta=0.0, temperature=0.0, epsilon=0.0):
    """
    Encourage a tie between logit[i_star] and the strongest competitor, optionally
    keeping other classes below by 'margin'. A tiny 'epsilon' tilts in favor of the competitor.
    """
    C = tf.shape(logits)[-1]
    a = logits[..., i_star]  # z[i*]
    # b = max_{j != i*} z[j]
    mask_i = tf.one_hot(i_star, C, on_value=tf.constant(-1e9, logits.dtype),
                        off_value=tf.constant(0., logits.dtype))
    z_wo_i = logits + mask_i
    if temperature and temperature > 0:
        b = temperature * tf.reduce_logsumexp(z_wo_i / temperature, axis=-1)
    else:
        b = tf.reduce_max(z_wo_i, axis=-1)
    # margin against the rest (optional)
    j_top = tf.argmax(z_wo_i, axis=-1)
    mask_j = tf.one_hot(j_top, C, on_value=tf.constant(-1e9, logits.dtype),
                        off_value=tf.constant(0., logits.dtype))
    rest_max = tf.reduce_max(logits + mask_i + mask_j, axis=-1)
    tie_term = tf.square(a - b)
    sep_term = tf.nn.softplus(rest_max - tf.minimum(a, b) + margin)
    tilt_term = -epsilon * (b - a)
    return tie_term + beta * sep_term + tilt_term

def optimize_to_competitor(model, x_nat, i_star,
                           steps=400,
                           lr=5e-2,
                           lam_prox=1e-2,
                           temperature=0.0,
                           epsilon=1e-6,
                           clip_min=0.0, clip_max=1.0,
                           verbose=False):
    """
    Optimize the input towards a near tie between i_star and its best competitor.
    Returns (x_adv, logits_adv, j_star).
    """
    x_nat = _ensure_batched(x_nat, model.input_shape)
    x_var = tf.Variable(tf.cast(x_nat, tf.float32))
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    for t in range(steps):
        with tf.GradientTape() as tape:
            logits = model(x_var, training=False)
            L = tie_loss(logits, i_star, margin=100.0, beta=5.0,
                         temperature=temperature, epsilon=epsilon)
            if lam_prox > 0:
                L = L + lam_prox * tf.reduce_mean(tf.square(x_var - x_nat))
        grads = tape.gradient(L, x_var)
        opt.apply_gradients([(grads, x_var)])
        x_var.assign(tf.clip_by_value(x_var, clip_min, clip_max))
        if verbose and (t % 100 == 0):
            lv = float(L.numpy())
            print(f"  [opt] step {t:04d} loss {lv:.6e}")
    logits_adv = model(x_var, training=False).numpy()[0]
    j_star = _choose_competitor_idx(logits_adv, i_star)
    return x_var.numpy(), logits_adv, j_star

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

def refine_tie_bisection(model, x_nat, x_adv, i_star, j_star,
                         tie_tol=1e-12, max_iter=60):
    """
    On the segment x(t) = x_nat + t * (x_adv - x_nat), find t s.t.
    z_i(x) - z_j(x) ~ 0 to within tie_tol.
    Returns x_tie (close to exact tie) and also a slightly tilted x_tie_eps where j* wins.
    """
    x_nat = _ensure_batched(x_nat, model.input_shape)
    x_adv = _ensure_batched(x_adv, model.input_shape)
    def diff(x):
        z = model(x, training=False).numpy()[0]
        return float(z[i_star] - z[j_star])
    f0 = diff(x_nat)
    f1 = diff(x_adv)
    # ensure we are crossing: if not, just return x_adv as best effort
    if not (f0 > 0 and f1 < 0):
        return x_adv, x_adv  # best-effort fallback
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x_mid = x_nat + mid * (x_adv - x_nat)
        fm = diff(x_mid)
        if abs(fm) <= tie_tol:
            x_tie = x_mid
            break
        if fm > 0:
            lo = mid
        else:
            hi = mid
    else:
        x_tie = x_nat + 0.5 * (lo + hi) * (x_adv - x_nat)
    # tiny tilt along +direction so j* wins but keep ultra-close to tie
    # choose a step that’s many orders larger than float machine epsilon, but tiny in L2
    delta = 1e-9
    x_tie_eps = x_tie + delta * (x_adv - x_nat)
    return x_tie, x_tie_eps
 
def random_search(model, x0):
    x0 = np.asarray(x0, dtype=np.float32).reshape(1, 28, 28, 1)  # adjust shape
    initial_x0 = x0.copy()
    last_found=0
    working=[]
    low=0.0
    high=0.10
    eps=high
    x_try=x0
    y_try = model(x_try, training=False).numpy()[0]  
    argmax=np.argmax(y_try)
 
    x0_mph = vector_to_mph(x0)
    x_try_mph = vector_to_mph(x_try)
    l2_mph = l2_norm_mph(x_try_mph, x0_mph)
    w={"x": x_try,
       "y": y_try,
       "norm": np.linalg.norm(x_try-x0),
       "norm_mph": l2_mph,
       "argmax": argmax,
       "eps": eps,
       "flips": False}
    working.append(w)
    orig_argmax=argmax
    # generate noise once and then just scale it
    noise = np.random.randn(*x0.shape)
  
    x_try = x0 + eps * noise
    # model output
    y_try = model(x_try, training=False).numpy()[0]
    # condition: argmax is not 0
    argmax=np.argmax(y_try)
    y_try_mut=y_try.copy()
    y_try_mut[argmax] = float("-inf")
    second_argmax=np.argmax(y_try_mut)
    assert second_argmax != argmax
  
    print("Trying to flip...")
    while argmax != second_argmax:
        # generate noise once and then just scale it
        noise = np.random.randn(*x0.shape)
  
        x_try = x0 + eps * noise
        # model output
        y_try = model(x_try, training=False).numpy()[0]
        # condition: argmax is not 0
        argmax=np.argmax(y_try)
    print("Found flip. Attempting to find boundary...")
  
    x0_mph = vector_to_mph(x0)
    x_try_mph = vector_to_mph(x_try)
    l2_mph = l2_norm_mph(x_try_mph, x0_mph)
    w={"x": x_try,
       "y": y_try,
       "norm": np.linalg.norm(x_try-x0),
       "norm_mph": l2_mph,
       "argmax": argmax,
       "eps": eps,
       "flips": True}
    working.append(w)
    iter_count=0
    while high>=low:
        if iter_count % 100 == 0:
            print(f"Boundary search, space: {high-low}")
        iter_count += 1
      
        mid = (high+low)/2
        eps=mid
        x_try = x0 + eps * noise
        # model output
        y_try = model(x_try, training=False).numpy()[0]
        x_try_mph = vector_to_mph(x_try)
        l2_mph = l2_norm_mph(x_try_mph, x0_mph)
      
        argmax=np.argmax(y_try)
        if argmax != orig_argmax:
            # search closer to the start
            high=np.nextafter(mid, np.float32(-np.inf))          
            flips=True
        else:
            # search away from the start
            low=np.nextafter(mid, np.float32(np.inf))
            flips=False
        w={"x": x_try,
            "y": y_try,
            "norm": np.linalg.norm(x_try-x0),
            "norm_mph": l2_mph,
            "argmax": argmax,
            "eps": eps,             
            "flips": flips}
        working.append(w)
    return working

def closest_opposite_flips(items):
    """
    Finds the closest pair of dicts from `items` such that their 'flips' differ.
    Closeness is defined as minimal L2 distance between their 'x' vectors, computed
    with mpmath to ensure we don't lose precision due to floating point.
    """
    best_pair = None
    best_dist = float("inf")
    for a, b in combinations(items, 2):
        if a["flips"] != b["flips"]:
            a_mph = vector_to_mph(a["x"])
            b_mph = vector_to_mph(b["x"])
            dist = l2_norm_mph(a_mph, b_mph)
            if dist < best_dist:
                best_dist = dist
                best_pair = (a, b)
    return best_pair, best_dist

def best_counter_example_from_flips(items):
    print("Finding best counter-example (if any) from flips...")
    cex_count = 0
    best_pair = None
    best_dist = float("0.0")
    found_cex = False
    for a, b in combinations(items, 2):
        if a["flips"] != b["flips"]:
            if a["flips"]:
                flips = a
                other = b
            else:
                flips = b
                other = a
            # XXX
            is_cex, max_eps = check_counter_example(other["x"], flips["x"], model)
            if is_cex:
                found_cex = True
                cex_count += 1
                if max_eps > best_dist:
                    best_dist = max_eps
                    best_pair = (flips, other)
    if not found_cex:
        print("No counter-example found, unexpectedly!")
        return None
    print(f"Best counter-example found. Found {cex_count} in total.")
    return best_pair, best_dist

# this never seems to work
def try_to_improve_cex_by_mutation(x0, x1, model):
    cex, max_eps = check_counter_example(x0, x1, model)
    assert cex
    print(f"Trying to improve counter-example by mutating x1...")
    # OK now try to make it a bit bigger by mutation
    NUM_MUTANTS=100000
    scale=0.01
    max_eps_before_mut=max_eps
    cex_mut_count=0                  
    for i in range(NUM_MUTANTS):
        if i % (NUM_MUTANTS/100) == 0:
            print(f"[mutating {i/NUM_MUTANTS*100}%] cex_mut_count {cex_mut_count}")
        noise = np.random.randn(*x1.shape)
        x1_mut = x1 + (noise * scale)
        cex, max_eps_mut = check_counter_example(x0, x1_mut, model)
        if cex:
            cex_mut_count += 1
            if max_eps_mut > max_eps:
                max_eps = max_eps_mut
                x1 = x1_mut
    print(f"Mutation produced {cex_mut_count} ({cex_mut_count/NUM_MUTANTS*100}%) counter-examples")
    print(f"After mutation: Got counter-example with max_eps {max_eps}!")                  
    print(f"Mutation improved max_eps by: {max_eps - max_eps_before_mut}")
    return x1

def try_to_improve_cex_by_extension(x0, x1, model):
    cex, max_eps = check_counter_example(x0, x1, model)
    assert cex
 
       
    # we extend the line from x0 to x1 looking for the largest counter-example we can
    follow = (x1-x0)
 
    def check_for_steps(steps):
        a = follow * steps       
        steps_to_find=0
        cex=False
        max_noisy_search=5000
        while steps_to_find<max_noisy_search and not cex:              
            noise = np.random.normal(loc=0.0, scale=np.max(np.abs(a))*0.1, size=a.shape)
            x1=x0+a+noise
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
        print(f"[cex search] steps {steps}, max_max_eps: {max_max_eps}")
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
        print(f"[cex search] mid {mid}, low {low}, high {high}, max_max_eps: {max_max_eps}")       
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
  
# =====================================================================================
# Main loop over dataset
# =====================================================================================
log_file="counter_examples.json"

def main():
    # assuming x_test in [0,1]; adjust if your preprocessing differs
    total = x_test.shape[0]
    print(f"Total test points: {total}")
    found = 0
    results = []  # collect summaries
    # infer channel shape from model
    want_shape = model.input_shape[1:]
 
   
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

        # crank up amplitude
        #x_nat *= 20.0;
        #x_nat = np.clip(x_nat, 0.0, 1.0)

        # Model logits & correctness check
        y_nat = model(x_nat, training=False).numpy()[0]
        i_star = int(np.argmax(y_nat))
        if i_star != y_true:
            continue  # only consider correctly classified points
        print(f"Running with index {idx}. Optimising to competitor...")
      
        # Try to find a competitor by optimizing a tie objective
        x_adv, y_adv, j_star = optimize_to_competitor(
            model, x_nat, i_star,
            steps=1000, lr=5e-2,
            lam_prox=5e-3, temperature=0.0, epsilon=0.0, #epsilon=1e-6,
            clip_min=0.0, clip_max=1.0, verbose=True
        )
        # If argmax hasn't changed, still try to refine; otherwise proceed
        argmax_adv = int(np.argmax(y_adv))
        if argmax_adv == i_star:
            # try a few restarts with stronger tilt
            success = False
            for eps_try in [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]:
                print(f"Optimising to competitor failed. Re-trying with larger eps {eps_try}...")
                x_adv2, y_adv2, j2 = optimize_to_competitor(
                    model, x_nat, i_star,
                    steps=1000, lr=5e-2,
                    lam_prox=5e-3, temperature=0.0, epsilon=eps_try,
                    clip_min=0.0, clip_max=1.0, verbose=True
                )
                if int(np.argmax(y_adv2)) != i_star:
                    x_adv, y_adv, j_star = x_adv2, y_adv2, j2
                    success = True
                    break
            if not success:
                # give up on this point
                print(f"Failed to optimise to competitor for index {idx}. Skipping")
                continue
        print("Refining optimisation by bisection...")
        # Refine to an (almost) exact tie along the segment
        x_tie, x_tie_eps = refine_tie_bisection(
            model, x_nat, x_adv, i_star, j_star,
            tie_tol=0, max_iter=800
        )
        # Sanity: make sure the "tilted" point differs in argmax from i*
        y_tie_eps = model(x_tie_eps, training=False).numpy()[0]
        while int(np.argmax(y_tie_eps)) == i_star:
            # If still not flipped, nudge a bit more
            print("nudging x_tie_eps ... ")
            x_tie_eps = x_tie_eps + 1e-6 * (x_adv - x_nat)
            y_tie_eps = model(x_tie_eps, training=False).numpy()[0]
        # Now check counter-example using preserved routine
        is_cex, max_eps = check_counter_example(x_nat, x_tie_eps, model)
        x_other=x_nat
        y_other=y_nat
        if not is_cex:
            print("Searching for counter-example after finidng tie...")
            # go searching from x_other to find a point that gives a counter-example. Ideas:
            # 1. Search along the line segment x_other <--> x_tie_eps
            # 2. Optimixe x_other to x_other' that has minimal slack (perhaps along that line?)
            # 3. Search outwards from x_tie_eps (how is that different from 1?)
            # try to find the exact boundary from this point
            working = random_search(model, x_tie_eps)
            if working == []:
                continue
            res = best_counter_example_from_flips(working)
            if res is None:
                continue
            best_pair, best_dist = res
            print(f"best pair dist is: {best_dist}")
            is_flip,is_not_flip = best_pair
            print(f"item is_flip:")
            print(f"      y: {is_flip['y']}")
            print(f" argmax: {is_flip['argmax']}")
            print(f"  flips: {is_flip['flips']}")
            print(f"item is_not_flip:")
            print(f"      y: {is_not_flip['y']}")
            print(f" argmax: {is_not_flip['argmax']}")      
            print(f"  flips: {is_not_flip['flips']}")
 
            # do a final test in float32 arithmetic
            y_is_not_flip = model(is_not_flip["x"], training=False).numpy()[0]      
            y_is_flip = model(is_flip["x"], training=False).numpy()[0]
            argmax_y_is_not_flip = np.argmax(y_is_not_flip)
            argmax_y_is_flip = np.argmax(y_is_flip)
            if argmax_y_is_not_flip == argmax_y_is_flip:
                print("float32 argmaxes do not compute as expected!")
                print(f"y_is_not_flip: {y_is_not_flip}")
                print(f"argmax_y_is_not_flip (expected 0): {argmax_y_is_not_flip}")
                print(f"y_is_flip: {y_is_flip}")
                print(f"argmax_y_filps (expected non-zero): {argmax_y_is_flip}")
                sys.exit(1)
            else:
                x_is_not_flip_mph = vector_to_mph(is_not_flip["x"])
                x_is_flip_mph = vector_to_mph(is_flip["x"])
                dist = l2_norm_mph(x_is_not_flip_mph, x_is_flip_mph)          
                print(f"Manually confirmed x0 and x1 such that ||x1-x0||=={dist}, F(x0)=={argmax_y_is_not_flip} but F(x1)=={argmax_y_is_flip}")
              
                x0 = is_not_flip["x"]
                x1 = is_flip["x"]
                cex, max_eps = check_counter_example(x0, x1, model)
                if not cex:
                    print("x0 and x1 do not constitute a counter-example, unexpectedly!")
                    sys.exit(1)
          
 
                print(f"Got counter-example with max_eps {max_eps}!")
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
 
