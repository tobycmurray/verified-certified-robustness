 
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
#mixed_precision.set_global_policy("bfloat16")

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
        max_noisy_search=5000
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
 
