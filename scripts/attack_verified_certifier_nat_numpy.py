"""Counter-example generation against the IEEE-754-compliant numpy execution.

Copy of attack_verified_certifier_nat.py, adapted so that counter-examples are
found against the pure-numpy, IEEE-754 gradual-underflow forward pass
(python-certifier's compliant_forward.py) instead of the Keras/TensorFlow
execution (which on this machine runs with FTZ enabled and cannot be turned off).

Division of labour:
  - DeepFool (ART) still runs against the Keras model to find a boundary-crossing
    adversarial point x_adv. This is only a heuristic to locate the decision
    boundary; it does not need compliant semantics.
  - Everything downstream -- the bisection search for a tie point, the
    counter-example check (certifier oracle on the model's logits), and the
    blackbox noisy line-search extension -- runs against the numpy model, so the
    final counter-examples (x0, x1, y0, y1, certifier_input.txt) are pairs whose
    labels flip under the IEEE-754-compliant numpy execution.

The numpy model is built from the *same* weight tensors held by the (possibly
bias-amplified) Keras model, transposed to the certifier's (out,in) orientation
and executed via compliant_forward (per-op round-to-nearest, no BLAS, no FTZ).

The whitebox extension (extend_cex_multi_ray) from the original script is not
ported: it was already bypassed there (contributes ~1.00x) and it needs TF
gradients, which the numpy model doesn't provide. The per-cex JSON keeps the
"whitebox_extend" stage record (skipped) so the format matches the original.

Run under the ART-equipped venv (the python-certifier venv has no ART):
  verified-certified-robustness/scripts/cav2025-artifact-venv

Usage (same CLI as the original):
  python attack_verified_certifier_nat_numpy.py float_format dataset \
      INTERNAL_LAYER_SIZES model_weights_csv_dir input_size lipschitz_json [max_cex]
"""

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

# the IEEE-754-compliant numpy forward pass lives in python-certifier
PYTHON_CERTIFIER = os.environ.get(
    "PYTHON_CERTIFIER", os.path.expanduser("~/git/python-certifier"))
sys.path.insert(0, PYTHON_CERTIFIER)
import compliant_forward


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

if len(sys.argv) < 7 or len(sys.argv) > 8:
    print(f"Usage: {sys.argv[0]} float_format dataset INTERNAL_LAYER_SIZES model_weights_csv_dir input_size lipschitz_json [max_cex]\n")
    sys.exit(1)

max_cex = int(sys.argv[7]) if len(sys.argv) == 8 else None

fmt = sys.argv[1]

if fmt not in ("float16", "float32", "float64"):
    print(f"Unsupported float format for the numpy backend: {fmt} "
          "(compliant_forward supports float16/float32/float64)")
    sys.exit(1)

try:
    mixed_precision.set_global_policy(fmt)
except Exception as e:
    print("Failed to set precision", str(e))
    sys.exit(1)

print(f"Running Keras (DeepFool only) with precision: {mixed_precision.global_policy()}")
print(f"Counter-example search runs against compliant numpy at: {fmt}")


sys.argv = sys.argv[1:]

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


# =====================================================================
# Optional: bias amplification of FP errors (set FP_BIAS=1e6)
# =====================================================================
def build_biased_model(original_model, B=1000.0, position="begin"):
    """
    Build a new model with compensating biases that amplify FP errors.

    position="begin" (default):
      Layer 0 gets a flat bias b = B on every neuron.
      Layer 1 gets bias = -(b @ W1) to cancel the bias contribution.
      Remaining layers get zero bias.

    position="end":
      Second-to-last layer gets a flat bias b = B on every neuron.
      Last layer gets bias = -(b @ W_last) to cancel the bias contribution.
      All other layers get zero bias.

    The Lipschitz bound (product of spectral norms of weight matrices)
    is unchanged because biases don't affect the gradient.
    """
    dense_layers = [l for l in original_model.layers
                    if isinstance(l, tf.keras.layers.Dense)]
    n_dense = len(dense_layers)

    want_shape = tuple(original_model.input_shape[1:])
    inp = Input(want_shape)
    z = Flatten()(inp)
    for i, dl in enumerate(dense_layers):
        is_last = (i == n_dense - 1)
        activation = None if is_last else 'relu'
        z = Dense(dl.units, use_bias=True, activation=activation)(z)
    new_model = Model(inp, z)

    new_dense = [l for l in new_model.layers
                 if isinstance(l, tf.keras.layers.Dense)]

    if position == "begin":
        bias_layer = 0
        correction_layer = 1
    elif position == "end":
        bias_layer = n_dense - 2
        correction_layer = n_dense - 1
    else:
        raise ValueError(f"Unknown bias position '{position}', expected 'begin' or 'end'")

    print(f"  Bias position: {position} "
          f"(bias on layer {bias_layer}, correction on layer {correction_layer})")

    # Bias layer: flat bias to create large intermediates
    W_bias = dense_layers[bias_layer].get_weights()[0]
    b_bias = np.full(W_bias.shape[1], B, dtype=np.float32)
    new_dense[bias_layer].set_weights([W_bias, b_bias])

    # Correction layer: compensating bias = -(b @ W_correction)
    W_corr = dense_layers[correction_layer].get_weights()[0]
    b_bias_64 = b_bias.astype(np.float64)
    b_corr = -(b_bias_64 @ W_corr.astype(np.float64)).astype(np.float32)
    new_dense[correction_layer].set_weights([W_corr, b_corr])

    # All other layers: original weights, zero bias
    for i in range(n_dense):
        if i in (bias_layer, correction_layer):
            continue
        Wi = dense_layers[i].get_weights()[0]
        bi = np.zeros(Wi.shape[1], dtype=np.float32)
        new_dense[i].set_weights([Wi, bi])

    new_model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

    return new_model, n_dense


fp_bias = float(os.environ.get("FP_BIAS", "0"))
fp_bias_pos = os.environ.get("FP_BIAS_POS", "begin")  # "begin" or "end"
bias_metadata = None
if fp_bias > 0:
    # The Python certifier needs the biased model's bias vectors to certify it.
    # Refuse to generate biased counter-examples unless we're told where to write
    # them -- silently dropping them produces cexs that cannot be certified.
    bias_output = os.environ.get("BIAS_OUTPUT", "")
    if not bias_output:
        sys.exit("ERROR: FP_BIAS is set but BIAS_OUTPUT is not. Refusing to generate "
                 "biased counter-examples without exporting the bias vectors the "
                 "certifier requires. Set BIAS_OUTPUT=<path/to/biases.txt>.")
    print(f"\n--- Accuracy BEFORE bias (original model) ---")
    loss_before, acc_before = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Accuracy: {acc_before:.4f}, Loss: {loss_before:.4f}")

    model, n_dense = build_biased_model(model, B=fp_bias, position=fp_bias_pos)
    print(f"\nBuilt biased model with B={fp_bias:.0e} "
          f"({n_dense} Dense layers, position={fp_bias_pos})")

    print(f"\n--- Accuracy AFTER bias (biased model) ---")
    loss_after, acc_after = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Accuracy: {acc_after:.4f}, Loss: {loss_after:.4f}")
    print(f"  Accuracy delta: {acc_after - acc_before:+.4f}")

    bias_metadata = {
        "fp_bias": fp_bias,
        "fp_bias_pos": fp_bias_pos,
        "n_dense_layers": n_dense,
        "acc_before_bias": float(acc_before),
        "loss_before_bias": float(loss_before),
        "acc_after_bias": float(acc_after),
        "loss_after_bias": float(loss_after),
        "acc_delta": float(acc_after - acc_before),
    }

    # Export bias vectors for the Python certifier (BIAS_OUTPUT validated above).
    # Extract the actual float32 biases from the model so the certifier
    # gets exactly the same values used during inference.
    dense_layers_new = [l for l in model.layers
                        if isinstance(l, tf.keras.layers.Dense)]
    bias_vecs = []
    for dl in dense_layers_new:
        w = dl.get_weights()
        b = w[1].astype(np.float64)  # upcast for exact decimal repr
        bias_vecs.append(b)
    def _fmt(x):
        return f"{x:.150f}"
    def _vec_bracket(v):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    bias_text = ",".join(_vec_bracket(v) for v in bias_vecs)
    os.makedirs(os.path.dirname(bias_output) or ".", exist_ok=True)
    with open(bias_output, "w") as bf:
        bf.write(bias_text)
    print(f"\nExported bias vectors to {bias_output} "
          f"({os.path.getsize(bias_output)} bytes)")
else:
    print("No bias amplification (set FP_BIAS=1e6 to amplify FP errors)")


# =====================================================================
# The compliant numpy model: same weights as the (possibly biased) Keras
# model, executed via compliant_forward (IEEE-754, per-op rounding, no FTZ).
# =====================================================================
NP_DT = compliant_forward._DT[fmt]

def build_numpy_model(keras_model):
    dense = [l for l in keras_model.layers
             if isinstance(l, tf.keras.layers.Dense)]
    Ws = []
    bs = []
    for l in dense:
        w = l.get_weights()
        # Keras kernel is (in, out); compliant_forward wants (out, in).
        # The transpose+cast is value-preserving: the variables are already
        # stored at the deployment format.
        Ws.append(np.ascontiguousarray(np.asarray(w[0]).T).astype(NP_DT))
        bs.append(np.asarray(w[1]).astype(NP_DT) if len(w) > 1 else None)
    if all(b is None for b in bs):
        bs = None
    elif any(b is None for b in bs):
        raise ValueError("model mixes biased and unbiased Dense layers")
    return (Ws, bs, NP_DT)

NP_MODEL = build_numpy_model(model)
print(f"Built compliant numpy model: "
      f"{len(NP_MODEL[0])} layers, dtype {NP_DT.__name__}, "
      f"biases: {'yes' if NP_MODEL[1] is not None else 'no'}")

def np_logits(x):
    """Logits of the compliant numpy execution for a single input (any shape)."""
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.asarray(compliant_forward.forward_logits(NP_MODEL, flat),
                      dtype=np.float64)

# quick sanity print: how far apart are the two executions on one test point?
_z_keras = model(x_test[:1], training=False).numpy()[0].astype(np.float64)
_z_numpy = np_logits(x_test[0])
print(f"Sanity (test point 0): ||keras - numpy||_2 = "
      f"{np.linalg.norm(_z_keras - _z_numpy):.3e}, "
      f"argmax keras={int(np.argmax(_z_keras))} numpy={int(np.argmax(_z_numpy))}")


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

def check_counter_example(x0, x1, verbose=False):
    """As in the original, but the model is the compliant numpy execution."""
    y0 = np_logits(x0)
    y1 = np_logits(x1)
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

    This runs against the KERAS model: it is only used to locate a boundary
    crossing; the returned point is re-judged under the numpy execution.
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

def extend_ray_to_numpy_flip(x_nat, x_adv, i_star):
    """Fallback for when DeepFool's Keras-adversarial point does not flip the
    numpy execution: the two executions' boundaries can be offset (grossly so
    for the bias-amplified models), so walk further along the ray
    x_nat -> x_adv (t > 1, clipped to the [0,1] box) until the numpy argmax
    leaves i_star. Returns (x_ext, True) on success, (x_adv, False) otherwise."""
    x_nat64 = np.asarray(x_nat, dtype=np.float64)
    x_adv64 = np.asarray(x_adv, dtype=np.float64)
    vec64 = x_adv64 - x_nat64
    for t in (1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        x_ext = np.clip(x_nat64 + t * vec64, 0.0, 1.0).astype(NP_DT).astype(np.float64)
        if int(np.argmax(np_logits(x_ext))) != i_star:
            print(f"  [ray extend] numpy flip found at t={t}")
            return x_ext, True
    return x_adv, False


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

def search_cex(x_nat, x_adv, verbose=False):
    """Bisection along the segment [x_nat, x_adv], judged by the numpy model.

    Mirrors the original: midpoints are computed in float64 and cast to the
    model's format before evaluation (the numpy forward casts its input to
    NP_DT, exactly as Keras casts to the compute dtype).
    """
    x_nat = _ensure_batched(x_nat, model.input_shape)
    x_adv = _ensure_batched(x_adv, model.input_shape)
    y_nat = np_logits(x_nat)
    y_adv = np_logits(x_adv)
    i_star = np.argmax(y_nat)
    j_star = np.argmax(y_adv)

    low  = np.float64(0.0)
    high = np.float64(1.0)
    x_nat64 = np.asarray(x_nat, dtype=np.float64)
    x_adv64 = np.asarray(x_adv, dtype=np.float64)
    vec64   = x_adv64 - x_nat64

    x_low=x_nat64
    x_high=x_adv64

    def to_model_dtype(x64):
        # same role as the Keras compute-dtype cast in the original
        return x64.astype(NP_DT).astype(np.float64)


    cex, max_eps = check_counter_example(x_low, x_high)
    steps=0
    # invariant argmax(y_low) == i_star and argmax(y_high) == j_star
    while high>=low and not cex:
        if verbose and steps%10==0:
            print(f"[search cex] steps {steps}, remaining search width {high-low}")
        mid = np.float64(0.5) * (low + high)
        x_mid64 = x_nat64 + mid * vec64                        # all float64 math here
        x_mid = to_model_dtype(x_mid64)
        y_mid = np_logits(x_mid)
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
        cex, max_eps = check_counter_example(x_low, x_high)

    y_low = np_logits(x_low)
    y_high = np_logits(x_high)
    assert i_star == np.argmax(y_low)
    assert j_star == np.argmax(y_high)

    # probably unnecssary to do this again
    cex, max_eps = check_counter_example(x_low, x_high)

    if verbose:
        print("search_cex returning: ")
        print(f"  y_low  (argmax {i_star}: {y_low}")
        print(f"  y_high (argmax {j_star}: {y_high}")
    return x_low, x_high, cex, max_eps

def try_to_improve_cex_by_extension(x0, x1):
    cex, max_eps = check_counter_example(x0, x1)
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
            cex, max_eps = check_counter_example(x0, x1)
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
    y_final=np_logits(x_final)
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
        if bias_metadata is not None:
            f.write(json.dumps({"_metadata": bias_metadata}, indent=2) + "\n")
        f.flush()
    # num_cexs_written counts only actual cexs and drives the max_cex stop, so the
    # _metadata record never eats a cex slot. A separating comma is written before
    # any element that something already precedes (the metadata record or a prior cex).
    num_cexs_written = 0
    for idx in range(total):
        x_nat0 = x_test[idx]
        y_true = int(np.argmax(y_test[idx]))

        # Ensure shape matches model input (e.g., add channel)
        x_nat = _ensure_batched(x_nat0, model.input_shape)

        # Numpy-model logits & correctness check: the attacked model is the
        # numpy execution, so the correctness gate uses it too.
        y_nat = np_logits(x_nat)
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

        # DeepFool step (against the Keras model)
        x_adv, y_adv_keras, j_star_keras = optimize_to_competitor_deepfool_art(
            classifier,
            x_nat,
            steps=100,          # try 50–150
            overshoot=1e-3,     # 1e-3 to 5e-3 often works well
            verbose=True,
        )

        # Judge the adversarial point under the NUMPY execution: if its argmax
        # hasn't changed there, first try walking further along the DeepFool ray
        # (the numpy boundary can be offset from the Keras one, grossly so for
        # bias-amplified models); only then retry DeepFool with more steps.
        argmax_adv = int(np.argmax(np_logits(x_adv)))
        if argmax_adv == i_star:
            x_adv, success = extend_ray_to_numpy_flip(x_nat, x_adv, i_star)

            # try a few restarts with more steps and increased threshold
            for steps_try in [200, 500, 1000]:
                if success:
                    break
                print(f"Optimising to competitor failed (under numpy). Re-trying with larger steps {steps_try}...")
                x_adv, y_adv_keras, j_star_keras = optimize_to_competitor_deepfool_art(
                    classifier,
                    x_nat,
                    steps=steps_try,
                    overshoot=5e-3,
                    verbose=True,
                )

                if int(np.argmax(np_logits(x_adv))) != i_star:
                    success = True
                    break
                x_adv, success = extend_ray_to_numpy_flip(x_nat, x_adv, i_star)
            if not success:
                # give up on this point
                print(f"Failed to optimise to competitor for index {idx}. Skipping")
                continue

        print("Searching for counter-example (against numpy execution)...")

        # Refine to an (almost) exact tie along the segment
        x0, x1, cex, max_eps = search_cex(x_nat, x_adv, verbose=False)
        if not cex:
            print("Search did not return a counter-example. Checking the other way around...")
            cex, max_eps = check_counter_example(x1, x0)
            if not cex:
                print("Other way around also wasn't a counter-example. Skipping...")
                continue
            print("Other way around is a counter-example. Swapping x0 and x1.")
            # swap x0 and x1
            temp=x1
            x1=x0
            x0=temp


        print(f"Got counter-example with max_eps {max_eps}!")

        # record stage 1 (DeepFool + bisection) result
        stage1_max_eps = str(max_eps)

        # Whitebox extension (extend_cex_multi_ray) not ported to the numpy
        # backend (needs TF gradients; was bypassed in the original anyway).
        stage2_max_eps = stage1_max_eps
        stage2_info = {"eps": 0.0, "slack": 0.0, "class": -1, "skipped": True}

        # blackbox noisy line-search extension
        print(f"Trying to improve counter-example by extension...")
        x1, best_step = try_to_improve_cex_by_extension(x0, x1)
        cex, max_eps = check_counter_example(x0, x1)
        assert cex
        print(f"Got counter-example with max_eps {max_eps}!")
        y0 = np_logits(x0)
        y1 = np_logits(x1)
        argmax_y0 = np.argmax(y0)
        argmax_y1 = np.argmax(y1)

        img_file = save_x_to_image(x1)
        x1_file = save_x_to_file(x1)
        x0_file = save_x_to_file(x0)

        x0_mph = vector_to_mph(x0)
        x1_mph = vector_to_mph(x1)
        dist = l2_norm_mph(x0_mph, x1_mph)
        print(f"Manually confirmed x0 and x1 such that ||x1-x0||=={dist}, F(x0)=={argmax_y0} but F(x1)=={argmax_y1} (F = compliant numpy)")
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
            "model_execution": "numpy_ieee754_compliant",
            "stages": {
                "deepfool_bisect": {
                    "max_eps": stage1_max_eps,
                },
                "whitebox_extend": {
                    "max_eps": stage2_max_eps,
                    "multi_ray_eps": stage2_info["eps"],
                    "multi_ray_slack": stage2_info["slack"],
                    "multi_ray_class": stage2_info["class"],
                },
                "blackbox_extend": {
                    "max_eps": str(max_eps),
                    "best_step": int(best_step),
                },
            },
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
            if bias_metadata is not None or num_cexs_written > 0:
                f.write(",\n")
            f.write(json.dumps(summary, indent=2, cls=NumpyEncoder) + "\n")
            f.flush()
            num_cexs_written += 1

        if max_cex is not None and num_cexs_written >= max_cex:
            print(f"\nReached max_cex={max_cex} counter-examples. Stopping.")
            break


def handle_interrupt(sig, frame):
    print("Caught CTRL-C.")
    # close out the log
    with open(log_file, "a", buffering=1) as f:  # line-buffered mode
        f.write("\n]\n")
        f.flush()
    sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    main()
    # close out the JSON array on normal exit
    with open(log_file, "a", buffering=1) as f:
        f.write("\n]\n")
        f.flush()
