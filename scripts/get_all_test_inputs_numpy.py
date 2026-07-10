"""Generate the certifier's "all test inputs" JSON with y1 logits computed by the
IEEE-754-compliant numpy execution (python-certifier's compliant_forward.py)
instead of the Keras/TensorFlow execution (which runs with FTZ on this machine).

Copy of get_all_test_inputs.py; the only semantic change is how y1 is computed:
the model weights are still loaded into a Keras model via doitlib (same CSV code
path as always), but the forward pass producing y1 runs in compliant_forward
(per-op round-to-nearest, gradual underflow, no BLAS, no FTZ) on the exact same
weight tensors, transposed to the certifier's (out,in) orientation.

Output schema is unchanged: [{index, max_eps, x1_file, y1, label}, ...] with
orig_<dataset>_x_<i>.npy files alongside, so run_tests.sh / robust_certifier.py
consume it as-is.

Differences from the original generator:
  - If an orig_*.npy already exists (e.g. writing a *_numpy.json variant into a
    directory provisioned by the Keras generator), it is NOT rewritten: it is
    verified equal to the freshly-loaded input and reused, so the two JSON
    variants provably share the same x files. A mismatch aborts.
  - Optional env var BIASES_FILE=<biases.txt> (certifier format): y1 is computed
    by the BIASED numpy model (natural weights + those bias vectors), for the
    adversarially-biased models' "all" runs. The Keras sanity comparison is
    skipped in that case (the Keras model built here is the natural one).

Usage (same CLI as the original):
  python get_all_test_inputs_numpy.py float_format dataset INTERNAL_LAYER_SIZES \
      model_weights_csv_dir input_size output_json_file epsilon [max_inputs]
"""
import json
import os
import sys
import doitlib
import numpy as np
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer
from tensorflow.keras import mixed_precision
import tensorflow as tf

# the IEEE-754-compliant numpy forward pass lives in python-certifier
PYTHON_CERTIFIER = os.environ.get(
    "PYTHON_CERTIFIER", os.path.expanduser("~/git/python-certifier"))
sys.path.insert(0, PYTHON_CERTIFIER)
import compliant_forward

def save_x_to_file(x, output_file=None):
    if output_file is None:
        # Create a unique temporary file in the current directory
        with tempfile.NamedTemporaryFile(prefix="x_", suffix=".npy", dir=".", delete=False) as f:
            output_file = f.name
    if os.path.exists(output_file):
        # A previous provisioning (e.g. the Keras generator) already wrote this
        # input: verify it is the same data and reuse it, so the _numpy.json
        # variant provably shares the existing x files. Never overwrite.
        existing = np.load(output_file)
        if not np.array_equal(existing, np.asarray(x, dtype=existing.dtype)):
            sys.exit(f"ERROR: {output_file} exists with DIFFERENT content; refusing "
                     f"to overwrite (test-data pipeline mismatch?)")
        return output_file
    np.save(output_file, x)
    return output_file

if len(sys.argv) not in (8, 9):
    print(f"Usage {sys.argv[0]} float_format dataset INTERNAL_LAYER_SIZES model_weights_csv_dir input_size output_json_file epsilon [max_inputs]\n")
    sys.exit(1)

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

print(f"Keras policy (weight loading only): {mixed_precision.global_policy()}")
print(f"y1 logits computed by compliant numpy execution at: {fmt}")

dataset = sys.argv[2]
INTERNAL_LAYER_SIZES = eval(sys.argv[3])
csv_loc = sys.argv[4] + "/"
input_size = int(sys.argv[5])
output_json = sys.argv[6]
epsilon = sys.argv[7]
max_inputs = int(sys.argv[8]) if len(sys.argv) == 9 else None
inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size,
                                      dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)
doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

# Build the compliant numpy model from the same weight tensors the Keras model
# holds (Keras kernel is (in,out); compliant_forward wants (out,in)).
NP_DT = compliant_forward._DT[fmt]
_dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
_Ws = [np.ascontiguousarray(np.asarray(l.get_weights()[0]).T).astype(NP_DT) for l in _dense]

# Optional adversarial-bias vectors (certifier biases.txt format): y1 becomes the
# BIASED model's compliant numpy execution (weights are unchanged by the bias
# construction, so _Ws stays the natural kernels).
BIASES_FILE = os.environ.get("BIASES_FILE", "")
_bs = None
if BIASES_FILE:
    from parsing import load_biases_from_file
    _biases = load_biases_from_file(BIASES_FILE)
    if len(_biases) != len(_Ws):
        sys.exit(f"ERROR: {BIASES_FILE} has {len(_biases)} bias vectors but the "
                 f"model has {len(_Ws)} layers")
    _bs = [np.array([float(v) for v in b], dtype=NP_DT) for b in _biases]
    print(f"Loaded adversarial biases from {BIASES_FILE}")
NP_MODEL = (_Ws, _bs, NP_DT)
print(f"Built compliant numpy model: {len(_Ws)} layers, dtype {NP_DT.__name__}, "
      f"biases: {'yes' if _bs is not None else 'no'}")

def np_logits(x):
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.asarray(compliant_forward.forward_logits(NP_MODEL, flat),
                      dtype=np.float64)

x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)
tabular = doitlib.datasets[dataset].get("tabular", False)
res = []
for i, x in enumerate(x_test):
    if max_inputs is not None and i >= max_inputs:
        break
    if tabular:
        x = np.expand_dims(x, axis=0)        # (D,) -> (1, D)
    else:
        x = np.expand_dims(x, axis=(0, -1))  # image: (H,W,C) -> (1,H,W,C,1)
    y = np_logits(x)
    if i == 0 and _bs is None:
        # one-off sanity print: how far apart are the two executions here?
        # (skipped for biased runs -- the Keras model here is the natural one)
        y_keras = model(x, training=False).numpy()[0].astype(np.float64)
        print(f"Sanity (input 0): ||keras - numpy||_2 = "
              f"{np.linalg.norm(y_keras - y):.3e}, "
              f"argmax keras={int(np.argmax(y_keras))} numpy={int(np.argmax(y))}")
    if i % 100 == 0:
        print(f"Saved {i} inputs so far...")
    output_file=f"orig_{dataset.replace('/', '_')}_x_{i}.npy"  # sanitize tfds config slash (emnist/balanced)
    x1_file = save_x_to_file(x, output_file)
    summary = {}
    summary["index"] = i
    summary["max_eps"] = epsilon
    summary["x1_file"] = x1_file
    summary["y1"] = y
    summary["label"] = int(np.argmax(y_test[i]))  # true class, for VRA (cert n correct)
    res.append(summary)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(output_json, 'w') as f:
    f.write(json.dumps(res, cls=NumpyEncoder, indent=2))
