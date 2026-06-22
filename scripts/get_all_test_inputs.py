import json
import sys
import doitlib
import numpy as np
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer
from tensorflow.keras import mixed_precision
import tensorflow as tf

def save_x_to_file(x, output_file=None):
    if output_file is None:
        # Create a unique temporary file in the current directory
        with tempfile.NamedTemporaryFile(prefix="x_", suffix=".npy", dir=".", delete=False) as f:
            output_file = f.name
    np.save(output_file, x)
    return output_file
    
if len(sys.argv) not in (8, 9):
    print(f"Usage {sys.argv[0]} float_format dataset INTERNAL_LAYER_SIZES model_weights_csv_dir input_size output_json_file epsilon [max_inputs]\n")
    sys.exit(1)

fmt = sys.argv[1]

try:
    mixed_precision.set_global_policy(fmt)
except Exception as e:
    print("Failed to set precision", str(e))
    sys.exit(1)

print(f"Running models with precision: {mixed_precision.global_policy()}")

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
    y = model(x, training=False).numpy()[0]
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
    
