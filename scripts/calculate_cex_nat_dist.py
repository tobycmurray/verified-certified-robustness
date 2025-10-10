import json
import sys
import doitlib
import numpy as np
from PIL import Image
from mpmath import mp, mpf, sqrt, nstr, floor

mp.dps = 60  # massive precision

def vector_to_mph(v):
    return list(map(mpf, np.asarray(v).flatten().tolist()))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

if len(sys.argv) != 4:
    print(f"Usage {sys.argv[0]} dataset input_size cex_json_file\n")
    sys.exit(1)

dataset = sys.argv[1]
input_size = int(sys.argv[2])
cex_json = sys.argv[3]

x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)
with open(cex_json, 'r') as f:
    data = json.load(f)

new_data = []
for cex in data:
    idx = cex["index"]
    x_nat = x_test[idx]
    x_b_file = cex["x1_file"]
    x_b = np.load(x_b_file)
    x_nat_mph = vector_to_mph(x_nat)
    x_b_mph = vector_to_mph(x_b)
    dist = l2_norm_mph(x_nat_mph, x_b_mph)
    cex["nat_x1_dist"] = str(dist)
    new_data.append(cex)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

print(json.dumps(new_data, indent=2, cls=NumpyEncoder))

