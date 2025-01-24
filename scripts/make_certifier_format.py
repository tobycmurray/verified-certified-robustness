import numpy as np

def mprint(string):
    print(string, end="")

import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

weights = []
i=0
# always one extra iteration than INTERNAL_LAYER_SIZES length
while i<=len(INTERNAL_LAYER_SIZES):
    weights.append(np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=",").tolist())
    i=i+1

m = 0

weightslen=len(weights)
w=0
for mat in weights:
    #print(f"Matrix {m} has dimensions: {len(mat)} x {len(mat[0])}")
    m = m + 1

    mprint("[")
    matlen=len(mat)
    r=0
    for row in mat:
        mprint("[")
        count=0
        n=len(row)
        for num in row:
            mprint(f"{num:.5f}")
            count=count+1
            if count<n:
                mprint(",")
        mprint("]")
        r=r+1
        if r<matlen:
            mprint(",")
    mprint("]")
    w=w+1
    if w<weightslen:
        mprint(",")

