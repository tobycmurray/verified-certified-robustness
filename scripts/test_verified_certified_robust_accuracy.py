import doitlib
import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import json
import os

import sys

if len(sys.argv) != 6:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES certifier_results.json model_weights_csv_dir input_size\n");
    sys.exit(1)

dataset=sys.argv[1]

INTERNAL_LAYER_SIZES=eval(sys.argv[2])

json_results_file=sys.argv[3]

csv_loc=sys.argv[4]+"/"

input_size=int(sys.argv[5])

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")
print(f"Running with input size: {input_size}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

inputs, outputs = doitlib.build_mnist_model(Input, Flatten, Dense, input_size=input_size, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)

print("Building zero-bias gloro model from saved weights...")

doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

# evaluate hte resulting model
print("Evaluating the resulting zero-bias gloro model...")


x_test, y_test = doitlib.load_test_data(dataset=dataset,input_size=input_size)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)


print(f"Test Loss (zero bias model): {loss:.4f}")
print(f"Test Accuracy (zero bias model): {accuracy:.4f}")


print("Generating output vectors from the test (evaluation) data...")

outputs = model.predict(x_test)

predicted_classes = np.argmax(outputs, axis=1)
true_classes = np.argmax(y_test, axis=1)

correct_classifications = predicted_classes == true_classes

robustness=[]
with open(json_results_file, 'r') as f:
    robustness = json.load(f)

print("Evaluating Verified Certified Robust Accuracy...\n")
i=0 # robustness index
j=0 # correct_classifications index
count_robust_and_correct=0
count_robust=0
count_correct=0
# the first item in this list is the Lipschitz bounds; others may be debug messages etc.
assert len(robustness) >= len(correct_classifications)+1
robustness=robustness[1:]
assert len(robustness) >= len(correct_classifications)
n=len(robustness)
while i<n:
    r = robustness[i]
    if "certified" in r:
        robust = r["certified"]
        correct = correct_classifications[j]
        if robust and correct:
            count_robust_and_correct=count_robust_and_correct+1
        if robust:
            count_robust=count_robust+1
        if correct:
            count_correct=count_correct+1
        if i%1000==0:
            print(f"...done {i} of {n} evaluations...\n");
        j=j+1
    i=i+1

assert j==10000
assert i>=10000

print(f"Proportion robust: {float(count_robust)/float(10000)}")
print(f"Proportion correct: {float(count_correct)/float(10000)}")
print(f"Proportion robust and correct: {float(count_robust_and_correct)/float(10000)}")

