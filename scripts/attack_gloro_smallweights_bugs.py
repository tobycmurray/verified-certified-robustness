# This code is used to attack two different bugs in the gloro implementation:
# 1. floating point imprecision leading to falsely calculating Lipschitz bounds of 0 for models with tiny weights
# 2. The normalization bug https://github.com/klasleino/gloro/issues/8
#
# Ordinarily the code will attack bug 2 (because that is triggered first)
# But with tha tbug fixed it then attacks bug 1 (floating point imprecision)
#
# What this code does is take a working (gloro) model and scale its weights to ensure that one layer (the second-to-last) has very small weights
# while the final layer has a corresponding large weights (thereby trying to ensure that the overall Lipschitz constants stay about the same as
# in the original model)
#
# The model can then be gloro trained without materially changing the Lipschitz constants (we show this for 200 training epochs)
# When attacking bug 1 the final model then still incorrectly reports a Lipschitz constant of 0 and certifies 100% of the test points
# When attacking but 2 the final mdoel under-estimates its Lipschitz constants, certifying (for an MNIST model) more than 99.87% of test points
# (when the true robustness was originally 95.8% over the test set)

import doitlib
import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import os

import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer

if len(sys.argv) != 7:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES model_weights_csv_dir output_file input_size eval_epsilon\n");
    sys.exit(1)

dataset=sys.argv[1]
    
INTERNAL_LAYER_SIZES=eval(sys.argv[2])

csv_loc=sys.argv[3]+"/"

output_file=sys.argv[4]

input_size=int(sys.argv[5])

epsilon=float(sys.argv[6])

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")
print(f"Running with input_size: {input_size}")


def mprint(string):
    print(string, end="")


def printlist(floatlist):
        count=0
        n=len(floatlist)
        for num in floatlist:
            mprint(f"{num:.5f}")
            count=count+1
            if count<n:
                mprint(",")

    
inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size, dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)

print("Building zero-bias gloro model from saved weights...")

doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)



# evaluate hte resulting model
print("Evaluating the resulting zero-bias gloro model...")

x_test, y_test = doitlib.load_test_data(dataset=dataset,input_size=input_size)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

orig_accuracy=accuracy

print(f"Test Loss (zero bias model): {loss:.4f}")
print(f"Test Accuracy (zero bias model): {accuracy:.4f}")

scaling_factor=2.0
second_final_layer = model.layers[-2]
second_final_weights = np.array(second_final_layer.get_weights()[0])
second_final_weights_saved=second_final_weights.copy()
final_layer = model.layers[-1]
final_weights = np.array(final_layer.get_weights()[0])
final_weights_saved=final_weights.copy()
i=0

from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate

g_orig = GloroNet(model=model, epsilon=epsilon)

g_orig.freeze_lc()

lipschitz_constants_orig = g_orig.lipschitz_constant()
#print(f"Lipschitz constants: {lipschitz_constants_orig}")

lipschitz_constants = g_orig.lipschitz_constant()
mean_orig=np.mean(lipschitz_constants_orig[lipschitz_constants_orig != -1])
mean=np.mean(lipschitz_constants[lipschitz_constants != -1])
print(f"Lipschitz mean: {mean}")

# we stop when the mean lipschitz bound is 20% lower than the original or more
mean_tolerance=0.2
mean_lower=(1.0-mean_tolerance)*mean_orig
mean_upper=(1.0+mean_tolerance)*mean_orig

gloro_outputs=g_orig.predict(x_test)
rejection_rate_orig=rejection_rate(gloro_outputs,gloro_outputs)

# continually scale weights in final layer and measure accuracy
while accuracy == orig_accuracy and mean > mean_lower and mean <= mean_upper:    
    final_weights_saved=final_weights.copy()
    final_weights *= scaling_factor
    second_final_weights_saved=second_final_weights.copy()
    second_final_weights /= scaling_factor
    i=i+1
    second_final_layer.set_weights([second_final_weights])
    final_layer.set_weights([final_weights])    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("")
    print(f"After doing {i+1} scalings: ")
    print(f"Test Loss (zero bias model): {loss:.4f}")
    print(f"Test Accuracy (zero bias model): {accuracy:.4f}")
    g = GloroNet(model=model, epsilon=epsilon)
    g.freeze_lc()
    lipschitz_constants = g.lipschitz_constant()
    mean=np.mean(lipschitz_constants[lipschitz_constants != -1])    
    #print(f"Lipschitz constants: {lipschitz_constants}")
    print(f"Lipschitz mean: {mean}")
    #gloro_outputs=g.predict(x_test)
    #rr=rejection_rate(gloro_outputs,gloro_outputs)
    #print(f"Rejection rate: {rr}")
    
#second_final_layer.set_weights([second_final_weights_saved])
#final_layer.set_weights([final_weights_saved])
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
assert accuracy==orig_accuracy



g = GloroNet(model=model, epsilon=epsilon)
g.freeze_lc()
lipschitz_constants = g.lipschitz_constant()
print(f"Lipschitz constants: {lipschitz_constants}")
mean=np.mean(lipschitz_constants[lipschitz_constants != -1])    
#print(f"Lipschitz constants: {lipschitz_constants}")
print(f"Lipschitz mean: {mean}")

gloro_outputs=g.predict(x_test)
rejection_rate_new=rejection_rate(gloro_outputs,gloro_outputs)

print(f"Original rejection rate: {rejection_rate_orig}. Final rejection rate: {rejection_rate_new}")

# now train a model to show that the attack is robust even under training

g = GloroNet(model=model, epsilon=epsilon)

train, test = doitlib.load_gloro_data(batch_size=256,input_size=input_size, dataset=dataset)

from gloro.training import losses
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import LrScheduler
from utils import get_optimizer

g.compile(
    loss=losses.get('sparse_crossentropy'),
    optimizer=get_optimizer('adam', 0.001),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # metrics=[rejection_rate]
)

epochs=200

print('training model...')
g.fit(
    train,
    epochs=epochs,
    validation_data=test,
    callbacks=[
        EpsilonScheduler('fixed'),
        LrScheduler('fixed'),
    ],
)

lipschitz_constants = g.lipschitz_constant()
print(f"Lipschitz constants: {lipschitz_constants}")
mean=np.mean(lipschitz_constants[lipschitz_constants != -1])    
#print(f"Lipschitz constants: {lipschitz_constants}")
print(f"Lipschitz mean: {mean}")

gloro_outputs=g.predict(x_test)
rejection_rate_new=rejection_rate(gloro_outputs,gloro_outputs)

print(f"Original rejection rate: {rejection_rate_orig}. Final rejection rate: {rejection_rate_new}")
