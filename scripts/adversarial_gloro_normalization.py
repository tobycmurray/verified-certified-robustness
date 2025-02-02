import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import sys
import os

from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate

# Define model
model = keras.Sequential([
    Dense(1, input_shape=(1,), activation=None, use_bias=False),  # 1 neuron, no bias, no activation    
    Dense(2, activation=None, use_bias=False)  # 2 neurons, no bias, no activation
])

pos_float = np.finfo(np.float32).tiny
neg_float = -np.finfo(np.float32).tiny

def fmt_symbolic(num):
    if num==np.finfo(np.float32).tiny:
        return "tiny"
    elif num==-np.finfo(np.float32).tiny:
        return "-tiny"
    elif num==0.0:
        return "0.0"
    else:
        return f"{num:.150f}"

def fmt_array(array):
    return np.array2string(array, separator=",", formatter={'float_kind': fmt_symbolic})

#LAYER_1_CONSTANT=0.000001
LAYER_1_CONSTANT=0.00001
LAYER_2_WEIGHTS=[1.0, -1.0]
ACTUAL_LIPSCHITZ_CONSTANT=abs(LAYER_1_CONSTANT)*abs(LAYER_2_WEIGHTS[0]-LAYER_2_WEIGHTS[1])


print("")
print(f"The actual Lipschitz constant for this network is: {ACTUAL_LIPSCHITZ_CONSTANT:.10f}")

print("")

# Manually set weights
# 10000000000.0 causes power iteration to return the wrong value due to imprecision / overflow
# let's try something small in the hope that the accuracy threshold is reached first before power iteration finishes
custom_weights1 = np.array([[LAYER_1_CONSTANT]], dtype=np.float32)  # Shape (1,1)
custom_weights2 = np.array([LAYER_2_WEIGHTS], dtype=np.float32)  # Shape (1,2)
print("Initialising model with custom weights...")
print(f"Custom weights layer 1: {fmt_array(custom_weights1)}")
print(f"Custom weights layer 2: {fmt_array(custom_weights2)}")

model.layers[0].set_weights([custom_weights1])
model.layers[1].set_weights([custom_weights2])



# Define training data
X_train = np.array([[pos_float], [neg_float]], dtype=np.float32)
Y_train = np.array([0, 1], dtype=np.int32)
    
epsilon=1.0

print("")
print(f"Building gloro model with epsilon {epsilon}...")
g = GloroNet(model=model, epsilon=epsilon)

print("Compiling model...")
g.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

initial_weights1 = g.layers[1].get_weights()[0]
initial_weights2 = g.layers[2].get_weights()[0]

print("Training gloro model with this training data: ")
i=0
while i<len(X_train):
    print(f"X: {fmt_symbolic(X_train[i])} -> Y: {Y_train[i]}")
    i+=1

EPOCHS=10
print(f"Training gloro model for {EPOCHS} epochs...")

# training produces lots of text so turn that off
#saved_stdout = sys.stdout
#devnull = open(os.devnull, 'w')
#sys.stdout = devnull
g.fit(X_train, Y_train, epochs=EPOCHS)
#sys.stdout = saved_stdout
#devnull.close()

trained_weights1=g.layers[1].get_weights()[0]
trained_weights2=g.layers[2].get_weights()[0]

layer_weights = [layer.get_weights()[0] for layer in g.layers if len(layer.get_weights()) > 0]

if not np.array_equal(initial_weights1,trained_weights1) or not np.array_equal(initial_weights2,trained_weights2):
    print("Training modified the weights!")
    print([initial_weights1,initial_weights2])
    print(layer_weights)
    sys.exit(1)
else:
    print("Training did not modify weights, as expected.")

print("")

save_dir="adversarial_gloro_normalization-results/"

lipschitz_constants = g.lipschitz_constant()


sub_lipschitz = g.sub_lipschitz


print("Gloro model lipschitz constants: ")
print(lipschitz_constants)

print("Gloro model sub lipschitz constants (for all layers but the final one): ")
print(sub_lipschitz)

print("The safe value for the sub lipschitz constant is: ", abs(LAYER_1_CONSTANT))

if sub_lipschitz < LAYER_1_CONSTANT:
    print("Gloro computed unsafe Lipschitz bounds")
else:
    print("Gloro bounds might be safe.")
    sys.exit(1)

print("")

print(f"Saving (trained) model weights and Lipschitz constants to {save_dir} ...")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through each layer, extract weights and biases, and save them with very high precision
for i, weights in enumerate(layer_weights):
    np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%.150f')

# Loop through each layer, extract weights and biases, and save them with very high precision
for i, c in enumerate(lipschitz_constants):
    np.savetxt(os.path.join(save_dir, f'logit_{i}_gloro_lipschitz.csv'), [c], delimiter=',', fmt='%.150f')
        

print("Building a fresh dense (non-gloro) model from the trained gloro weights...")

# now build a new model with the trained weights to get some outputs to run the certifier on
model2 = keras.Sequential([
    Dense(1, input_shape=(1,), activation=None, use_bias=False),  # 2 neurons, no bias, no activation    
    Dense(2, input_shape=(1,), activation=None, use_bias=False)  # 2 neurons, no bias, no activation
])
model2.layers[0].set_weights([trained_weights1])
model2.layers[1].set_weights([trained_weights2])

# Test with positive and negative input
test_inputs = np.array([[0], [1.0], [-1.0], [0.2], [-0.2], [1.2], [-1.2], [100000000000000000000000000000000000]], dtype=np.float32)

print("")
print("Running the gloro model on the test inputs...")

gloro_outputs = g.predict(test_inputs)
print("Gloro outputs for test inputs:")
i = 0
while i<len(test_inputs):
    print(f"Input:   {test_inputs[i]}")
    print(f"Output:  {gloro_outputs[i]}")
    i+=1

print(f"The gloro model certified this percentage of the otuputs as robust at epsilon {epsilon}: {(1.0 - rejection_rate(gloro_outputs,gloro_outputs))*100}%")

print("")
print("Running the fresh model on the test inputs...")

outputs = model2.predict(test_inputs)

inputs_outputs = {}

print("Outputs for test inputs (to run the certifier on):")
i = 0
while i<len(test_inputs):
    print(f"Input:   {test_inputs[i]}")
    print(f"Output:  {fmt_array(outputs[i])}")
    i+=1
