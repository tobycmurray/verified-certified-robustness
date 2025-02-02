import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import sys
import os
import math

from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate

# Define model
model = keras.Sequential([
    Dense(1, input_shape=(1,), activation=None, use_bias=False),  # 1 neuron, no bias, no activation    
    Dense(2, activation=None, use_bias=False)  # 2 neurons, no bias, no activation
])

tiny = np.finfo(np.float32).tiny

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

WEIGHTS=[[0.00001],[1.0, -1.0]]

# lipschitz constants of each layer -- the value for the second layer is the margin Lipschitz constants for the two logits
LIPSCHITZ_CONSTANTS=[abs(WEIGHTS[0][0]),abs(WEIGHTS[1][0]-WEIGHTS[1][1])]
LIPSCHITZ_CONSTANT=math.prod(LIPSCHITZ_CONSTANTS)

NUM_LAYERS=len(WEIGHTS)

initial_weights=[np.array([w], dtype=np.float32) for w in WEIGHTS]

print("")
print(f"The actual Lipschitz constant for this network is: {LIPSCHITZ_CONSTANT}")

print("")


print("Initialising model with custom weights...")
for i in range(len(initial_weights)):
    print(f"Layer {i}: {initial_weights[i]}")
    model.layers[i].set_weights([initial_weights[i]])



# Define training data to ensure that model training won't update the weights
# alternatively, we could train for 0 epochs
X_train = np.array([[tiny], [-tiny]], dtype=np.float32)
Y_train = np.array([0, 1], dtype=np.int32)
    
epsilon=1.0

print("")
print(f"Building gloro model with epsilon {epsilon}...")
g = GloroNet(model=model, epsilon=epsilon)

print("Compiling model...")
g.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))


print("Training gloro model with this training data: ")
for i in range(len(X_train)):
    print(f"X: {fmt_symbolic(X_train[i])} -> Y: {Y_train[i]}")

EPOCHS=10
print(f"Training gloro model for {EPOCHS} epochs...")

# training produces lots of text so turn that off
#saved_stdout = sys.stdout
#devnull = open(os.devnull, 'w')
#sys.stdout = devnull
g.fit(X_train, Y_train, epochs=EPOCHS)
#sys.stdout = saved_stdout
#devnull.close()

trained_weights = [layer.get_weights()[0] for layer in g.layers if len(layer.get_weights()) > 0]
assert len(trained_weights) == len(initial_weights)

for i in range(len(initial_weights)):
    if not np.array_equal(initial_weights[i],trained_weights[i]):
        print(f"Training modified the weights in layer {i}!")
        print("     Original weights: ")
        print(initial_weights[i])
        print("     New weights: ")
        print(trained_weights[i])
        sys.exit(1)
        
print("Training did not modify weights, as expected.")
print("")

save_dir="adversarial_gloro_normalization-results/"

lipschitz_constants = g.lipschitz_constant()

sub_lipschitz = g.sub_lipschitz

print("Gloro model lipschitz constants: ")
print(lipschitz_constants)

print("Gloro model sub lipschitz constants (for all layers but the final one): ")
print(sub_lipschitz)

print("The safe value for the sub lipschitz constant is: ", LIPSCHITZ_CONSTANTS[0])

if sub_lipschitz < LIPSCHITZ_CONSTANTS[0]:
    print("Gloro computed unsafe Lipschitz bounds")
else:
    print("Gloro bounds might be safe.")
    sys.exit(1)

print("")

print(f"Saving (trained) model weights and Lipschitz constants to {save_dir} ...")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through each layer, extract weights and biases, and save them with very high precision
for i, weights in enumerate(trained_weights):
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

for i in range(len(trained_weights)):
    model2.layers[i].set_weights([trained_weights[i]])

# Test with positive and negative input
test_inputs = np.array([[0], [1.0], [-1.0], [0.2], [-0.2], [1.2], [-1.2], [100000000000000000000000000000000000]], dtype=np.float32)

print("")
print("Running the gloro model on the test inputs...")

gloro_outputs = g.predict(test_inputs)
bot_index = tf.cast(tf.shape(gloro_outputs)[1] - 1, 'int64')
preds = tf.argmax(gloro_outputs, axis=1)

print("Gloro outputs for test inputs:")
i = 0
while i<len(test_inputs):
    print(f"Input:   {test_inputs[i]}")
    print(f"Output:  {gloro_outputs[i]}")
    if preds[i] == bot_index:
        print(f"Gloro says this output was NOT robust")
    else:
        print(f"Gloro says this output was robust")
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
