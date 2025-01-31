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
    Dense(2, input_shape=(1,), activation=None, use_bias=False)  # 2 neurons, no bias, no activation
])

#pos_float = np.finfo(np.float32).max
#neg_float = np.finfo(np.float32).min
pos_float = np.finfo(np.float32).tiny
neg_float = -np.finfo(np.float32).tiny

# Manually set weights
custom_weights = np.array([[pos_float, neg_float]], dtype=np.float32)  # Shape (1,2)
saved_custom_weights = custom_weights
model.layers[0].set_weights([custom_weights])

# Verify weights
print("Weights after initialization:")
print(model.layers[0].get_weights())

# Define training data
X_train = np.array([[1.0], [-1.0]], dtype=np.float32)
Y_train = np.array([0, 1], dtype=np.int32)
#    [max_float, min_float],  # Expected output for input 1.0
#    [1.0, 0.0],  # Expected output for input 1.0    
#    [min_float, max_float]   # Expected output for input -1.0
#    [0.0, 1.0],  # Expected output for input 1.0    
#], dtype=np.float32)


    
epsilon=1.0

g = GloroNet(model=model, epsilon=epsilon)

g.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-10), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

g.fit(X_train, Y_train, epochs=0, verbose=1)

print("Weights after training:")
print(g.layers[1].get_weights())

trained_weights=g.layers[1].get_weights()
manhattan_distance = np.sum(np.abs(trained_weights - saved_custom_weights))
print(f"Manhattan distance between new and old weights is: {manhattan_distance}")

save_dir="adversarial_gloro-results/"
layer_weights = [layer.get_weights()[0] for layer in g.layers if len(layer.get_weights()) > 0]

lipschitz_constants = g.lipschitz_constant()

print("Gloro model lipschitz constants: ")
print(lipschitz_constants)

for i, weights in enumerate(layer_weights):
    np.savez(f"layer_{i}_weights.npz", weights=weights)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop through each layer, extract weights and biases, and save them
for i, weights in enumerate(layer_weights):
    np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%.80f')

# Loop through each layer, extract weights and biases, and save them
for i, c in enumerate(lipschitz_constants):
    np.savetxt(os.path.join(save_dir, f'logit_{i}_gloro_lipschitz.csv'), [c], delimiter=',', fmt='%f')
        
print('model weights extracted.')


# now build a new model with the trained weights to get some outputs to run the certifier on
model2 = keras.Sequential([
    Dense(2, input_shape=(1,), activation=None, use_bias=False)  # 2 neurons, no bias, no activation
])
model2.layers[0].set_weights([custom_weights])

# Test with positive and negative input
test_inputs = np.array([[1.0], [-1.0], [0.2], [-0.2]], dtype=np.float32)

gloro_outputs = g.predict(test_inputs)
print("Gloro outputs for test inputs:")
i = 0
while i<len(test_inputs):
    print(f"Input:   {test_inputs[i]}")
    print(f"Output:  {gloro_outputs[i]}")
    i+=1

print(f"The gloro model certified this proportion of the otuputs as robust at epsilon {epsilon}: {1.0 - rejection_rate(gloro_outputs,gloro_outputs)}")

outputs = model2.predict(test_inputs)

print("Outputs for test inputs (to run the certifier on):")
i = 0
while i<len(test_inputs):
    print(f"Input:   {test_inputs[i]}")
    print(f"Output:  {np.array2string(outputs[i], formatter={'float_kind': lambda x: f'{x:.80f}'})}")
    i+=1
