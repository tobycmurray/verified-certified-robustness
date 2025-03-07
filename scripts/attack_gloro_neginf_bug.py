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
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES model_weights_csv_dir output_adversarial_image_file input_size eval_epsilon\n");
    sys.exit(1)

dataset=sys.argv[1]
    
INTERNAL_LAYER_SIZES=eval(sys.argv[2])

csv_loc=sys.argv[3]+"/"

output_file=sys.argv[4]

input_size=int(sys.argv[5])

epsilon=float(sys.argv[6])

print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")
print(f"Running with input_size: {input_size}")
    
inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size, dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)

print("Building zero-bias gloro model from saved weights...")

doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

# Initialize input variable (randomly or meaningfully)
x = tf.Variable(tf.random.normal([1] + list(model.input_shape[1:])))

initial_learning_rate=0.1
final_learning_rate=5e-7
max_steps=10000

# Solve for decay_rate using the equation: final_lr = initial_lr * decay_rate^(max_steps)
decay_rate = np.exp(np.log(final_learning_rate / initial_learning_rate) / max_steps)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1,  # Apply decay at each step
    decay_rate=decay_rate,
    staircase=False  # Smooth decay
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# we are searching for inputs that map to the 0 vector output
target = tf.constant([0.0] * 10, dtype=tf.float32)

# Optimization loop
for step in range(max_steps):  # Number of optimization steps
    with tf.GradientTape() as tape:
        y_pred = model(x, training=False)  # Forward pass

        # we compute the loss as the mean absolute value of difference to the target
        z = y_pred - target
        loss = tf.reduce_mean(tf.abs(z))

    # Compute gradients of loss w.r.t input x
    grads = tape.gradient(loss, x)

    # Apply gradients (update x)
    optimizer.apply_gradients([(grads, x)])

    # Optional: Print loss for monitoring
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.numpy()}")

    if loss.numpy() == 0.0:
        print(f"Converged at step {step}, Loss: {loss.numpy()}")
        break

# save a copy
saved_x = x

x_final = x.numpy()

y_final = model(x, training=False).numpy()

import numpy as np

max_logit = np.max(y_final)

count_max_logits = np.sum(y_final == max_logit)

# Print with high precision
np.set_printoptions(precision=80, suppress=False)
print("Final output: ", y_final)
print("max_logit: ", max_logit)
print(f"There are {count_max_logits} that have this same value")



# now test the gloro model
from gloro.models import GloroNet


g = GloroNet(model=model, epsilon=epsilon)

g.freeze_lc()

gloro_y_pred = g.predict(saved_x)

print("gloro_y_pred: ", gloro_y_pred)
bot_index = tf.cast(tf.shape(gloro_y_pred)[1] - 1, 'int64')
preds = tf.argmax(gloro_y_pred, axis=1)
if preds[0] == bot_index:
    print("Gloro says this output is NOT robust. Exiting.")
    sys.exit(1)
else:
    print(f"Gloro says this output is robust at epsilon {epsilon}!")
    
# save the found input as an image

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

image.save(output_file)
print(f"Adversarial image that Gloro says is robust saved to: {output_file}")
