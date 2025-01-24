import doitlib
import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import os

import sys

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer

# currently unused
class MinMax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._flat_op = Flatten()

    def call(self, x):
        x_flat = self._flat_op(x)
        x_shape = tf.shape(x_flat)

        grouped_x = tf.reshape(
            x_flat,
            tf.concat([x_shape[:-1], (-1, 2)], -1))

        min_x = tf.reduce_min(grouped_x, axis=-1, keepdims=True)
        max_x = tf.reduce_max(grouped_x, axis=-1, keepdims=True)

        sorted_x = tf.reshape(
            tf.concat([min_x, max_x], axis=-1),
            tf.shape(x))

        return sorted_x

    def lipschitz(self):
        return 1.

if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir output_file input_size\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

output_file=sys.argv[3]

input_size=int(sys.argv[4])

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

inputs, outputs = doitlib.build_mnist_model(Input, Flatten, Dense, input_size=input_size, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)

print("Building zero-bias gloro model from saved weights...")

doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

# evaluate hte resulting model
print("Evaluating the resulting zero-bias gloro model...")

x_test, y_test = doitlib.load_mnist_test_data(input_size=input_size)

loss, accuracy = model.evaluate(x_test, y_test, verbose=2)


print(f"Test Loss (zero bias model): {loss:.4f}")
print(f"Test Accuracy (zero bias model): {accuracy:.4f}")


print("Generating output vectors from the test (evaluation) data...")

outputs = model.predict(x_test)
n=len(outputs)

#print(f"We have {n} test outputs we could try the certifier on.")
#print("How many do you want?")
#user_input = int(input(f"Enter a number between 0 and {n}: "))
user_input=n

# Check if input is in the range
if 0 <= user_input <= n:


    # Create a directory to save the images
    #output_dir = "mnist_images"
    #os.makedirs(output_dir, exist_ok=True)
    
    saved_stdout=sys.stdout
    with open(output_file,'w') as f:
        sys.stdout=f
    
        for i in range(user_input):
            test_output = outputs[i].tolist()
            printlist(test_output)
            #mprint(" ")
            #mprint(epsilon)
            mprint("\n")

            # Get the image data
            #image_array = x_test_orig[i]    
            # Convert the image array to a PIL Image object
            #image = Image.fromarray(image_array)    
            # Save the image to a file
            #output_path = os.path.join(output_dir, f"mnist_image_{i}.png")
            #image.save(output_path)
    sys.stdout=saved_stdout
else:
    print("Invalid number entered. No outputs for you!")
        

