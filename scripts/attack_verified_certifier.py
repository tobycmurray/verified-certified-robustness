# This code attempts to find a pair of points (xa,xb) where (in float32 arithmetic) xa maps to
# the 0 output vector but xb is the closest float32 that does not map to the 0 output vector.
#
# The idea is that the certifier will certify xb ||xb-xa|| as robust, even though xa is a
# (float32 arithmetic) couter-example, since argmax(M(xa)) != argmax(M(xb))
#
# this works well for the 0 output vector since we can relatively easily find inputs that map to
# the 0 output vector (by gradient descent) and because rounding to 0 is clearly hiding precision
# due to floating point rounding. In that sense the 0 output vector is a good foothold to attack,
# in a way that other artibrary outputs may not be.
#
# Something interesting to note is that xa and xb might not be guaranteed to be a counter-example.
# There are lots of reasons for this. One curious one is that there exist models M and inputs x
# such that M(x) can be 0 in real arithmetic (and even in float32 arithmetic) but non-zero when
# computed by Keras/Tensorflow. In particular, we have found example inputs where, writing W
# for the first layer weight matrix, ReLU(W.x) = 0 in real and float32 arithmetic (when implemented
# manually) but when computed by Keras/Tensorflow, the output is non-zero (e.g. a single small
# non-zero quantity in the output 3.166496753692627e-08).
#
# This happens because W.x is a vector of negative numbers (when computed using manual float32 or real
# arithmetic), and so ReLU(R.x) is the zero vector. However, when computed by TensorFlow,
# one of the entries in W.x ends up being very small and positive (eg. 3.166496753692627e-08), and so
# ReLU(W.x) here is non-zero.
#
#Therefore, there are lots of cases:
#
# Expression  | Real arithmetic | Float32 arithmetic  | Keras/Tensorflow implementation
# ReLU(M . x) | 0               | 0                   | non-zero (due to TF weirdness)
# M . x       | 0               | 0                   | 0
# M . x       | non-zero        | 0 (due to rounding) | 0 (due to rounding)
# ReLU(M . x) | 0               | 0                   | 0
# ReLU(M . x) | non-zero        | 0 (due to rounding) | 0 (due to rounding)

import doitlib
import numpy as np
import tensorflow as tf
from sys import stdout
from PIL import Image
import os
import json
import sys
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer

from rational_dense_net import *

# artibtrary precision math
from mpmath import mp, mpf, sqrt, nstr

# massive precision
mp.dps=150

def vector_to_mph(v):
    return list(map(mpf, v.flatten().tolist()))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES model_weights_csv_dir input_size\n");
    sys.exit(1)


dataset=sys.argv[1]
    
INTERNAL_LAYER_SIZES=eval(sys.argv[2])

csv_loc=sys.argv[3]+"/"

input_size=int(sys.argv[4])


    
inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size, dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)


doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

            

def find_input_for(model, target, min_distance_from=None):
    # Initialize input variable (randomly or meaningfully)
    #x = tf.Variable(tf.zeros([1] + list(model.input_shape[1:])))
    
    alpha = 0.01  # scale factor for noise
    x = tf.Variable(alpha * tf.random.normal([1] + list(model.input_shape[1:])))

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

    lam = 0.5
    for step in range(max_steps):
        min_from = min_distance_from if min_distance_from is not None else x
        if min_distance_from is None:
            assert  tf.reduce_sum(tf.square(x - min_from)) == 0.0
        with tf.GradientTape() as tape:
            y_pred = model(x, training=False)
            loss_target = tf.reduce_sum(tf.abs(y_pred - target))
            #delta = tf.clip_by_value(x - min_from, -1e6, 1e6)
            #loss_distance = tf.norm(delta, ord=2)
            loss_distance = tf.reduce_sum(tf.square(x - min_from))

            loss = loss_target + lam * loss_distance
        
        grads = tape.gradient(loss, x)
        grads = tf.clip_by_norm(grads, 1.0)  # prevent explosion
        optimizer.apply_gradients([(grads, x)])
    

        # Optional: Print loss for monitoring
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

        if loss.numpy() == 0.0:
            #print(f"Converged at step {step}, Loss: {loss.numpy()}")
            break
    return x

count_zeros = 0



# given x0 that maps to the 0 vector (with argmax 0), use binary search explore (x0 + i * noise) to find the boundary
# between the argmax 0 output and some non-zero output. We return all points found along the
# way. However the idea is that we find two points (xa, xb) where xa maps to the 0 vector and xb
# maps to non-zero vector. In reality xa will map to non-zero also in real arithmetic (due to floating
# point imprecision/rounding), and xb will be very small, and ||xb-xa|| will therefore also be very
# small. Our certifier will mistakenly conclude that xb is robust for ||xb-xa|| and for radii larger than
# that, even though xa is a counter-example in reality (because the certifier only thinks in terms of
# real arithmetic and in real arithemtic xa is non-zero and will surely share the same argmax as xb.
def random_search(model, x0): 
    # generate noise once and then just scale it
    noise = np.random.randn(*x0.shape)
    x0 = np.asarray(x0, dtype=np.float32).reshape(1, 28, 28, 1)  # adjust shape
    initial_x0 = x0.copy()
    last_found=0
    working=[]
    low=0.0
    high=0.05
    eps=high
    x_try=x0

    x_try = x0 + eps * noise
    # model output
    y_try = model(x_try, training=False).numpy()[0]

    # condition: argmax is not 0
    argmax=np.argmax(y_try)
    if argmax == 0:
        print(f"high {high} didn't produce a flip. Re-run and try again.")
        sys.exit(1)

    x0_mph = vector_to_mph(x0)
    x_try_mph = vector_to_mph(x_try)
    l2_mph = l2_norm_mph(x_try_mph, x0_mph)

    w={"x": x_try,
       "y": y_try,
       "norm": np.linalg.norm(x_try-x0),
       "norm_mph": l2_mph,
       "argmax": argmax,
       "eps": eps,
       "flips": True}
    working.append(w)
    
    while high>=low:
        mid = (high+low)/2
        eps=mid
        x_try = x0 + eps * noise
        # model output
        y_try = model(x_try, training=False).numpy()[0]
        x_try_mph = vector_to_mph(x_try)
        l2_mph = l2_norm_mph(x_try_mph, x0_mph)
        
        argmax=np.argmax(y_try)
        if argmax == 0:
            # not found yet -- search upper half
            low=np.nextafter(mid, np.float32(np.inf))
            flips=False
        else:
            # found here
            flips=True
            high=np.nextafter(mid, np.float32(-np.inf))
        w={"x": x_try,
            "y": y_try,
            "norm": np.linalg.norm(x_try-x0),
            "norm_mph": l2_mph,
            "argmax": argmax,
            "eps": eps,               
            "flips": flips}
        working.append(w)
    return working
        
    

from itertools import combinations

def closest_opposite_flips(items):
    """
    Finds the closest pair of dicts from `items` such that their 'flips' differ.
    Closeness is defined as minimal L2 distance between their 'x' vectors, computed
    with mpmath to ensure we don't lose precision due to floating point.
    """
    best_pair = None
    best_dist = float("inf")

    for a, b in combinations(items, 2):
        if a["flips"] != b["flips"]:
            a_mph = vector_to_mph(a["x"])
            b_mph = vector_to_mph(b["x"])
            dist = l2_norm_mph(a_mph, b_mph)
            if dist < best_dist:
                best_dist = dist
                best_pair = (a, b)

    return best_pair, best_dist

# first search around for an input that produces the 0 vector output
# keep going until we find it
while count_zeros < 10:
    # we are searching for inputs that map to the 0 vector output
    target = tf.constant([0.0] * 10, dtype=tf.float32)
    print("Searching for an input that maps to the 0 output vector...")
    x = find_input_for(model, target)
    
    x_final = x.numpy()
    
    y_final = model(x, training=False).numpy()


    max_logit = np.max(y_final)

    count_zeros = np.sum(y_final == 0.0)

    if count_zeros==10:
        x0 = x_final

        print("Looking for perturbation that produces non-zero argmax...")
        working=random_search(model, x0)

        print("Finding closest pair that flips the argmax...")
        best_pair, best_dist = closest_opposite_flips(working)

        print(f"best pair dist is: {best_dist}")
        a,b = best_pair
        print(f"item a:")
        print(f"      y: {a['y']}")
        print(f" argmax: {a['argmax']}")
        print(f"  flips: {a['flips']}")
        print(f"item b:")
        print(f"      y: {b['y']}")
        print(f" argmax: {b['argmax']}")        
        print(f"  flips: {b['flips']}")

        # print out whichever filps as the certifier input
        if a['flips']:
            to_print=a
            is_zero=b
        else:
            to_print=b
            is_zero=a


        # do a final test in float32 arithmetic
        y_is_zero = model(is_zero["x"], training=False).numpy()[0]        
        y_flips = model(to_print["x"], training=False).numpy()[0]
        argmax_y_is_zero = np.argmax(y_is_zero)
        argmax_y_flips = np.argmax(y_flips)
        if argmax_y_is_zero != 0 or argmax_y_flips == 0:
            print("float32 argmaxes do not compute as expected!")
            print(f"y_is_zero: {y_is_zero}")
            print(f"argmax_y_is_zero (expected 0): {argmax_y_is_zero}")
            print(f"y_flips: {y_flips}")
            print(f"argmax_y_filps (expected non-zero): {argmax_y_flips}")
            sys.exit(1)
        else:
            x_is_zero_mph = vector_to_mph(is_zero["x"])
            x_flips_mph = vector_to_mph(to_print["x"])
            dist = l2_norm_mph(x_is_zero_mph, x_flips_mph)
            if dist != best_dist:
                print(f"couldn't confirm distance between found pair!")
                print(f"best_dist: {best_dist}")
                print(f"(recomputed) dist: {dist}")
                sys.exit(1)
        print(f"Manually confirmed x0 and x1 such that ||x1-x0||=={dist}, F(x0)=={argmax_y_is_zero} but F(x1)=={argmax_y_flips}")

        
        x_final=to_print["x"]

        # trace the output
        #rational_model = keras_to_rational_dense_net(model)
        #x_final_frac = to_fraction_list(x_final.reshape(-1))
        #y_frac, y_float, y_keras = rational_model.forward(x_final_frac, x_final.reshape(-1), x_final, model)
        
        
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

        import tempfile
        
        # Create a unique temporary file in the current directory
        with tempfile.NamedTemporaryFile(suffix=".png", dir=".", delete=False) as f:
            output_file = f.name

        image.save(output_file)
        print(f"Image that produces the flip saved to: {output_file}")
        
        print("Give this to the certifier (output vector followed by radius): ")
        y=to_print['y']
        
        def mprint(string):
            print(string, end="")

        radius=best_dist
        while radius < 1000000*best_dist:
            for i in range(len(y)):
                s = "{:.150f}".format(y[i])
                mprint(s)
                if i<len(y)-1:
                    mprint(",")
            mprint(" ")
            s = str(radius)
            mprint(s)
            print("")
            radius = radius*10.0
        
            
        sys.exit(1)
        
