import tensorflow as tf

# artibtrary precision math
from mpmath import mp, mpf, sqrt

import doitlib
import os
import json
import sys
import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate, vra, clean_acc


if len(sys.argv) != 7:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES model_weights_csv_dir epsilon input_size max_tries\n");
    sys.exit(1)

dataset=sys.argv[1]
INTERNAL_LAYER_SIZES=eval(sys.argv[2])

csv_loc=sys.argv[3]+"/"

epsilon=float(sys.argv[4])

input_size=int(sys.argv[5])

max_tries=int(sys.argv[6])

inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size, dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)


doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)

labels_true = np.argmax(y_test, axis=1)


# using the load_model method produces a model that is a *lot* more conservative than the originally-trained model. Don't know why
# Therefore, instead, we load the model weights from the saved ones and then freeze the lc with a "max_tries" value manually tuned
# to ensure that the resulting model produces the same statistics as the originally-trained one and is therefore no more conservative
#g = GloroNet.load_model("2025-01-25_09:27:46/model.keras.gloronet",converge=False)

# Build the gloronet from the saved weights
g = GloroNet(model=model, epsilon=epsilon)

g.freeze_lc(max_tries=max_tries)

gloro_y_pred = g.predict(x_test)
gloro_clean_acc = float(clean_acc(y_test, gloro_y_pred).numpy())
gloro_rejection_rate = float(rejection_rate(y_test, gloro_y_pred).numpy())
gloro_robustness=1.0-gloro_rejection_rate
gloro_vra = float(vra(y_test, gloro_y_pred).numpy())
bot_index = tf.cast(tf.shape(gloro_y_pred)[1] - 1, 'int64')
preds = tf.argmax(gloro_y_pred, axis=1)
n=len(x_test)
bots=0
answers=[]

def print_debug_message(msg):
    answer={}
    answer["debug_msg"] = msg
    answers.append(answer)
print_debug_message(f"For posterity: max_tries:  {max_tries}")    
print_debug_message(f"For checking: gloro clean accuracy (compare to saved value) is: {gloro_clean_acc}")
print_debug_message(f"For checking: gloro rejection rate (compare to saved value) is: {gloro_rejection_rate}")
print_debug_message(f"For checking: gloro robustness (compare to saved value) is: {gloro_robustness}")
print_debug_message(f"For checking: gloro VRA (compare to saved value) is: {gloro_vra}")

for p in preds:
    answer = {}
    if p == bot_index:
        bots=bots+1
        answer["certified"]=False
    else:
        answer["certified"]=True
    answers.append(answer)

print(json.dumps(answers))




