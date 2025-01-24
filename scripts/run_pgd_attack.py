# Derived from examples/adversarial_training_data_augmentation.py
# of the ART repo: https://github.com/Trusted-AI/adversarial-robustness-toolbox
# License: MIT
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

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

from art.attacks.evasion import ProjectedGradientDescent, AutoAttack, FastGradientMethod, MomentumIterativeMethod
from art.estimators.classification import KerasClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer
from art.utils import load_dataset

# make output more readable when we get annoying warnings being printed
print("")
print("")

if len(sys.argv) != 8 and len(sys.argv) != 6:
    print(f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir epsilon MAX_ITER input_size [certifier_results.json disagree_output_dir]\n");
    sys.exit(1)

INTERNAL_LAYER_SIZES=eval(sys.argv[1])

csv_loc=sys.argv[2]+"/"

epsilon=float(sys.argv[3])

MAX_ITER=int(sys.argv[4])

input_size=int(sys.argv[5])

json_results_file=None
disagree_output_dir=None
if len(sys.argv) == 8:
    json_results_file=sys.argv[6]
    disagree_output_dir=sys.argv[7]+"/"

    if os.path.exists(disagree_output_dir):
        raise FileExistsError(f"The directory '{disagree_output_dir}' already exists.")


print(f"Running with internal layer dimensions: {INTERNAL_LAYER_SIZES}")
print(f"Running with input size: {input_size}")

print(f"Running attacks with epsilon: {epsilon}")

print(f"MAX_ITER: {MAX_ITER}")


inputs, outputs = doitlib.build_mnist_model(Input, Flatten, Dense, input_size=input_size, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)

print("Building zero-bias gloro model from saved weights...")


doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)

x_test, y_test = doitlib.load_mnist_test_data(input_size=input_size)

labels_true = np.argmax(y_test, axis=1)


# Build a Keras image augmentation object and wrap it in ART
batch_size = 50

classifier = KerasClassifier(model, clip_values=(0.0,1.0), use_logits=False)
model.summary()

x_test_pred_vecs = classifier.predict(x_test)
x_test_pred = np.argmax(x_test_pred_vecs, axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test, axis=1))

# build a list of targets for targeted attacks
targets = []
for i in range(len(x_test_pred)):
    t = None
    if x_test_pred[i] != labels_true[i]:
        # prefer the true label where applicable
        t = labels_true[i]
    else:
        # otherwise, pick the second highest
        indexed_vector = sorted(enumerate(x_test_pred_vecs[i]), key=lambda x: x[1], reverse=True)
        t = indexed_vector[1][0]
    targets.append(t)
targets = np.array(targets)

attacks = []

saved_epsilon=epsilon

# try to avoid false positives
epsilon=epsilon-0.00000005

#auto = AutoAttack(estimator=classifier, norm=2, eps=epsilon)
#attacks.append(auto)
NUM_RANDOM_INIT=1
fast = FastGradientMethod(estimator=classifier, norm=2, eps=epsilon, num_random_init=NUM_RANDOM_INIT, eps_step=0.01, minimal=True, targeted=True)
attacks.append(fast)
fast2 = FastGradientMethod(estimator=classifier, norm=2, eps=epsilon, num_random_init=NUM_RANDOM_INIT, eps_step=0.01, minimal=True, targeted=False)
attacks.append(fast2)

momentum = MomentumIterativeMethod(estimator=classifier, norm=2, eps=epsilon, eps_step=0.01, max_iter=MAX_ITER, verbose=False, targeted=False)
attacks.append(momentum)
momentum2 = MomentumIterativeMethod(estimator=classifier, norm=2, eps=epsilon, eps_step=0.01, max_iter=MAX_ITER, verbose=False, targeted=True)
attacks.append(momentum2)

pgd = ProjectedGradientDescent(classifier, norm=2, eps=epsilon, eps_step=0.01, max_iter=MAX_ITER, verbose=False, num_random_init=NUM_RANDOM_INIT, targeted=False)
attacks.append(pgd)
pgd2 = ProjectedGradientDescent(classifier, norm=2, eps=epsilon, eps_step=0.01, max_iter=MAX_ITER, verbose=False, num_random_init=NUM_RANDOM_INIT, targeted=True)
attacks.append(pgd2)

epsilon=saved_epsilon

x_test_adv=[]
predict_adv=[]
labels_adv=[]
for a in attacks:
    print(f"Running attack {a}...")
    if a.targeted:
        s = a.generate(x_test,targets)
    else:
        s = a.generate(x_test)
    # generate adversarial examples using the given attack
    x_test_adv.append(s)
    # Evaluate the model on the adversarial samples
    p = model.predict(s)
    predict_adv.append(p)
    labels_adv.append(np.argmax(p, axis=1))

n=labels_adv[0].shape[0]
assert n == x_test.shape[0]


robustness_log=[]

if json_results_file is not None:
    with open(json_results_file, 'r') as f:
        robustness_log = json.load(f)

robustness = [d for d in robustness_log if "certified" in d]

assert len(robustness)==n or (robustness==[] and json_results_file is None)

disagree=0
false_positive=0
max_fp_norm=-1.0
min_fp_norm=-1.0

max_disagree_norm=-1.0
min_disagree_norm=-1.0

unsound=0

def vector_to_mph(v):
    return list(map(mpf, v.flatten().tolist()))

def l2_norm_mph(vector1, vector2):
    return sqrt(sum((x - y)**2 for x, y in zip(vector1, vector2)))

i=0
while i<n:
    i_disagrees=False
    for ai in range(len(attacks)):
        x_adv=x_test_adv[ai][i]
        x=x_test[i]
        if labels_adv[ai][i] != x_test_pred[i]:
            # calculate the norm using arbitrary precision arithmetic to make sure it really is a valid attack
            x_adv_mph = vector_to_mph(x_adv)
            x_mph = vector_to_mph(x)        
            l2_norm = l2_norm_mph(x_adv_mph, x_mph)
            if (l2_norm > epsilon):
                if false_positive == 0:
                    max_fp_norm=l2_norm
                    min_fp_norm=l2_norm
                else:
                    if max_fp_norm < l2_norm:
                        max_fp_norm=l2_norm
                    if min_fp_norm > l2_norm:
                        min_fp_norm=l2_norm
                # FIXME: at the moment we are double counting false positives across the different attacks!
                false_positive=false_positive+1
            else:
                if l2_norm==0.0:
                    print("Got a disagreement with a zero norm!")
                    print(f"   x    : {x}")
                    print(f"   x_adv: {x_adv}")
                    print(f"   l2   : {l2_norm}")
                # first disagreement
                if disagree == 0 and not i_disagrees:
                    max_disagree_norm=l2_norm
                    min_disagree_norm=l2_norm
                else:
                    if max_disagree_norm < l2_norm:
                        max_disagree_norm=l2_norm
                    if min_disagree_norm > l2_norm:
                        min_disagree_norm=l2_norm
                # this is not a false positive
                i_disagrees=True        
            
                if robustness!=[]:

                    r = robustness[i]
                    robust = r["certified"]

                    if robust:
                        # found an attack when the certifier said the output was robust!
                        unsound=unsound+1
                        x=x_test[i]
                        x_adv=x_test_adv[ai][i]
                        lab_y=x_test_pred[i]                
                        y_adv=predict_adv[ai][i]
                        lab_y_adv=np.argmax(y_adv, axis=0)
                        if disagree_output_dir is not None:
                            os.makedirs(disagree_output_dir)
                        input_path = os.path.join(disagree_output_dir, f"input_{i}.npy")
                        np.savetxt(disagree_output_dir+f"/unsound_{i}_x.csv", x, delimiter=',', fmt='%f')
                        np.savetxt(disagree_output_dir+f"/unsound_{i}_x_adv.csv", x_pgd, delimiter=',', fmt='%f')
                        with open(disagree_output_dir+f"/unsound_{i}_summary.txt", "w") as f:
                            f.write(f"L2 Norm    : {l2_norm}\n")
                            f.write(f"Y label    : {lab_y}\n")                    
                            f.write(f"Y PGD      : {y_pgd}\n")
                            f.write(f"Y PGD label: {lab_y_pgd}\n")
    if i_disagrees:
        disagree=disagree+1
    i=i+1

agree=n-disagree
print(f"Model accuracy: {nb_correct_pred/x_test.shape[0] * 100}")

#assert(agree >= np.sum(labels_pgd == labels_true))

print("Robustness on PGD adversarial samples: %.2f%%" % (agree / n * 100))
if disagree > 0:
    print(f"Norms of non-false-positive vectors that cause classification changes: min: {min_disagree_norm}; max: {max_disagree_norm}")

print(f"False positives in PGD attack: {false_positive}")
if false_positive > 0:
    print(f"Norms of false positive vectors: min: {min_fp_norm}; max: {max_fp_norm}")

print(f"Number of PGD attacks succeeding against certified robust outputs: {unsound}")
