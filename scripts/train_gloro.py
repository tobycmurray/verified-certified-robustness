# This code was derived from the file tools/training/train_gloro.py from the gloro github repository
# Copyright (c) 2021 Klas Leino
# License: MIT; See https://github.com/klasleino/gloro/blob/a218dcdaaa41951411b0d520581e96e7401967d7/LICENSE
#
# Contributors (beyond those who contributed to the "gloro" project): Hira Syeda, Toby Murray
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import doitlib
import json

import tensorflow as tf
from gloro.utils import print_if_verbose
from utils import get_data
from utils import get_optimizer
from gloro.models import GloroNet
from gloro.layers import Dense
from gloro.layers import Flatten
from gloro.layers import Input
from gloro.layers import MinMax
from gloro.training import losses
from tensorflow.keras import backend as K
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import LrScheduler
from gloro.training.callbacks import TradesScheduler
from gloro.training.metrics import rejection_rate, vra, clean_acc

from sklearn.metrics import confusion_matrix


def train_gloro(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=None,
        batch_size=None,
        optimizer='adam',
        lr=0.001,
        lr_schedule='fixed',
        trades_schedule=None,
        verbose=False,
        INTERNAL_LAYER_SIZES=[64],
        input_size=28
):
    _print = print_if_verbose(verbose)

    # Load data and set up data pipeline.
    _print('loading data...')

    train, test = doitlib.load_mnist_gloro_data(batch_size=batch_size,input_size=input_size)
    
    
    # Create the model.
    _print('creating model...')

    inputs, outputs = doitlib.build_mnist_model(Input, Flatten, Dense, input_size=input_size, internal_layer_sizes=INTERNAL_LAYER_SIZES)

    g = GloroNet(inputs, outputs, epsilon)

    if verbose:
        g.summary()

    # Compile and train the model.
    _print('compiling model...')

    g.compile(
        loss=losses.get(loss),
        optimizer=get_optimizer(optimizer, lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        # metrics=[rejection_rate]
    )

    print('training model...')
    g.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=[
                      EpsilonScheduler(epsilon_schedule),
                      LrScheduler(lr_schedule),
                  ] + ([TradesScheduler(trades_schedule)] if trades_schedule else []),
    )

    print('model training done.')

    return g


def script(
        dataset,
        epsilon,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=100,
        batch_size=128,
        optimizer='adam',
        lr=1e-3,
        #lr_schedule='decay_to_0.000001',
        lr_schedule='fixed',
        trades_schedule=None,
        plot_learning_curve=False,
        plot_confusion_matrix=False,
        INTERNAL_LAYER_SIZES=[64],
        input_size=28,
):

    g = train_gloro(
        dataset,
        epsilon,
        epsilon_schedule=epsilon_schedule,
        loss=loss,
        augmentation=augmentation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        lr_schedule=lr_schedule,
        trades_schedule=trades_schedule,
        INTERNAL_LAYER_SIZES=INTERNAL_LAYER_SIZES,
        input_size=input_size
    )

    print('getting model accuracy numbers...')
    # Access the training accuracy
    final_training_accuracy = g.history.history['sparse_categorical_accuracy'][-1]

    # Access the validation accuracy (if validation_data was provided)
    final_validation_accuracy = g.history.history['val_sparse_categorical_accuracy'][-1]    

    print(f'model training accuracy: {final_training_accuracy}; validation accuracy: {final_validation_accuracy}')
    
    if plot_learning_curve:
        print('plotting learning curve...')
        history = g.history.history
    
        # learning curve
        # accuracy
        acc = history['sparse_categorical_accuracy']
        val_acc = history['val_sparse_categorical_accuracy']

        # loss
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(acc) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(epochs, acc, 'r', label='Training Accuracy')
        ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

        ax2.plot(epochs, loss, 'r', label='Training Loss')
        ax2.plot(epochs, val_loss, 'b', label='Validation Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')

        plt.tight_layout()
        plt.show()
        fig.savefig('learning_curve.png')

        print('learning curve plotted.')

    print('saving model summary...')
    g.summary()
    g.save("model.keras")
    
    saved_stdout = sys.stdout    
    with open('gloro.summary', 'w') as sys.stdout:
        g.summary()
    sys.stdout=saved_stdout
    
    print('model summary saved.')

    layer_weights = [layer.get_weights()[0] for layer in g.layers if len(layer.get_weights()) > 0]

    assert (len(layer_weights) == len(INTERNAL_LAYER_SIZES)+1)
            
    lipschitz_constants = g.lipschitz_constant()

    for i, weights in enumerate(layer_weights):
        np.savez(f"layer_{i}_weights.npz", weights=weights)

    # Create a directory to save the files if it does not exist
    save_dir = 'model_weights_csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop through each layer, extract weights and biases, and save them
    for i, weights in enumerate(layer_weights):
        np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%f')

    # Loop through each layer, extract weights and biases, and save them
    for i, c in enumerate(lipschitz_constants):
        np.savetxt(os.path.join(save_dir, f'logit_{i}_gloro_lipschitz.csv'), [c], delimiter=',', fmt='%f')
        
    print('model weights extracted.')


    # compute the model's clean accuracy and robustness (rejection rate) via the gloro methods
    x_test, y_test = doitlib.load_mnist_test_data(input_size)
    y_pred = g.predict(x_test)
    accuracy = float(clean_acc(y_test, y_pred).numpy())
    reject_rate = float(rejection_rate(y_test, y_pred).numpy())
    robustness = 1.0 - reject_rate
    the_vra = float(vra(y_test, y_pred).numpy())

    # save the statistics calculated by gloro
    data = {
        "comment": "these statistics are unverified and calculated by the gloro implementation",
        "accuracy": accuracy,
        "rejection_rate": reject_rate,
        "robustness": robustness,
        "vra": the_vra
    }
    file_path = os.path.join(save_dir, "gloro_model_stats.json")
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    
    print("SUMMARY")
    print(f"lr_schedule: {lr_schedule}")
    print(f"epsilon: {epsilon}")
    print(f"dense layer sizes: {INTERNAL_LAYER_SIZES}")
    print(f"(gloro) lipschitz constants: {lipschitz_constants}")
    print(f"(gloro) (clean) accuracy: {accuracy}")
    print(f"(gloro) robustness (1 - rejection rate): {robustness}")
    print(f"(gloro) rejection rate: {reject_rate}")    
    print(f"(gloro) VRA: {the_vra}")
    
    # At the end of your script
    K.clear_session()

import sys

if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} epsilon INTERNAL_LAYER_SIZES epochs [input_size]\n");
    sys.exit(1)

epsilon=float(sys.argv[1])
    
internal_layers=eval(sys.argv[2])

epochs=int(sys.argv[3])

input_size=int(sys.argv[4])

print(f"Running with internal layer dimensions: {internal_layers}")

script(
    dataset='mnist',
    epsilon=epsilon,
    #epsilon_schedule='[0.01]-log-[50%:1.1]',
    epsilon_schedule='fixed',
    batch_size=32,
    lr=1e-3,
    #lr_schedule='decay_after_half_to_0.000001',
    lr_schedule='decay_to_0.000001',
    epochs=epochs,
    # Table B.1 in Leino et al. lists "0.1,2.0,500" in the "loss" but for others it is 1.5 so ...
    #loss='sparse_trades_kl.1.5',
    loss='sparse_crossentropy',    
    augmentation='none',
    INTERNAL_LAYER_SIZES=internal_layers,
    input_size=input_size)
