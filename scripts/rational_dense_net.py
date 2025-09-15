from fractions import Fraction
from typing import List
import tensorflow as tf
import numpy as np


# ===============================================================
# Rational Neural Network Implementation
# ===============================================================
class RationalDenseNet:
    def __init__(self, weights: List[List[List[Fraction]]]):
        """
        weights: list of layers,
                 each is a 2D list (matrix) of Fractions [out_dim][in_dim].
                 No biases are supported.
        """
        self.weights = weights

    def relu(self, x: Fraction) -> Fraction:
        return x if x > 0 else Fraction(0)

    def forward(self, x: List[Fraction]) -> List[Fraction]:
        """
        Forward pass with Fraction input vector.
        """
        for layer_index, W in enumerate(self.weights):
            # Matrix multiplication: y = W * x
            y = [sum(W_row[i] * x[i] for i in range(len(x))) for W_row in W]
            # Apply ReLU except for last layer
            if layer_index < len(self.weights) - 1:
                y = [self.relu(val) for val in y]
            x = y
        return x


# ===============================================================
# Conversion from Keras -> RationalDenseNet
# ===============================================================
def keras_to_rational_dense_net(model: tf.keras.Model) -> RationalDenseNet:
    """
    Convert a TensorFlow/Keras model with only Dense (no bias, ReLU activations)
    into a RationalDenseNet with Fraction weights.
    """
    weights_list = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # Extract weights
            W = layer.get_weights()[0]

            # Convert to Fractions (note: Keras stores as [in_dim, out_dim])
            W_fraction = [[Fraction(str(w_ij)) for w_ij in row] for row in W.T]
            weights_list.append(W_fraction)

        else:
            # Skip non-Dense layers (e.g., Input, Flatten)
            continue

    return RationalDenseNet(weights_list)


# ===============================================================
# Input Conversion Utilities
# ===============================================================
def to_fraction_list(x, max_denominator=None):
    """
    Convert a 1D numpy array (or list of floats/ints) into a list of Fractions.

    Parameters
    ----------
    x : array-like
        Input vector (e.g. flattened MNIST image).
    max_denominator : int or None
        If set, approximate floats with denominator <= max_denominator.
        If None, store exact rationals (may produce large denominators).
    """
    fractions = []
    for val in x:
        if isinstance(val, (int, np.integer)):
            fractions.append(Fraction(val, 1))
        elif isinstance(val, (float, np.floating)):
            if max_denominator:
                fractions.append(Fraction.from_float(val).limit_denominator(max_denominator))
            else:
                fractions.append(Fraction(str(val)))  # exact string parse
        else:
            raise TypeError(f"Unsupported type {type(val)} for conversion to Fraction")
    return fractions
