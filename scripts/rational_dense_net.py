from fractions import Fraction
from typing import List
import tensorflow as tf
import numpy as np
import textwrap

class RationalDenseNetTracing:
    def __init__(self, weights_frac, weights_float):
        self.weights_frac = weights_frac
        self.weights_float = weights_float

    @staticmethod
    def relu_frac(x: Fraction) -> Fraction:
        return x if x > 0 else Fraction(0)

    @staticmethod
    def relu_float(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def forward(self, x_frac: List[Fraction], x_float: np.ndarray, x_raw: np.ndarray, model: tf.keras.Model):
        """
        Forward pass with:
        - Fraction input (list of Fractions, flattened),
        - float32 input (flattened np.ndarray),
        - raw input (unflattened, as Keras expects, e.g. (28,28,1)),
        - and the original Keras model.
        
        Traces the manual Fraction and float32 computations side-by-side,
        and compares against Keras pre-activations layer by layer.
        Also prints what Keras actually fed into each Dense layer.
        """
        print("=== Forward pass start ===")

        keras_pre_models = []
        keras_in_models = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Model that outputs what goes INTO this Dense layer
                in_model = tf.keras.Model(inputs=model.input, outputs=layer.input)
                keras_in_models.append((layer.name, in_model))

                # Model that outputs the pre-activation (x @ W)
                pre_tensor = tf.linalg.matmul(layer.input, layer.kernel)
                pre_model = tf.keras.Model(inputs=model.input, outputs=pre_tensor)
                keras_pre_models.append((layer.name, pre_model))

        for dense_idx, (W_frac, W_float, (lname_in, keras_in), (lname_pre, keras_pre)) in enumerate(
                zip(self.weights_frac, self.weights_float, keras_in_models, keras_pre_models)
        ):
            print(f"\n-- Dense layer {dense_idx} --")
            print(f" Input length: {len(x_frac)}")
            print(f" First few inputs (Fraction): {x_frac[:5]}")
            print(f" First few inputs (float32): {x_float[:5]}")

            # Format keras input
            if x_raw.ndim == 3:  # (28,28,1)
                keras_input = np.expand_dims(x_raw, axis=0)
            else:
                keras_input = x_raw

            # Get Keras input to this layer
            keras_inp = keras_in.predict(keras_input, verbose=0).flatten()
            print(f" Keras input to this layer (first 5): {keras_inp[:5]}")
            
            out_dim, in_dim = len(W_frac), len(W_frac[0]) if W_frac else 0
            print(f" Weight matrix shape: {out_dim} x {in_dim}")
            if out_dim > 0:
                print(f" First row weights (Fraction, first 5): {W_frac[0][:5]}")
                print(f" First row weights (float32, first 5): {W_float[0,:5]}")


            # --- After printing inputs, check for zero vectors ---
            # Fraction inputs: exact zero check
            if all(val == 0 for val in x_frac):
                print(" ✅ Fraction inputs are exactly all zero!")

            # Manual float32 inputs: near-zero check
            #if np.allclose(x_float, 0.0, atol=1e-12):
            if np.all(x_float==0.0):            
                print(" ✅ Manual float32 inputs are (numerically) all zero!")

            # Keras inputs: near-zero check
            if np.all(keras_inp==0.0):
                print(" ✅ Keras inputs are (numerically) all zero!")
                

            # If mismatch: e.g. Fractions are zero but Keras is not
            if all(val == 0 for val in x_frac) and not np.all(keras_inp==0.0):
                print(" ⚠️ MISMATCH: Fraction inputs all zero, but Keras inputs are non-zero!")

            if np.all(x_float==0.0) and not np.all(keras_inp==0.0):
                print(" ⚠️ MISMATCH: Manual float32 inputs all zero, but Keras inputs are non-zero!")
                # Get all indices where keras_inp != 0.0
                nz_indices = np.where(keras_inp != 0.0)[0]
                print(f"   Non-zero Keras inputs ({len(nz_indices)} total):")
                for idx in nz_indices:
                    print(f"    idx={idx}, value={keras_inp[idx]}")
            
            # Manual Fraction pre-activation
            y_frac = [sum(row[i] * x_frac[i] for i in range(len(x_frac))) for row in W_frac]
            print(f" Raw pre-activation (Fraction, first 5): {y_frac[:5]}")

            # Manual float32 pre-activation
            y_float = W_float @ x_float
            print(f" Raw pre-activation (float32, first 5): {y_float[:5]}")

            # Keras pre-activation
            keras_y = keras_pre.predict(keras_input, verbose=0).flatten()
            print(f" Raw pre-activation (Keras, first 5): {keras_y[:5]}")

            # Discrepancy check
            mask = np.isclose(y_float, keras_y, atol=1e-7, rtol=1e-7)
            if not np.all(mask):
                print("⚠️ Discrepancy between manual float32 and Keras pre-activation!")

                # Find indices where they differ
                diff_indices = np.where(~mask)

                for idx in zip(*diff_indices):
                    print(f"Index {idx}: y_float={y_float[idx]}, keras_y={keras_y[idx]}")            

            if np.all(y_float<=0.0):
                print("[INFO] float32 pre-activations are all negative, so next layer inputs will all be 0.0")

                if not np.all(keras_y<=0.0):
                    print("⚠️  keras pre-activations are NOT all negative, so next layer inputs should differ from float32")
                    # Find the indices where keras_y > 0
                    pos_indices = np.where(keras_y > 0.0)

                    for idx in zip(*pos_indices):
                        print(f"Index {idx}: y_float={y_float[idx]}, keras_y={keras_y[idx]}")
                        np.set_printoptions(threshold=np.inf)

                        # Save arrays
                        np.save("W.npy", W_float.astype(np.float32))   # weights stored as float32
                        np.save("x.npy", x_float.astype(np.float64))   # input stored as float64 (important!)

                        print("W_float dtype:", W_float.dtype, "shape:", W_float.shape, "ndim:", W_float.ndim, "C/F order:", W_float.flags)
                        print("x_float dtype:", x_float.dtype, "shape:", x_float.shape, "ndim:", x_float.ndim, "C/F order:", x_float.flags)
                        print("Result dtype of @ :", (W_float @ x_float).dtype)


                        prog = textwrap.dedent(f"""
import numpy as np
import tensorflow as tf

# Load arrays (bit-identical to generator's save)
W_rowmajor = np.load("W.npy")   # float32
x0 = np.load("x.npy")           # float64

print("W_rowmajor dtype:", W_rowmajor.dtype, "shape:", W_rowmajor.shape)
print("x0 dtype:", x0.dtype, "shape:", x0.shape)

# Build single-layer Dense model
model = tf.keras.Sequential([
    tf.keras.layers.Dense({W_float.shape[0]}, use_bias=False, activation=None,
                          input_shape=(x0.shape[0],))
])
model.layers[0].set_weights([W_rowmajor.T])

# Keras output
keras_y = model(x0.reshape(1, -1).astype(np.float32), training=False).numpy()[0]

# Manual NumPy matmul (promotion happens automatically: float32 @ float64 → float64)
manual_y = (W_rowmajor @ x0).astype(np.float32)

print("Keras output:", keras_y)
print("Manual output:", manual_y)

mismatches = np.where(np.sign(keras_y) != np.sign(manual_y))
for i in mismatches[0]:
    print(f"Mismatch at index {{i}}: keras={{keras_y[i]}}, manual={{manual_y[i]}}")
""")

                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix="-reproducer.py", dir=".", delete=False, mode="w") as f:
                            output_file = f.name
                            f.write(prog)

                        print(f"Reproducer saved to file {output_file}")
                        print("Also saved W.npy and x.npy")
                        
                        import sys
                        sys.exit(1)
        
            if np.all(keras_y<=0.0):
                print("[INFO] keras pre-activations are all negative, so next layer inputs will all be 0.0")
                    
            # Apply ReLU for next layer’s input
            y_frac = [self.relu_frac(val) for val in y_frac]
            y_float = self.relu_float(y_float)

            # Update state
            x_frac, x_float = y_frac, y_float

        print("\n=== Forward pass end ===")
        print(f" Final output (Fraction, first 10): {x_frac[:10]}")
        print(f" Final output (float32, first 10): {x_float[:10]}")
        return x_frac, x_float, keras_y
    

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
def keras_to_rational_dense_net(model: tf.keras.Model) -> "RationalDenseNetTracing":
    """
    Convert a TensorFlow/Keras model with only Dense (no bias, ReLU activations)
    into a RationalDenseNetTracing with both Fraction weights and original float32 weights.
    """
    weights_frac_list = []
    weights_float_list = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # Extract weights: shape (in_dim, out_dim)
            W = layer.get_weights()[0]

            # Transpose once → shape (out_dim, in_dim)
            W_T = W.T.astype(np.float32)

            # Fractions
            W_fraction = [[Fraction(str(w_ij)) for w_ij in row] for row in W_T]
            weights_frac_list.append(W_fraction)

            # Store float32 transposed version
            weights_float_list.append(W_T)

    return RationalDenseNetTracing(weights_frac_list, weights_float_list)

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
