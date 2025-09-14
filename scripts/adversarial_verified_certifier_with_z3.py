# This script finds and validates a candidate attack on our verified robustness certifier
# Specifically, it considers a neural network with just two neurons: w1 and w2 that maps
# 1-element inputs x to 2-element outputs [y1, y2], with no activation function and no
# bias terms. Graphically it would look like this:
#      +---- w1 ------> y1 
#     /
#   x
#     \
#      +---- w2 ------> y2
#
# where y1 = w1 * x and y2 = w2 * x
#
# Specifically, we search for w1 and w2, plus inputs xa and xb, such that
# w2 > w1 and xb > xa, and
# y2b > y1b, but y1a == y2a
# and thus argmax([y1a,y2a]) == 0 but argmax([y1b,y2b]) == 1
#
# Since w2 > w1, the only way that y1a == y2a is by floating point rounding.
# Thus floating point rounding is causing the argmax to (appear to) flip,
# when it shouldn't. The idea is that the example found could be one that
# our certifier certifies as robust, against a radius of xb - xa, even though
# when run on the floating-point model, it won't be robust at this radius.

# pip install z3-solver
from z3 import *
import sys

print("Finding a solution...")

# setting this to True finds a solution but not one that the certifier certifies as robust
WEIGHTS_NOT_ADJACENT=False

F32 = Float32()
RNE = RNE()

def show_f32_exact(name, expr, m):
    val = m.eval(expr, model_completion=True)
    bits = m.eval(fpToIEEEBV(expr), model_completion=True).as_long()
    print(f"{name}: bits=0x{bits:08x}  exact={val}")

# not necessary to find a solution
def fp_next_up_pos_normal(a_f32, b_f32):
    """b_f32 is the next float32 after a_f32 (for positive normal a_f32)."""
    a_bits = fpToIEEEBV(a_f32)
    b_bits = fpToIEEEBV(b_f32)
    return And(
        fpIsNormal(a_f32),
        fpIsNormal(b_f32),
        fpGT(a_f32, FPVal(0.0, F32)),      # positive
        fpGT(b_f32, a_f32),
        b_bits == a_bits + BitVecVal(1, 32)
    )

def fp_not_next_up(a_f32, b_f32):
    a_bits = fpToIEEEBV(a_f32)
    b_bits = fpToIEEEBV(b_f32)
    return b_bits != a_bits + BitVecVal(1, 32)

# not necessary to find a solution
def in_binade(x_f32, e):
    """x in [2^e, 2^(e+1)) as float32 normals (helps the solver)."""
    lo = FPVal(2**e, F32)
    hi = FPVal(2**(e+1), F32)
    return And(fpIsNormal(x_f32), fpGEQ(x_f32, lo), fpLT(x_f32, hi))

def fp_is_finite_normal(v):
    return And(Not(fpIsNaN(v)), Not(fpIsInf(v)), Not(fpIsSubnormal(v)))

# Vars (all float32)
w1f, w2f, xaf, xbf = FP('w1f', F32), FP('w2f', F32), FP('xaf', F32), FP('xbf', F32)

s = Solver()
s.set(timeout=30000)  # 30 second timeout
# s.set(rlimit=1000000)  # resource budget

if WEIGHTS_NOT_ADJACENT:
    weight_lower_bound = 0.0
else:
    weight_lower_bound = 0.1

# Basic domains
s.add(fpIsNormal(w1f), fpIsNormal(w2f))
s.add(fpGT(w1f, FPVal(weight_lower_bound, F32)), fpGT(w2f, w1f), fpGT(1.0, w2f))  # 0.1 < w1 < w2 < 1.0

s.add(fpIsNormal(xaf), fpIsNormal(xbf))
s.add(fpGT(xaf, FPVal(weight_lower_bound, F32)), fpGT(xbf, xaf), fpGT(1.0, xbf))  # 0.1 < xa < xb < 1.0

if WEIGHTS_NOT_ADJACENT:
    s.add(fp_not_next_up(w1f, w2f))

# these constraints turned out not to be necessary to find solutions
#
# Make weights adjacent (greatly shrinks search)
#s.add(fp_next_up_pos_normal(w1f, w2f))
# Inputs constrained to a single binade (ULP uniform there)
#E_BIN = 1                    # works for 23 and 5 and 1
#s.add(in_binade(xaf, E_BIN))  # xa ∈ [2^E_BIN, 2^(E_BIN+1))
#s.add(fp_next_up_pos_normal(xaf, xbf))  # xb = nextUp(xa)


# Float32 products
y1af = fpMul(RNE, xaf, w1f) # y1a = xa * w1
y2af = fpMul(RNE, xaf, w2f) # y2a = xa * w2
y1bf = fpMul(RNE, xbf, w1f) # y1b = xb * w1
y2bf = fpMul(RNE, xbf, w2f) # y2b = xb * w2

# Keep outputs finite normals (avoid subnormals/NaNs/Infs)
for y in (y1af, y2af, y1bf, y2bf):
    s.add(fp_is_finite_normal(y))

# --- Desired behavior (pure float32 semantics) ---
# At xa: tie -> ArgMax would pick neuron 1 if ties go to index 0
s.add(fpEQ(y1af, y2af))
# note if we try for fpGT here, we don't find a solution.
# constraining this case to have the output be equal seems essential

# At xb: strict neuron 2 win
s.add(fpGT(y2bf, y1bf))

# this is not strictly necessary and, in any case, it doeesn't
# guarantee the certifier will say the solution is robust
#
# we want it to cetify robust, i.e. L[i][x] * e < v'[x] - v'[i]
# in our case, e is the input-difference: xb - xa
# L[i][x] is the weights-difference: w2 - w1
# v'[x] - v'[i] is the logit-difference: y2b - y1b
#s.add(fpLT(fpMul(RNE, fpSub(RNE, w2f, w1f), fpSub(RNE, xbf, xaf)), fpSub(RNE, y2bf, y1bf)))

print(s.to_smt2())


print("Solving...")
res = s.check()
print("Result:", res)

if res != sat:
    print("No model found under current bounds.")
    sys.exit(0)
    
m = s.model()

show_f32_exact("w1f", w1f, m)
show_f32_exact("w2f", w2f, m)
show_f32_exact("xaf", xaf, m)
show_f32_exact("xbf", xbf, m)
show_f32_exact("y1af", y1af, m)
show_f32_exact("y2af", y2af, m)
show_f32_exact("y1bf", y1bf, m)
show_f32_exact("y2bf", y2bf, m)

def get_bits(f):
    bits = m.eval(fpToIEEEBV(f)).as_long()
    return eval(f"0x{bits:08x}")

# now build the model using the weights found and test it using the inputs found
import numpy as np
import tensorflow as tf

print("\nTesting the found solution with a TensorFlow model...")

# reconstruct the float32 values from the bitvectors just to be safe
w1_bits = get_bits(w1f)
w2_bits = get_bits(w2f)
xa_bits = get_bits(xaf)
xb_bits = get_bits(xbf)

def f32_from_bits(u32):
    """Exact float32 from a 32-bit IEEE-754 pattern."""
    u = tf.constant(u32, dtype=tf.uint32)
    return tf.bitcast(u, tf.float32)

w1 = f32_from_bits(w1_bits)
w2 = f32_from_bits(w2_bits)
xa = f32_from_bits(xa_bits)
xb = f32_from_bits(xb_bits)

# Build Dense(2) no-bias float32 and set weights [[w1, w2]]
dense = tf.keras.layers.Dense(units=2, use_bias=False, dtype=tf.float32)
_ = dense(tf.constant([[0.0]], tf.float32))  # build
dense.set_weights([np.array([[w1.numpy(), w2.numpy()]], dtype=np.float32)])

# Run the model on xa, xb
X = tf.stack([xa, xb])[:, None]  # shape [2,1]
Y = dense(X)

# Print decimal (round-trip-safe for f32) and exact bit patterns
np.set_printoptions(floatmode='unique', precision=20)

print("Weights: ")
print("  w1=%.150g" % (w1))
print("  w2=%.150g" % (w2))

print("Inputs: ")
print("  xa=%.150g" % (xa))
print("  xb=%.150g" % (xb))
print("Input difference: xb - xa = %.150g" % (xb - xa))

print("\nTensorFlow float32 outputs:")
y_np = Y.numpy()
y_bits = tf.bitcast(Y, tf.uint32).numpy()
labels = ["xa", "xb"]
answers={}
for lbl, xv, yv, b in zip(labels, X.numpy().flatten(), y_np, y_bits):
    arg = int(np.argmax(yv))  # tie -> index 0
    print(f"{lbl}: x={xv:.9g} -> "
          f"y=[{yv[0]:.9g}, {yv[1]:.9g}]  "
          f"bits=[0x{b[0]:08x}, 0x{b[1]:08x}]  ArgMax_f32={arg}")
    answers[lbl]={"x": xv, "y": yv, "argmax": arg}



argmax_af=int(answers["xa"]["argmax"])
argmax_bf=int(answers["xb"]["argmax"])

assert argmax_af != argmax_bf
print("the model successfully demonstrates the argmax flip, i.e. lack of robustness")
    
def fraction_to_decimal(frac: Fraction, digits: int = 150) -> str:
    from decimal import Decimal, getcontext
    getcontext().prec = digits
    return str(Decimal(frac.numerator) / Decimal(frac.denominator))

y1a = answers["xa"]["y"][0]
y1b = answers["xb"]["y"][0]
xsy2a = answers["xa"]["y"][1]
y2b = answers["xb"]["y"][1]

assert y2b - y1b > 0
assert xb - xa > 0


print("\n")
print("Neural network in text format (for certifier). Place this in a .txt file")
print("[[%.150g,%.150g]]" % (w1, w2))
print("\n")



print("Give this to the certifier (output vector followed by radius): ")
print("%.150g,%.150g %.150g" % (y1b,y2b,(xb - xa)))


