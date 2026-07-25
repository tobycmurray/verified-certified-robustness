"""Stage 0 of the biased-model refutation experiment.

Aggressive L2 PGD/FGM attack on the adversarially-BIASED model, run over the
natural test set, judging each candidate adversarial under BOTH deployed fp32
executions (Keras/TF and IEEE-754-compliant numpy). Produces the raw data the
later stages consume:

  logits.json  -- for EVERY test point: the biased model's fp32 output logits
                  under Keras AND under numpy, plus the true label. Later stages
                  feed these (execution-specific) logits to the certifier, whose
                  margin comes from the stored logits, not a recompute.

  flips.json   -- for each point with a within-eps deployed flip: per execution
                  {flipped, xadv_file, l2, argmax_x, argmax_xadv}. Membership is
                  per execution (a candidate may flip Keras, numpy, both, or
                  neither); the smallest-L2 flipping candidate is kept per exec.

  x_<idx>_x0.npy / x_<idx>_xadv_{keras,numpy}.npy -- the input pairs.

Design notes:
  * PGD SEARCH is Keras-only (gradients live there); it merely PROPOSES
    candidates. The counter-example CRITERION is applied independently per
    execution: keep x_adv for exec E iff f_E(x_adv) != f_E(x).
  * only correctly-classified-under-E points are eligible for exec E (VRA).
  * B (sound norm bounds) is NOT computed here -- that needs the certifier's
    exact-rational arithmetic and is Stage 1's job. Here we gate candidates with
    a high-precision mpmath L2 <= epsilon (false-positive rejection); Stage 1 does
    the authoritative exact-Q B_l2 < eps filter.

Runs under the ART-equipped venv (cav2025-artifact-venv). The numpy execution is
python-certifier's compliant_forward (imported via PYTHON_CERTIFIER).

Usage:
  run_pgd_attack_biased.py DATASET LAYERS CSV_DIR INPUT_SIZE OUT_DIR
      [--epsilon 0.3] [--max-iter 500] [--restarts 10] [--subset N]
      [--fp-bias 1e6] [--fp-bias-pos end]
"""
import os
import sys
import json
import argparse

import numpy as np
import tensorflow as tf                       # eager (NO disable_eager_execution)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

import doitlib

from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier

from mpmath import mp, mpf, sqrt as mpsqrt
mp.dps = 50

# IEEE-754-compliant numpy forward pass lives in python-certifier.
PYTHON_CERTIFIER = os.environ.get(
    "PYTHON_CERTIFIER", os.path.expanduser("~/git/python-certifier"))
sys.path.insert(0, PYTHON_CERTIFIER)
import compliant_forward


# ---------------------------------------------------------------------------
# Biased model construction (position="end"): a flat bias B on the penultimate
# Dense layer, cancelled exactly in real arithmetic by -(B @ W_last) on the last
# layer. Same function as the unbiased model in exact arithmetic; only its fp32
# rounding differs. Lifted from attack_verified_certifier_nat_numpy.py.
# ---------------------------------------------------------------------------
def build_biased_model(original_model, B, position="end"):
    dense_layers = [l for l in original_model.layers
                    if isinstance(l, tf.keras.layers.Dense)]
    n_dense = len(dense_layers)

    inp = Input(tuple(original_model.input_shape[1:]))
    z = Flatten()(inp)
    for i, dl in enumerate(dense_layers):
        z = Dense(dl.units, use_bias=True,
                  activation=(None if i == n_dense - 1 else 'relu'))(z)
    new_model = Model(inp, z)
    new_dense = [l for l in new_model.layers
                 if isinstance(l, tf.keras.layers.Dense)]

    if position == "end":
        bias_layer, corr_layer = n_dense - 2, n_dense - 1
    elif position == "begin":
        bias_layer, corr_layer = 0, 1
    else:
        raise ValueError(f"bad position {position!r}")

    W_bias = dense_layers[bias_layer].get_weights()[0]
    b_bias = np.full(W_bias.shape[1], B, dtype=np.float32)
    new_dense[bias_layer].set_weights([W_bias, b_bias])

    W_corr = dense_layers[corr_layer].get_weights()[0]
    b_corr = -(b_bias.astype(np.float64) @ W_corr.astype(np.float64)).astype(np.float32)
    new_dense[corr_layer].set_weights([W_corr, b_corr])

    for i in range(n_dense):
        if i in (bias_layer, corr_layer):
            continue
        Wi = dense_layers[i].get_weights()[0]
        new_dense[i].set_weights([Wi, np.zeros(Wi.shape[1], dtype=np.float32)])

    new_model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    return new_model, n_dense


# ---------------------------------------------------------------------------
# The compliant numpy model: same (biased) weights as the Keras model, executed
# via compliant_forward (per-op round-to-nearest, no BLAS, no FTZ) at float32.
# Lifted from attack_verified_certifier_nat_numpy.py.
# ---------------------------------------------------------------------------
def build_numpy_model(keras_model, fmt="float32"):
    NP_DT = compliant_forward._DT[fmt]
    dense = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.Dense)]
    Ws, bs = [], []
    for l in dense:
        w = l.get_weights()
        Ws.append(np.ascontiguousarray(np.asarray(w[0]).T).astype(NP_DT))  # (out,in)
        bs.append(np.asarray(w[1]).astype(NP_DT) if len(w) > 1 else None)
    if all(b is None for b in bs):
        bs = None
    elif any(b is None for b in bs):
        raise ValueError("model mixes biased and unbiased Dense layers")
    return (Ws, bs, NP_DT)


def make_np_logits(np_model):
    def np_logits(x):
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        return np.asarray(compliant_forward.forward_logits(np_model, flat),
                          dtype=np.float64)
    return np_logits


def l2_mp(a, b):
    """high-precision L2 distance (false-positive gate; Stage 1 does exact-Q B)."""
    af = np.asarray(a, dtype=np.float64).ravel().tolist()
    bf = np.asarray(b, dtype=np.float64).ravel().tolist()
    return mpsqrt(sum((mpf(x) - mpf(y)) ** 2 for x, y in zip(af, bf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("layers", help="e.g. [128,128,128,128,128,128,128,128]")
    ap.add_argument("csv_dir", help="model_weights CSV directory")
    ap.add_argument("input_size", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--subset", type=int, default=0, help="first N test points (0=all)")
    ap.add_argument("--fp-bias", type=float, default=1e6)
    ap.add_argument("--fp-bias-pos", default="end", choices=["end", "begin"])
    args = ap.parse_args()

    layers = eval(args.layers)
    csv_loc = args.csv_dir.rstrip("/") + "/"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- build the (biased) model ---
    inputs, outputs = doitlib.build_model(
        Input, Flatten, Dense, input_size=args.input_size,
        dataset=args.dataset, internal_layer_sizes=layers)
    model = Model(inputs, outputs)
    doitlib.load_and_set_weights(csv_loc, layers, model)
    if args.fp_bias > 0:
        model, n_dense = build_biased_model(model, B=args.fp_bias, position=args.fp_bias_pos)
        print(f"Built biased model: B={args.fp_bias:.0e}, pos={args.fp_bias_pos}, {n_dense} Dense layers")

    x_test, y_test = doitlib.load_test_data(input_size=args.input_size, dataset=args.dataset)
    if args.subset > 0:
        x_test, y_test = x_test[:args.subset], y_test[:args.subset]
    n = x_test.shape[0]
    num_classes = int(y_test.shape[1])
    labels_true = np.argmax(y_test, axis=1)
    print(f"Attacking {n} points at L2 eps={args.epsilon}, "
          f"{args.restarts} restarts x {args.max_iter} iters")

    # --- numpy execution of the same biased weights ---
    np_model = build_numpy_model(model, "float32")
    np_logits = make_np_logits(np_model)
    print(f"numpy model: {len(np_model[0])} layers, biases={'yes' if np_model[1] is not None else 'no'}")

    # --- Keras (TF2-eager) classifier ---
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        nb_classes=num_classes,
        input_shape=tuple(model.input_shape[1:]),
        clip_values=(0.0, 1.0),
    )

    # --- clean logits under BOTH executions, for every point ---
    print("Computing clean logits (keras + numpy) for all points...")
    keras_clean = classifier.predict(x_test)                       # (n, K)
    keras_argmax = np.argmax(keras_clean, axis=1)
    numpy_clean = np.stack([np_logits(x_test[i]) for i in range(n)])  # (n, K)
    numpy_argmax = np.argmax(numpy_clean, axis=1)
    print(f"  clean accuracy: keras {np.mean(keras_argmax == labels_true)*100:.2f}%, "
          f"numpy {np.mean(numpy_argmax == labels_true)*100:.2f}%")

    # --- attack battery (targets for the targeted variants: 2nd-highest keras) ---
    targets = []
    for i in range(n):
        if keras_argmax[i] != labels_true[i]:
            targets.append(labels_true[i])
        else:
            order = np.argsort(keras_clean[i])[::-1]
            targets.append(int(order[1]))
    targets = np.array(targets)

    eps_atk = args.epsilon - 5e-8   # tiny shrink to avoid boundary false positives
    attacks = [
        FastGradientMethod(estimator=classifier, norm=2, eps=eps_atk, eps_step=0.01,
                           minimal=True, num_random_init=args.restarts, targeted=True),
        FastGradientMethod(estimator=classifier, norm=2, eps=eps_atk, eps_step=0.01,
                           minimal=True, num_random_init=args.restarts, targeted=False),
        ProjectedGradientDescent(classifier, norm=2, eps=eps_atk, eps_step=0.01,
                                 max_iter=args.max_iter, num_random_init=args.restarts,
                                 targeted=False, verbose=False),
        ProjectedGradientDescent(classifier, norm=2, eps=eps_atk, eps_step=0.01,
                                 max_iter=args.max_iter, num_random_init=args.restarts,
                                 targeted=True, verbose=False),
    ]

    adv_batches = []       # per attack: (n, ...) adversarial inputs
    keras_adv_argmax = []  # per attack: (n,) keras argmax at the adversarial
    for a in attacks:
        print(f"Running {type(a).__name__} (targeted={a.targeted})...")
        s = a.generate(x_test, targets) if a.targeted else a.generate(x_test)
        adv_batches.append(s)
        keras_adv_argmax.append(np.argmax(classifier.predict(s), axis=1))

    # --- per point, per execution: smallest-L2 within-eps candidate that flips E ---
    def numpy_adv_argmax(ai, i):
        return int(np.argmax(np_logits(adv_batches[ai][i])))

    flips = []
    for i in range(n):
        rec = {"index": int(i), "true_label": int(labels_true[i]),
               "keras": None, "numpy": None}
        for exec_name, clean_arg, flip_arg in (
                ("keras", keras_argmax, lambda ai: int(keras_adv_argmax[ai][i])),
                ("numpy", numpy_argmax, lambda ai: numpy_adv_argmax(ai, i))):
            if clean_arg[i] != labels_true[i]:
                continue  # only correctly-classified-under-E points are eligible
            best = None
            for ai in range(len(attacks)):
                adv_arg = flip_arg(ai)
                if adv_arg == clean_arg[i]:
                    continue                       # not a flip under E
                l2 = l2_mp(adv_batches[ai][i], x_test[i])
                if l2 > mpf(str(args.epsilon)):
                    continue                       # outside the ball
                if best is None or l2 < best[1]:
                    best = (ai, l2, adv_arg)
            if best is not None:
                ai, l2, adv_arg = best
                xadv_file = f"x_{i}_xadv_{exec_name}.npy"
                np.save(os.path.join(args.out_dir, xadv_file), adv_batches[ai][i])
                rec[exec_name] = {"flipped": True, "xadv_file": xadv_file,
                                  "l2_mp": str(l2), "argmax_x": int(clean_arg[i]),
                                  "argmax_xadv": int(adv_arg), "attack_idx": int(ai)}
        if rec["keras"] is not None or rec["numpy"] is not None:
            np.save(os.path.join(args.out_dir, f"x_{i}_x0.npy"), x_test[i])
            flips.append(rec)

    # --- write outputs ---
    with open(os.path.join(args.out_dir, "logits.json"), "w") as f:
        json.dump([{"index": int(i), "true_label": int(labels_true[i]),
                    "keras_logits": keras_clean[i].astype(float).tolist(),
                    "numpy_logits": numpy_clean[i].astype(float).tolist(),
                    "keras_argmax": int(keras_argmax[i]),
                    "numpy_argmax": int(numpy_argmax[i])} for i in range(n)], f)

    n_k = sum(1 for r in flips if r["keras"])
    n_n = sum(1 for r in flips if r["numpy"])
    meta = {"dataset": args.dataset, "layers": layers, "epsilon": args.epsilon,
            "max_iter": args.max_iter, "restarts": args.restarts, "subset": n,
            "fp_bias": args.fp_bias, "fp_bias_pos": args.fp_bias_pos,
            "num_classes": num_classes,
            "keras_flips": n_k, "numpy_flips": n_n}
    with open(os.path.join(args.out_dir, "flips.json"), "w") as f:
        json.dump({"meta": meta, "flips": flips}, f, indent=2)

    print(f"\nDone. within-eps deployed flips: keras {n_k}/{n}, numpy {n_n}/{n}")
    print(f"Wrote logits.json ({n} points), flips.json ({len(flips)} flip points) to {args.out_dir}/")


if __name__ == "__main__":
    main()
