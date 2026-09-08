#!/usr/bin/env python3
"""Empirical l2 robust accuracy of a certifier-format dense model under PGD.

Purpose (POPL'27 reviewer B, Q3): split the gap between clean accuracy and the
real-arithmetic verified robust accuracy into (a) points that are demonstrably
NOT robust (an attack finds a misclassified point within eps) and (b) points
where certification is loose. PGD robust accuracy upper-bounds the true robust
accuracy, so
    (PGD robust acc) - (real VRA)        upper-bounds the certification looseness,
    (clean acc)      - (PGD robust acc)  lower-bounds the genuinely non-robust share.

Attack: ART's ProjectedGradientDescent (norm=2), run against the Keras float32
model (validated tooling, as for the DeepFool step of the counter-example
search). Every verdict is then RE-JUDGED under the IEEE-754-compliant numpy
execution (python-certifier/compliant_forward.py), which is the execution the
paper certifies: a point counts as attacked only if the numpy execution
misclassifies the adversarial point AND ||x_adv - x||_2 <= eps (float64). If ART's
float32 projection lands a hair outside the ball, the perturbation is shrunk
back onto it before re-judging.

Usage:
  python pgd_l2_robust_accuracy.py dataset INTERNAL_LAYER_SIZES model_weights_csv_dir \
      input_size eps [--steps 100] [--restarts 5] [--batch 256] [--n N] [--out out.json]
Run under cav2025-artifact-venv.
"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

import doitlib

PYTHON_CERTIFIER = os.path.expanduser(os.environ.get("PYTHON_CERTIFIER", "~/git/python-certifier"))
sys.path.insert(0, PYTHON_CERTIFIER)
import compliant_forward  # noqa: E402

from art.attacks.evasion import ProjectedGradientDescent  # noqa: E402
from art.estimators.classification import TensorFlowV2Classifier  # noqa: E402

if len(sys.argv) < 6:
    print(__doc__)
    sys.exit(1)


def opt(flag, default, cast):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


dataset = sys.argv[1]
INTERNAL_LAYER_SIZES = eval(sys.argv[2])
csv_loc = sys.argv[3].rstrip("/") + "/"
input_size = int(sys.argv[4])
eps = float(sys.argv[5])
steps = opt("--steps", 100, int)
restarts = opt("--restarts", 5, int)
batch = opt("--batch", 256, int)
n_limit = opt("--n", None, int)
out_path = opt("--out", f"results/pgd_l2_{dataset}_eps{eps}.json", str)

inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size,
                                      dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)
doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)
num_classes = int(model.output_shape[-1])
x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)
if n_limit is not None:
    x_test, y_test = x_test[:n_limit], y_test[:n_limit]
labels = np.argmax(y_test, axis=1)
N = len(x_test)
print(f"{dataset}: {N} test points, eps={eps}, PGD steps={steps} restarts={restarts}", flush=True)

# ---- compliant numpy execution (same construction as attack_verified_certifier_nat_numpy) ----
NP_DT = np.float32
dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
Ws = [np.ascontiguousarray(np.asarray(l.get_weights()[0]).T).astype(NP_DT) for l in dense]
bs = [np.asarray(l.get_weights()[1]).astype(NP_DT) if len(l.get_weights()) > 1 else None for l in dense]
bs = None if all(b is None for b in bs) else bs
NP_MODEL = (Ws, bs, NP_DT)


def np_pred(x):
    flat = np.asarray(x, dtype=np.float64).reshape(-1)
    return int(np.argmax(np.asarray(compliant_forward.forward_logits(NP_MODEL, flat), dtype=np.float64)))


t0 = time.time()
clean_np = np.array([np_pred(x) for x in x_test])
clean_keras = np.argmax(model.predict(x_test, batch_size=1024, verbose=0), axis=1)
correct_np = clean_np == labels
print(f"clean acc: numpy {correct_np.mean():.4f}, keras {(clean_keras == labels).mean():.4f} "
      f"({time.time() - t0:.0f}s)", flush=True)

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    nb_classes=num_classes,
    input_shape=tuple(model.input_shape[1:]),
    clip_values=(0.0, 1.0),
)
attack = ProjectedGradientDescent(classifier, norm=2, eps=eps, eps_step=eps / 10.0,
                                  max_iter=steps, num_random_init=restarts,
                                  batch_size=batch, verbose=False)

records = []
attacked_np = np.zeros(N, dtype=bool)
attacked_keras = np.zeros(N, dtype=bool)
idx_all = np.arange(N)
for start in range(0, N, batch):
    sel = idx_all[start:start + batch]
    sel = sel[correct_np[sel]]  # only attack points the (numpy) model gets right
    if len(sel) == 0:
        continue
    x = x_test[sel]
    x_adv = attack.generate(x=x, y=y_test[sel]).astype(np.float32)
    for k, i in enumerate(sel):
        xi = x[k].astype(np.float64)
        xa = x_adv[k].astype(np.float64)
        d = float(np.linalg.norm((xa - xi).ravel()))
        if d > eps:  # float32 projection overshoot: shrink back onto the ball
            xa = xi + (xa - xi) * (eps / d) * (1 - 1e-7)
            xa = np.clip(xa, 0.0, 1.0).astype(np.float32).astype(np.float64)
            d = float(np.linalg.norm((xa - xi).ravel()))
        p_np = np_pred(xa)
        p_keras = int(np.argmax(model.predict(xa[None].astype(np.float32), verbose=0)[0]))
        ok = d <= eps
        attacked_np[i] = ok and p_np != labels[i]
        attacked_keras[i] = ok and p_keras != labels[i]
        records.append({"idx": int(i), "label": int(labels[i]), "dist": d,
                        "adv_pred_numpy": p_np, "adv_pred_keras": p_keras,
                        "attacked_numpy": bool(attacked_np[i]), "attacked_keras": bool(attacked_keras[i])})
    done = start + batch
    print(f"  {min(done, N)}/{N}: attacked so far (numpy) {int(attacked_np.sum())} "
          f"of {int(correct_np[:done].sum())} correct ({time.time() - t0:.0f}s)", flush=True)

summary = {
    "dataset": dataset, "eps": eps, "n": int(N), "pgd_steps": steps, "pgd_restarts": restarts,
    "clean_acc_numpy": float(correct_np.mean()),
    "clean_acc_keras": float((clean_keras == labels).mean()),
    "pgd_robust_acc_numpy": float((correct_np & ~attacked_np).mean()),
    "pgd_robust_acc_keras": float(((clean_keras == labels) & ~attacked_keras).mean()),
    "attacked_numpy": int(attacked_np.sum()),
    "attacked_keras": int(attacked_keras.sum()),
    "n_correct_numpy": int(correct_np.sum()),
    "seconds": time.time() - t0,
}
print("\nSUMMARY:", json.dumps(summary, indent=1), flush=True)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"summary": summary, "records": records}, f)
print(f"Wrote {out_path}")
