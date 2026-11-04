#!/usr/bin/env python3
"""Empirical l2 (non-)robustness of a certifier-format dense model under PGD / APGD.

Purpose: partition the test set into points the real-arithmetic Lipschitz check certifies
robust, points an attack BREAKS, and neither. A point is broken iff a point within eps is
classified differently by the execution, the negation of the paper's robustness definition
(Definition 2.1), regardless of the true label; every test point is attacked, correctly
classified or not, against the label the compliant numpy execution predicts for it. The
"neither" share upper-bounds the looseness of the Lipschitz bound (plus attack weakness);
accuracy plays no role. (The 2026-09 POPL-response version attacked only the correctly
classified points against their true labels, framing the split against clean accuracy, which
conflates how robust the model was trained to be with how well the check certifies; that
framing was dropped on 2026-11-05 and is not reproducible from this script.)

Attack: ART's ProjectedGradientDescent (norm=2), run against the Keras float32
model (validated tooling, as for the DeepFool step of the counter-example
search). Every verdict is then RE-JUDGED under the IEEE-754-compliant numpy
execution (python-certifier/compliant_forward.py), which is the execution the
paper certifies: a point counts as attacked only if the numpy execution
misclassifies the adversarial point AND ||x_adv - x||_2 <= eps, the distance computed
EXACTLY (fractions.Fraction over the two float32 vectors, eps the exact decimal given on
the command line), as reverify_cex_suite.py judges the paper's counterexamples. ART is
given a radius a hair inside eps (eps * (1 - inset), --inset, default 1e-5) so that its
float32 projection lands inside the ball by construction; nothing is shrunk afterwards,
and an adversarial the exact check rejects is simply not counted (reported as
flips_outside_ball). The adversarials of every point whose label flipped are saved
(float32) next to the JSON for independent re-verification.

Usage:
  python pgd_l2_robust_accuracy.py dataset INTERNAL_LAYER_SIZES model_weights_csv_dir \
      input_size eps [--steps 100] [--restarts 5] [--batch 256] [--n N] [--out out.json]
      [--attack pgd|apgd] [--inset 1e-5]
Run under cav2025-artifact-venv.
"""
import json
import os
from fractions import Fraction
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

from art.attacks.evasion import ProjectedGradientDescent, AutoProjectedGradientDescent  # noqa: E402
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
eps_str = sys.argv[5]
eps = float(eps_str)
eps_q = Fraction(eps_str)  # the exact radius the verdict uses
inset = opt("--inset", 1e-5, float)
eps_att = eps * (1.0 - inset)  # the radius ART attacks at
steps = opt("--steps", 100, int)
restarts = opt("--restarts", 5, int)
batch = opt("--batch", 256, int)
n_limit = opt("--n", None, int)
attack_kind = opt("--attack", "pgd", str)  # pgd | apgd (APGD-CE then APGD-DLR on survivors, as in AutoAttack)
out_path = opt("--out", f"results/{attack_kind}_l2_{dataset}_eps{eps}_predlabel_exact.json", str)
adv_path = out_path[:-5] + "_adv.npz" if out_path.endswith(".json") else out_path + "_adv.npz"

inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size,
                                      dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
model = Model(inputs, outputs)
doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)
num_classes = int(model.output_shape[-1])
x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)
x_test = np.asarray(x_test, dtype=np.float32)  # the points are float32, as the deployed execution sees them
if n_limit is not None:
    x_test, y_test = x_test[:n_limit], y_test[:n_limit]
labels = np.argmax(y_test, axis=1)
N = len(x_test)
print(f"{dataset}: {N} test points, eps={eps_str} (exact), attack radius {eps_att!r} (inset {inset}), "
      f"PGD steps={steps} restarts={restarts}", flush=True)

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
att_labels_np, att_labels_keras = clean_np, clean_keras  # attack away from the predicted class
y_att = np.eye(num_classes, dtype=y_test.dtype)[clean_np]
eligible = np.ones(N, dtype=bool)  # every point, correctly classified or not
print(f"attacking all {N} points against the numpy execution's predicted labels", flush=True)

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    nb_classes=num_classes,
    input_shape=tuple(model.input_shape[1:]),
    clip_values=(0.0, 1.0),
)
if attack_kind == "pgd":
    attacks = [ProjectedGradientDescent(classifier, norm=2, eps=eps_att, eps_step=eps_att / 10.0,
                                        max_iter=steps, num_random_init=restarts,
                                        batch_size=batch, verbose=False)]
else:  # the two gradient-based components of AutoAttack (Croce & Hein 2020), run in sequence
    attacks = [AutoProjectedGradientDescent(classifier, norm=2, eps=eps_att, eps_step=2 * eps_att, max_iter=steps,
                                            nb_random_init=restarts, batch_size=batch, loss_type=lt, verbose=False)
               for lt in ("cross_entropy", "difference_logits_ratio")]


def generate(x, y):
    """Run the attack list in sequence, keeping each point's first successful (Keras-judged) adversarial."""
    x_adv = x.copy()
    todo = np.ones(len(x), dtype=bool)
    for atk in attacks:
        if not todo.any():
            break
        xa = atk.generate(x=x[todo], y=y[todo]).astype(np.float32)
        pred = np.argmax(model.predict(xa, batch_size=1024, verbose=0), axis=1)
        idx = np.where(todo)[0]
        succ = pred != np.argmax(y[todo], axis=1)
        x_adv[idx[succ]] = xa[succ]
        todo[idx[succ]] = False
    return x_adv

def sq_dist_exact(a, b):
    """Exact squared l2 distance of two float32 vectors (every float32 value is a rational)."""
    return sum((Fraction(float(u)) - Fraction(float(v))) ** 2 for u, v in zip(a.ravel().tolist(), b.ravel().tolist()))


records = []
adv_idx, adv_x = [], []
flips_outside = 0
attacked_np = np.zeros(N, dtype=bool)
attacked_keras = np.zeros(N, dtype=bool)
idx_all = np.arange(N)
for start in range(0, N, batch):
    sel = idx_all[start:start + batch]
    sel = sel[eligible[sel]]
    if len(sel) == 0:
        continue
    x = x_test[sel]
    x_adv = generate(x, y_att[sel])
    for k, i in enumerate(sel):
        xi = x[k]  # float32
        xa = np.asarray(x_adv[k], dtype=np.float32)
        d = float(np.linalg.norm((xa.astype(np.float64) - xi.astype(np.float64)).ravel()))  # advisory only
        p_np = np_pred(xa)
        p_keras = int(np.argmax(model.predict(xa[None], verbose=0)[0]))
        flip_np = p_np != att_labels_np[i]
        flip_keras = p_keras != att_labels_keras[i]
        ok = None
        if flip_np or flip_keras:  # the exact check is only needed where the label flipped
            ok = sq_dist_exact(xi, xa) <= eps_q * eps_q
            if flip_np and not ok:
                flips_outside += 1
            adv_idx.append(int(i))
            adv_x.append(xa.reshape(-1))
        attacked_np[i] = bool(flip_np and ok)
        attacked_keras[i] = bool(flip_keras and ok)
        records.append({"idx": int(i), "label": int(labels[i]), "attack_label_numpy": int(att_labels_np[i]), "dist": d,
                        "dist_le_eps_exact": ok, "adv_pred_numpy": p_np, "adv_pred_keras": p_keras,
                        "attacked_numpy": bool(attacked_np[i]), "attacked_keras": bool(attacked_keras[i])})
    done = start + batch
    print(f"  {min(done, N)}/{N}: attacked so far (numpy) {int(attacked_np.sum())} "
          f"of {int(eligible[:done].sum())} eligible ({time.time() - t0:.0f}s)", flush=True)

summary = {
    "dataset": dataset, "eps": eps_str, "attack_eps": eps_att, "inset": inset, "n": int(N), "attack": attack_kind,
    "label_mode": "predicted", "distance_check": "exact", "pgd_steps": steps, "pgd_restarts": restarts,
    "flips_numpy": int(sum(1 for r in records if r["adv_pred_numpy"] != r["attack_label_numpy"])),
    "flips_outside_ball": int(flips_outside),
    "n_eligible": int(eligible.sum()),
    "unbroken_rate_numpy": float((~attacked_np).mean()),  # fraction of ALL N not attacked (upper-bounds the robust rate in predicted mode)
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
np.savez_compressed(adv_path, idx=np.array(adv_idx, dtype=np.int64),
                    x_adv=np.stack(adv_x).astype(np.float32) if adv_x else np.zeros((0,), dtype=np.float32))
print(f"Wrote {out_path} and {adv_path} ({len(adv_idx)} adversarials)")
