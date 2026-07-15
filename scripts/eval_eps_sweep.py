# Evaluate gloro certified robustness of saved seed-sweep models across a grid of
# evaluation epsilons, to map the shape of the certified-radius distribution
# (hollow below eps_train, pile-up + cliff just above) and the cross-seed variance
# as a function of eval epsilon. Companion to seed_variance_sweep.sh; model rebuild
# follows get_gloro_robustness_results.py (build_model + load_and_set_weights +
# GloroNet + freeze_lc), evaluation follows the end of train_gloro.py (set
# g.epsilon, predict, gloro metric helpers).
#
# Usage:
#   eval_eps_sweep.py dataset INTERNAL_LAYER_SIZES input_size max_tries out_tsv \
#                     eps1,eps2,... seed_run_dir [seed_run_dir ...]
#
# Each seed_run_dir must contain model_weights_csv/. Rows are appended to out_tsv
# (tag_seed dir basename, eps, clean_acc, vra, robustness), flushed per row so an
# interrupted sweep keeps completed rows. Validation: the row at the training
# sweep's eval epsilon should reproduce that seed's vra in summary.tsv.
import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from tensorflow.keras import backend as K

import doitlib
from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate, vra, clean_acc

if len(sys.argv) < 8:
    print(f"Usage: {sys.argv[0]} dataset INTERNAL_LAYER_SIZES input_size max_tries out_tsv eps1,eps2,... seed_run_dir...")
    sys.exit(1)

dataset = sys.argv[1]
INTERNAL_LAYER_SIZES = eval(sys.argv[2])
input_size = int(sys.argv[3])
max_tries = int(sys.argv[4])
out_tsv = sys.argv[5]
eps_list = [float(e) for e in sys.argv[6].split(",")]
run_dirs = sys.argv[7:]

x_test, y_test = doitlib.load_test_data(dataset=dataset, input_size=input_size)
print(f"test data loaded: {len(x_test)} instances", flush=True)

write_header = not os.path.exists(out_tsv)
out = open(out_tsv, "a")
if write_header:
    out.write("run\teps\tclean_acc\tvra\trobustness\n")

for run_dir in run_dirs:
    run = os.path.basename(os.path.normpath(run_dir))
    csv_loc = os.path.join(run_dir, "model_weights_csv") + "/"
    if not os.path.isdir(csv_loc):
        print(f"SKIP {run}: no model_weights_csv", flush=True)
        continue
    inputs, outputs = doitlib.build_model(Input, Flatten, Dense, input_size=input_size,
                                          dataset=dataset, internal_layer_sizes=INTERNAL_LAYER_SIZES)
    model = Model(inputs, outputs)
    doitlib.load_and_set_weights(csv_loc, INTERNAL_LAYER_SIZES, model)
    g = GloroNet(model=model, epsilon=eps_list[0])
    g.freeze_lc(max_tries=max_tries)
    for eps in eps_list:
        g.epsilon = eps
        y_pred = g.predict(x_test, verbose=0)
        acc = float(clean_acc(y_test, y_pred).numpy())
        rej = float(rejection_rate(y_test, y_pred).numpy())
        the_vra = float(vra(y_test, y_pred).numpy())
        out.write(f"{run}\t{eps}\t{acc}\t{the_vra}\t{1.0 - rej}\n")
        out.flush()
        print(f"{run} eps={eps}: acc={acc:.4f} vra={the_vra:.4f} rob={1.0 - rej:.4f}", flush=True)
    K.clear_session()

out.close()
