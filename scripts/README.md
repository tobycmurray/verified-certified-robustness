Scripts to empirically evaluate the robustness certifier.

The top-level script is the shell script: `doit_verified_robust_gloro.sh`.

That script:
1. Trains an Gloro MNIST model (using `train_gloro.py`)
2. Manually constructs a "zero bias" version of the model and uses that to produce output vectors for all MNIST test points (using `zero_bias_saved_model.py`)
3. Saves the resulting "zero bias" model's weights in a format the certifier can understand (using `make_certifier_format.py`)
4. Runs the certifier over the output vectors produced by step 2, using the weights produced by step 3
5. Measures the resulting "zero bias" model's (using `test_verified_certified_robust_accuracy.py`):
  * accuracy: the proportion of MNIST test points correctly classified
  * robustness: the proportion of the MNIST test points that the cerfifier says are robust
  * verified robust accuracy (VRA): the proportion of MNIST test points that are both accurately classified and certified robust

The top-level script takes arguments to specify parameters to use to define the model architecture and how to train it, as well as the evaluation epsilon value used by the certifier when certifying robustness.

There is also a script for running adversarial attacks on a model to estimate robustness upper bounds:
`run_pgd_attack.py`.

# Reproducing Paper Plots and Tables

To reproduce the results in the paper:

## Fig 9 (the performance plots)

To generate the data for 7x7 run

```
./doit_verified_robust_gloro.sh 0.3 "[16]" 0.3 ../main n 500 256 7
```
for each of `n` in 1 to 10.

For each run, this will produce output that ends in something like the following (notwithstanding the different parameter values):

```
Proportion robust: 0.324
Proportion correct: 0.728
Proportion robust and correct: 0.3124

Certifier started at:         2024-12-21 12:05:46
Certifier produced bounds at: 2024-12-21 12:05:48
(The difference between these quantities is therefore the time taken to compute those bounds.)
Certifier finished at: 2024-12-21 12:07:00

All done.
Artifacts and results all saved in: 2024-12-21_12:05:32/
Parameters saved in: 2024-12-21_12:05:32/params.txt, whose contents follows:
    (Global) Model input size: 7x7
    (Training) Gloro epsilon: 0.3
    (Training) INTERNAL_LAYER_SIZES: [16]
    (Training) Epochs: 3
    (Training) Batch size: 32
    (Certifier) Eval epsilon: 0.3
    (Certifier) GRAM_ITERATIONS: 4

Model weights saved in: 2024-12-21_12:05:32/model_weights_epsilon_0.3_[16]_3
Model outputs saved in: 2024-12-21_12:05:32/all_mnist_outputs_epsilon_0.3_[16]_3.txt
Neural network (for certifier) saved in: 2024-12-21_12:05:32/neural_net_mnist_epsilon_0.3_[16]_3.txt
Model outputs for evaluation saved in: 2024-12-21_12:05:32/all_mnist_outputs_epsilon_0.3_[16]_3_eval_0.3.txt
Certified robustness results saved in: 2024-12-21_12:05:32/results_epsilon_0.3_[16]_3_eval_0.3_gram_4.json
Timestamped certifier output saved in: 2024-12-21_12:05:32/results_epsilon_0.3_[16]_3_eval_0.3_gram_4.json.timestamps
```

The "Proportion robust" is the "Certified Robustness" proportion. 

The time to compute the bounds is the difference between the first two reported timestamps (per the output). That
difference can be calculated by pasting the two values into Excel and having it compute the subtraction (formatting the result
cell with "Custom" format "hh:mm:ss"). If the difference exceeds 23:59:59 (1 day) then Excel doesn't compute it correctly.
In that case, calculate it manually. 

Write down these certified robustness proportions and times. Manually convert the times into quantities in seconds. Then update
the values in `robustness_vs_gram_input_7_[16].csv`.

To produce the PDF file for the 7x7 plot, then run:
```
python3 plot_robustness_vs_gram.py "Certifier Performance (7x7 MNIST)" results/robustness_vs_gram_input_7_\[16\].csv results/measured_robustness_input_7_\[16\].txt performance_7.pdf
```
This saves the PDF plot in `performance_7.pdf` which can then be copied over to the paper directory.

For the 14x14 plot, the process is the same except starts by running:

```
./doit_verified_robust_gloro.sh 0.3 "[64]" 0.3 ../main n 500 256 14
```

The updated certified robustness proportions and times should then be saved to `results/robustness_vs_gram_input_14_[64].csv`.
To produce the 14x14 plot, run:
```
python3 plot_robustness_vs_gram.py "Certifier Performance (14x14 MNIST)" results/robustness_vs_gram_input_14_\[64\].csv results/measured_robustness_input_14_\[64\].txt performance_14.pdf
```
which produces `performance_14.pdf`.

(See further below for how to caluclate the Measured Robustness figures for these plots. However that probabaly isn't necessary
unless we are training models with new paramters, i.e. updates to the certifier won't meaningfully change the Measured Robustness
numbers because they have nothing to do with the certifier.)


## Table 1 (the performance table)

The data for each row of this table can be reproduced by running the `doit_verified_robust_gloro.sh` script with the corresponding parameters.
For example, the final row can be reproduced by running:

```
./doit_verified_robust_gloro.sh 0.45 "[64]" 0.3 ../main 6 500 256 28
```

The first parameter is the training epsilon value. The second is a list where each element represents a hidden layer and the number of
neurons in it. Here we use a single hidden layer (single-element list) of 64 neurons. The next parameter is the evaluation epsilon value:
i.e. the value of epsilon that the certifier is certifying outputs as robust against. The path to the certifier comes next.
The number of gram iterations (here 6) comes next, followed by the training number of epochs and batch sizes. Leave these as 500 and 256.
The final parameter is the size of the input for the MNIST model. For these examples we use full MNIST models (28x28 size input).

This will produce data to allow calculating figures for each row of the table. The "Accuracy" column is the
"Proportion correct". The "Certified Robustness" column is the "Proportion robust". "VRA" is the "Proportion robust and correct".


## Calculating Measured Robustness

I don't think there is a need to re-calculate these numbers. They do not depend on the certifier in any way. 

However, these figures can be calculated after model training (i.e. after running the `doit_verified_robust_gloro.sh` script).

For example, to calculate the figures for the 7x7 MNIST model run:
```
python3 run_pgd_attack.py [16] model_weights_csv_dir 0.3 500 7
```
where `model_weights_csv_dir` is the directory in which the `doit` script reports that model weights were saved in, e.g. in the
output above we see:
```
Model weights saved in: 2024-12-21_12:05:32/model_weights_epsilon_0.3_[16]_3
```
which means this directory is `"2024-12-21_12:05:32/model_weights_epsilon_0.3_[16]_3"`.

Note: you may need to surround these directory names with double-quotes "" because they contain special characters like '[' etc.

Here the 500 refers to the "strength" of the attacks run. In all examples in the paper I set this to 500. 7 refers to the model's input
size (here 7x7).

This script produces output that finishes with e.g.:
```
Model accuracy: 72.8
Robustness on PGD adversarial samples: 53.28%
Norms of non-false-positive vectors that cause classification changes: min: 0.00599679290863438; max: 0.289841752514313
False positives in PGD attack: 0
Number of PGD attacks succeeding against certified robust outputs: 0
```

The model accuracy should match the figure reported earlier as the "Proportion correct" by the `doit..sh` script.

The "Robustness on PDG adversarial samples" is the "Measured Robustness" figure.

Do similarly for the 14x14 MNIST model, replacing the 7 above with 14.

Then save the resulting figures to `measured_robustness_input_7_[16].txt` (for the 7x7 MNIST model)
and `measured_robustness_input_14_[64].txt` (for the 14x14 MNIST model) and rerun the python scripts to generate the plots.

Do likewise to calculate Measured Robustness for each row of Table 1. 
