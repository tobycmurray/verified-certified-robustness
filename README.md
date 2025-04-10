# A Formally Verified Robustness Certifier for Neural Networks

## CAV 2025 Artifact

The artifact is packaged as a Docker container, which is known to work
on x86-64 hosts. (Unfortunately, virtualisation failures seem to prevent
the container working on ARM-based hosts like Apple M1 etc.)

### Artifact Resource Requirements

An x86-64 based machine is required, with Docker. Experiments and
results replicated below were carried out on machines with 16GB RAM.


### Artifact Set Up

Load and enter the Docker container:

```
gunzip cav2025-artifact.tar.gz
docker image load < cav2025-artifact.tar
docker run -it cav2025-artifact:latest bash

```

Enter the Python virtual environment used for Python scripts that train
and use the ML models used in the paper's evaluation:
```
source cav2025-artifact/bin/activate
```

## Artifact Structure

The artifact has the following directory structure, and is located in the
`/workspace` directory of the container's filesystem:
* `robustness-verifier/` - the verified robustness certifier
* `cav2025-artifact/` - the artifact's Python virtual environment directory
* `cav2025-models/` - the three models used in the paper's evaluation
* `gloro/` - Leino et al.'s globally robust neural networks implementation

## Claims / Results supported by this Artifact

1. Our certifier is formally verified againt the specifications as described in the paper
2. The performance results of our certifier on the ML models in the paper (Table 1)
3. The performance results of our certifier on the ML models in the paper (Figure 8)


## Claim 1: Verifying and Building the Certifier

### Verifying the Certifier (will take up to 1--2 minutes)

To get Dafny to verify the certifier, run these commands:
```
cd robustness-verifier
./verify_all.sh
```

This should produce output like the following, showing that each file of the
certifier's Dafny implementation was verified:

```
dafny verify basic_arithmetic.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 52 verified, 0 errors
dafny verify linear_algebra.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 96 verified, 0 errors
dafny verify main.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 2 verified, 0 errors
dafny verify neural_networks.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 10 verified, 0 errors
dafny verify parsing.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 31 verified, 0 errors
dafny verify robustness_certification.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 42 verified, 0 errors
dafny verify l2_extra.dfy --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 4 verified, 0 errors
dafny verify operator_norms.dfy --isolate-assertions --solver-path "/opt/z3-4.13.4-x64-glibc-2.35/bin/z3"

Dafny program verifier finished with 680 verified, 0 errors
```

### The Top-Level Specification Being Verified

The claim that the formal verification establishes is that our certifier says that an
output is robust only if it is robust (i.e. soundness). This is formally
specifid in the top-level Dafny file for our certifire `main.dfy` on lines 106--125.
Specifically, note the following in this section of the source code of `main.dfy`:
```
      var robust: bool := CertifyMargin(outputVector, errorMargin, lipBounds);
      /* Verification guarantees that 'true' is only printed when for all input
      vectors v where applying the neural network to v results in the given
      output vector, this input-output pair of vectors is robust with respect
      to the given error margin. */
      assert robust ==> forall v: Vector |
        IsInput(v, neuralNet) && NN(neuralNet, v) == outputVector ::
        Robust(v, outputVector, errorMargin, neuralNet);
```
Here, the boolean variable `robust` tracks whether the certifier believes the
`outputVector` is robust for perturbation bound `errorMargin`. The subsequent
`assert` statement (which is verified by Dafny when verifying `main.dfy`)
ensures that whenever `robust` is true (the certifier believes the output
is robust), the output is actually robust for perturbation bound `errorMargin`.

The definition of `Robust` is in `robustness_certification.dfy` (line 23),
and is described in the paper in Section 3.4.

The definition of `IsInput` is in `neural_networks.dfy` (line 24), and is
described in the paper in Section 3.1.

The definition of `NN` is in `neural_networks.dfy` (line 64), and is
described in the paper in Section 3.2, where it is called "ApplyNN".


### Building the Certifier (will take up to about a minute)

To build the certifier binary, in the `robustness-verifier/` directory, run:
```
dafny build --unicode-char:false --target:cs main.dfy IO/FileIO.cs
```

This should produce the executable `main`, which is the main certifier binary.

If you run the certifier binary, it will produce basic usage information:
```
Usage: main <neural_network_input.txt> <GRAM_ITERATIONS>
```

Its first argument is the name of a text file containing the neural network weights
in a custom, textual format. The second argument is the number of gram iterations
to run, when computing Lipschitz bounds. Higher numbers of gram iterations causes
the certifier to run for longer, but produce tighter (and thus less conservative)
robustness certifications. For neural networks of the order evaluated in the paper,
12 gram iterations produces sufficiently tight Lipschitz bounds.

The certifier produces output in JSON format (simply to aid its automated evaluation
for this research). This output includes lots of debugging messages.

It first computes Lipschitz bounds before then accepting input vectors on `stdin`.
Its output certifies each input vector against a given perturbation radius.

The certifier's input format is explained later, below.



## Claim 2: Reproducing the Paper's Performance Evaluation (Table 1)

The main results of the performance evaluation appear in Table 1 of the paper,
where we apply our certifier to three ML models: MNIST, Fashion MNIST, and CIFAR-10.

Naturally, depending on the performance of your machine, the time taken
to reproduce these results could be longer than the times indicated below.
This also means, of course, that the results you obtain in terms of
certifier running time could be significantly different to those reported
in our paper or below.

Figure 8 (the performance plot) also graphically depicts performance results
on the MNIST model. See below for steps to reproduce that (Claim 3).

The `scripts/` subdirectory of `robustness-verifier/` contains scripts that can
be used to reproduce the evaluation results.

Before running these scripts, you need to have built the certifier (see above).

### ML Models and Result Files

These scripts produce a lot of data files. The `cav2025-models/` directory
contains copies of the three ML models used in the paper's evaluation, as well
as the data files produced from that evaluation.

The models appear in Table 1 of the paper and their corresponding locations
in the `cav2025-models/` directory are:

* MNIST (row 1): `2025-01-25_09:27:46-mnist/`
* Fashion MNIST (row 2): `2025-01-30_10:58:01-fashion_mnist/`
* CIFAR-10 (row 3): `2025-01-28_20:39:32-cifar10/`

### Main Evaluation Script

The main evaluation script in the `robustness-verifier/scripts/` directory is called
`doit_verified_robust_gloro.sh`. To reproduce the results in Table 1 this script can
be run.

It takes many arguments (that specify the location of the ML model to evaluate,
ML model internal architecture, training and evaluation parameters, etc). These
include the value of `epsilon` used to evaluate the robustness of the model,
and the number of gram iterations to run the certifier for, as shown in Table 1 for
each model.


### MNIST (will take about 20--40 minutes)

To reproduce the MNIST results, from the `robustness-verifier/scripts/` directory, run
this command (making sure you have first done `source cav2025-artifact/bin/activate`
to enter the Python virtual environment):

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-25_09:27:46-mnist"/ mnist 0.45  "[128,128,128,128,128,128,128,128]" 0.3 ../main 11 500 32

```

The first parameter is the path to the directory containing the MNIST model.
0.45 is the value of the training epsilon that was used when training this model, using
the `gloro` framework of Leino et al.  The "[128,...,128]" part specifies the internal
model architecture (8 layers, each of 128 neurons), corresponding to the "Hidden Neurons"
column of Table 1 in the paper. 0.3 is the evaluation `epsilon` value, corresponding to the
"epsilon" column of Table 1 in the paper. `../main` is the relative path to the certifier
binary. 11 is the number of gram iterations to run the certifier for (corresponding to the
"Gram" column of Table 1 in the paper). 500 and 32 are respectively the number of epochs
and batch size used when originally training this model using Leino et al.'s `gloro`
framework.

This script will ask if you wish to proceed. Enter "y" to do so. It may then
produce the following error message:
```
Certifier JSON results file ../../cav2025-models/2025-01-25_09:27:46-mnist//results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json already exists!
```
This is because the artifact already contains the result files for the experiments reported
in the paper.

Remove this JSON file and re-run the above command, e.g.:

```
rm ../../cav2025-models/2025-01-25_09:27:46-mnist//results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-25_09:27:46-mnist"/ mnist 0.45  "[128,128,128,128,128,128,128,128]" 0.3 ../main 11 500 32
```

This command will produce a lot of output, beginning something like this:
```
We will evaluate an existing mnist model in ../../cav2025-models/2025-01-25_09:27:46-mnist/.
Proceed? (yes/no): y
Proceeding...


Artifacts and results will live in: ../../cav2025-models/2025-01-25_09:27:46-mnist//

Running with these parameters (saved in ../../cav2025-models/2025-01-25_09:27:46-mnist//params.txt):
    Data set: mnist
    (Global) Model input size: 28x28
    (Training) Gloro epsilon: 0.45
    (Training) INTERNAL_LAYER_SIZES: [128,128,128,128,128,128,128,128]
    (Training) Epochs: 500
    (Training) Batch size: 32
    (Certifier) Eval epsilon: 0.3
    (Certifier) GRAM_ITERATIONS: 11


Running the certifier. This may take a while...
2025-04-09 08:11:11 [
2025-04-09 08:11:12 { "debug_msg": "Generating spectral norm 0 of 9..." },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 0 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 1 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 2 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 3 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 4 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 5 of : 784" },
2025-04-09 08:11:13 { "debug_msg": "MTM outer loop, i: 6 of : 784" },
...
```

It should take approximately 20--40 minutes to complete, at least. The final output
it produces contains various numbers that look something like the following:
```
...
...done 28000 of 29924 evaluations...

...done 29000 of 29924 evaluations...

Proportion robust: 0.9574
Proportion correct: 0.984
Proportion robust and correct: 0.954

Unverified model statistics (to compare to the above verified ones):
{
    "comment": "these statistics are unverified and calculated by the gloro implementation",
    "accuracy": 0.984000027179718,
    "rejection_rate": 0.042399998754262924,
    "robustness": 0.9576000012457371,
    "vra": 0.9541000127792358
}
Certifier started at:         2025-04-09 08:11:11
Certifier produced bounds at: 2025-04-09 08:27:10
(The difference between these quantities is therefore the time taken to compute those bounds.)
Certifier finished at: 2025-04-09 08:29:13

All done.
Artifacts and results all saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//
Parameters saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//params.txt, whose contents follows:
    Data set: mnist
    (Global) Model input size: 28x28
    (Training) Gloro epsilon: 0.45
    (Training) INTERNAL_LAYER_SIZES: [128,128,128,128,128,128,128,128]
    (Training) Epochs: 500
    (Training) Batch size: 32
    (Certifier) Eval epsilon: 0.3
    (Certifier) GRAM_ITERATIONS: 11

Model weights and (unverified) gloro lipschitz constants saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500
(Unverified) gloro model statistics saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500/gloro_model_stats.json
Model outputs saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//all_mnist_outputs_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt
Neural network (for certifier) saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//neural_net_mnist_epsilon_0.45_[128,128,128,128,128,128,128,128]_500.txt
Model outputs for evaluation saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//all_mnist_outputs_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3.txt
Certified robustness results saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json
Timestamped certifier output saved in: ../../cav2025-models/2025-01-25_09:27:46-mnist//results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json.timestamps
```

### Matching the Output to the Results in Table 1

The following columns of Table 1 are evident in this output, as follows:

* "Time (hh:mm:ss)" records the time taken for the certifier to compute the Lipschitz bounds. From the output above, we see that the certifier was at `2025-04-09 08:11:11`
and produced the Lipschitz bounds at `2025-04-09 08:27:10`, i.e. 15 minutes and 59 seconds later. This is slightly quicker than the figure reported in row 1 of Table 1 of the
paper (17 minutes and 2 seconds).
* "Verified Robustness" records the percentage of the test points that were certified robust. We see in the output above "Proportion robust: 0.9574" which corresponds exactly
to the percentage reported in row 1 of Table 1 (95.74%). We also see that the gloro unverified certifier of Leino et al. certifies the proportion 0.9576000012457371, i.e.
95.76%, i.e. 0.02 percentage points above the percentage certified by our verified certifier, as indicated by the "(-0.02)" in the "Verified Robustness" column of row 1 of
Table 1.
* "VRA" records the percentage of the test points that were both classified accurately by the trained model and certified robust by our certifier. We see that the
"Proportion robust and correct: 0.954" in the output above, i.e., a VRA of 95.40% as shown in this column. Similarly to the percentage certified robust, the gloro unverified
certifier of Leino et al. produces a VRA of 0.9541000127792358, i.e 95.41%, which is 0.01 percentage points above our certifier, as indicated by the "(-0.01)" in this column
of row 1 of Table 1.

## Fashion MNIST (will take about 20--50 minutes)

The results in the second row of Table 1 (for Fashion MNIST) can be reproduced by following the same procedure as for MNIST above. However, running the script as follows:

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-30_10:58:01-fashion_mnist"/ fashion_mnist 0.26 "[256,128,128,128,128,128,128,128,128,128,128,128]" 0.25 ../main 12 500 64

```

As indicated in Table 1, it may take around 20 minutes for the script to complete. On other machines, this is known to take as long as 50 minutes to complete. It produces output akin to that for MNIST, above.

## CIFAR-10 (will take about 20--40 hours)

The results for CIFAR-10 (third row of Table 1) can be reproduced by running the script as follows:

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-28_20:39:32-cifar10"/ cifar10  0.1551 "[512,256,128,128,128,128,128,128]" 0.141 ../main 12 800 256
```

The output can be matched to the results in row 3 of Table 1 by following the same procedure as for MNIST above.


## Claim 3: Reproducing the Paper's Performance Evaluation (Figure 8) 

The performance plot (Figure 8) shows our certifier's performance on the above
MNIST model for different numbers of gram iterations.

### Unverified Robustness

The "Unverified Robustness" number (95.76%) in this figure is the
percentage of test points that the unverified gloro certifier of
Leino et al. certifies as robust, for perturbation bound (epsilon) of 0.3.
This is recorded in the result file in the artifact, at the time the
model was trained:
`cav2025-models/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500/gloro_model_results.json`.

### Measured Robustness (this will take 30--80 minutes)

The "Measured Robustness" number was calculated by running adversarial
attacks on this MNIST model, which are implemented by the
`run_pgd_attack.py` script in the `robustness-verifier/scripts`
directory.

First make sure you have run `source cav2025-artifact/bin/activate` to
enter the Python virtual environment.

Then install necessary packages needed for this script:

```
pip install mpmath adversarial-robustness-toolbox
```


Then, from the `robustness-verifier/scripts/` directory,
run the script as follows:

```
python3 run_pgd_attack.py  mnist "[128,128,128,128,128,128,128,128]" "../../cav2025-models/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"/ 0.3 500 28
```

It will take about half an hour and produce output that finishs with lines like this:

```
Running attack FastGradientMethod(norm=2, eps=0.29999995, eps_step=0.01, targeted=True, num_random_init=1, batch_size=32, minimal=True, summary_writer=None, )...
Running attack FastGradientMethod(norm=2, eps=0.29999995, eps_step=0.01, targeted=False, num_random_init=1, batch_size=32, minimal=True, summary_writer=None, )...
Running attack MomentumIterativeMethod(norm=2, eps=0.29999995, eps_step=0.01, decay=1.0, targeted=False, num_random_init=0, batch_size=32, max_iter=500, random_eps=False, summary_writer=None, verbose=False, )...
Running attack MomentumIterativeMethod(norm=2, eps=0.29999995, eps_step=0.01, decay=1.0, targeted=True, num_random_init=0, batch_size=32, max_iter=500, random_eps=False, summary_writer=None, verbose=False, )...
Running attack ProjectedGradientDescent(norm=2, eps=0.29999995, eps_step=0.01, decay=None, targeted=False, num_random_init=1, batch_size=32, max_iter=500, random_eps=False, summary_writer=None, verbose=False, )...
Running attack ProjectedGradientDescent(norm=2, eps=0.29999995, eps_step=0.01, decay=None, targeted=True, num_random_init=1, batch_size=32, max_iter=500, random_eps=False, summary_writer=None, verbose=False, )...
We have 0 gloro robustness results to compare against
Model accuracy: 98.4
Robustness on PGD adversarial samples: 97.46%
Norms of non-false-positive vectors that cause classification changes: min: 0.00740313376351122; max: 0.299999999534099
False positives in PGD attack: 5
Norms of false positive vectors: min: 0.300000002028088; max: 0.300000015495363
Number of PGD attacks succeeding against certified robust outputs: 0
Number of PGD attacks succeeding against gloro certified robust outputs: 0
```

The relevant line here is: "Robustness on PGD adversarial samples: 97.46%", which is the value shown in Figure 8.
Note that since this value is obtained from an attack performed via SGD style optimisation, you may get a different
value when you run this script but hopefully it will be close to the reported value.

### Verified Robustness and Certifier Running Time (this will take up to a few hours to complete in full)

These numbers were obtained by repeatedly running the main evaluation script
```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-25_09:27:46-mnist"/ mnist 0.45  "[128,128,128,128,128,128,128,128]" 0.3 ../main <n> 500 32
```

on the MNIST model to find the percentage of points our certifier classifies as robust
and the time taken to compute the Lipschitz bounds (see the MNIST section for reproducing Table 1 results above)
for different values of `<n>` in the line above (which represents the number of gram iterations).

The plot in Figure 8 was produced by running the above command 10 times, replacing `<n>` above with the values 1 through 10,
and recording for each the proportion certified as robust by our verified certifier, and the difference between
the times at which the certifier was started and produced the Lipschitz bounds.

Recall the output of the main evaluation script includes stuff like this:

```
Proportion robust: 0.9574
Proportion correct: 0.984
Proportion robust and correct: 0.954

Unverified model statistics (to compare to the above verified ones):
{
    "comment": "these statistics are unverified and calculated by the gloro implementation",
    "accuracy": 0.984000027179718,
    "rejection_rate": 0.042399998754262924,
    "robustness": 0.9576000012457371,
    "vra": 0.9541000127792358
}
Certifier started at:         2025-04-09 08:11:11
Certifier produced bounds at: 2025-04-09 08:27:10
```

Here we see the proportion certified robust is 95.74% (this is the "Verified Robustness" measure in Figure 8)
and the time taken to compute the Lipschitz bounds was 15 minutes 59 seconds (this is the
"Certifier Running Time" in Figure 8).

# Other Documentation

## Certifier Input Format

The certifier takes arguments according to its usage information:
```
Usage: main <neural_network_input.txt> <GRAM_ITERATIONS>
```

The program expects a neural network schema provided in the text file name
<neural_network_input.txt>. This file must strictly contain only the following
characters: `0123456789.,[]`. A neural network schema is a comma-separated list
of matrices. Each matrix is a comma-separated list of vectors, and is enclosed
in square brackets `[]`. These vectors are considered the 'rows' of the matrix.
Each vector is a comma-separated list of real numbers, and is enclosed in
square brackets `[]`. A real number is a non-empty contiguous sequence of
integers, followed by a period `.`, followed by another non-empty contiguous
sequence of integers.

Upon initialisation, the program computes the margin Lipschitz bounds for each
pair of logits of the output vector of the neural network read from the input file.
The program's subsequent input is a series of output vectors and epsilon values
(perturbation bounds) to certify each output vector against.

Each output vector is represented by a comma-separated list of reals, followed by a
space, followed by a perturbation bound (epsilon) represented by a real.

The size of the given output vector must be compatible with the given neural network.

An example of stdin input is: `3.0,3.0,4.0 0.49`.


## Training and Evaluating Your Own Model

The main `doit_verified_robust_gloro.sh` script in the
`robustness-verifier/scripts/` directory can also be used to train a
globally robust model and then run our certifier over it to obtain
results akin to those in Table 1 of our paper.

The script supports training three kinds of models, namely the three kinds
of models that appear in Table 1 (MNIST, Fashion MNIST, and CIFAR-10).

The script provides very rudimentary usage information when run with no
arguments:

```
Usage ./doit_verified_robust_gloro.sh "train_and_eval" dataset training_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]
Usage ./doit_verified_robust_gloro.sh basedir dataset training_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]
```

This shows two ways of running the script. The second way, in which the first
argument passed to the script
is a directory, is used to evaluate an existing model (see e.g.
Claim 2 above).

The first way, in which the first argument passed to the script is the
word `train_and_eval` causes the script to train a new globally robust
model, using the `gloro` framework of Leino et al.

Here, we document each of the parameters that appear in this usage:
```
./doit_verified_robust_gloro.sh "train_and_eval" dataset training_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]
```

* `dataset` must be one of `mnist`, `fashion_mnist` and `cifar10`, and
specifies what type of model to train.
* `training_epsilon` is the perturbation bound used during globally robust
training. In the table in the CAV 2025 paper in the appendix listing
the training hyperparamters, this parameter is called `\epsilon_{train}`.
* `INTERNAL_LAYER_SIZES` is a Python literal that evaluates to a list of positive numbers. The length of the list defines the number of internal layers in the neural network that will be trained, where entry `i` defines the number of
neurons in internal layer `i` (indexed from 0). In particular, the script
builds and trains a fully-connected, Dense neural network whose input and
output size is defined by the `dataset` parameter. To train a model with no
internal layers, use `[]` for this parameter.
* `eval_epsilon` is the perturbation bound against which the trained model's
robustness will be evaluated (over its test set). This is parameter `\epsilon`
in Table 1 of the CAV 2025 paper.
* `robustness_certifier_binary` is the path to the executable file produced
by building our certifier (see Claim 1 above). Usually, this is `../main`
when this script is run from the `scripts/` subdirectory of the
`robustness-verifier/` directory.
* `GRAM_ITERATIONS` specifies the number of gram iterations to use when our
certifier is run, to compute safe Lipschitz bounds for the trained model.
Increasing this number means our certifier takes longer to run but will
produce less conservative robustness certifications.
* `epochs` is the number of training epochs to run model training for. In
the table in the Appendix of the CAV 2025 paper showing the model training
hyperparameters, this is `# epochs`.
* `batch_size` is the batch size to use during model training. 
In
the table in the Appendix of the CAV 2025 paper showing the model training
hyperparameters, this is `batch size`.

