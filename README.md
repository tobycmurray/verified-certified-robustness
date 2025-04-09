# A Formally Verified Robustness Certifier for Neural Networks

## CAV 2025 Artifact

The artifact is packaged as a Docker container, which is known to work
on x86-64 hosts. (Unfortunately, virtualisation failures seem to pevent
the container working on ARM-based hosts like Apple M1 etc.)

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

The artifact has the following directory structure, and is located in the
`/workspace` directory of the container's filesystem:
```
* `robustness-verifier/` - the verified robustness certifier
* `cav2025-artifact/` - the artifact's Python virtual environment directory
* `cav2025-models/` - the three models used in the paper's evaluation
* `gloro/` - Leino et al.'s globally robust neural networks implementation
```



## Verifying and Building the Certifier

### Verifying the Certifier

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

### Building the Certifier

To build the certifier binary, in the `robustness-verifier/` directory, run:
```
dafny build --unicode-char:false --target:cs main.dfy IO/FileIO.cs
```

This should prooduce the executable `main`, which is the main certifier binary.

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

## Reproducing the Paper's Evaluation

The `scripts/` subdirectory of `robustness-verifier/` contains scripts that can
be used to reproduce the evaluation results.

These scripts produce a lot of data files. The `cav2025-models/` directory
contains copies of the three ML models used in the paper's evaluation, as well
as the data files produced from that evaluation.

The models appear in Table 1 of the paper and their corresponding locations
in the `cav2025-models/` directory are:

* MNIST (row 1): `2025-01-25_09:27:46-mnist/`
* Fashion MNIST (row 2): `2025-01-30_10:58:01-fashion_mnist/`
* CIFAR-10 (row 3): `2025-01-28_20:39:32-cifar10/`

The main evaluation script in the `robustness-verifier/scripts/` directory is called
`doit_verified_robust_gloro.sh`. To reproduce the results in Table 1 this script can
be run.

It takes many arguments (that specify the location of the ML model to evaluate,
ML model internal architecture, training and evaluation parameters, etc). These
include the value of `epsilon` used to evaluate the robustness of the model,
and the number of gram iterations to run the certifier for, as shown in Table 1 for
each model.

### MNIST

To reproduce the MNIST results, from the `robustness-verifier/scripts/` directory, run
this command (making sure you have first done `source cav2025-artifact/bin/activate`
to enter the Python virtual environment:

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

It should take approximately 20 minutes to complete, at least. The final output
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
95.76%, i.e. 0.02 percentage points above the percentage certified by our verified certifier (as indicated by the "(-0.02)" in the "Verified Robustness" column of row 1 of
Table 1.
* "VRA" records the percentage of the test points that were both classified accurately by the trained model and certified robust by our certifier. We see that the
"Proportion robust and correct: 0.954" in the output above, i.e., a VRA of 95.40% as shown in this column. Similarly to the percentage certified robust, the gloro unverified
certifier of Leino et al. produces a VRA of 0.9541000127792358, i.e 95.41%, which is 0.01 percentage points above our certifier, as indicated by the "(-0.01)" in this column
of row 1 of Table 1.

## Fashion MNIST

The results in the second row of Table 1 (for Fashion MNIST) can be reproduced by following the same procedure as for MNIST above. However, running the script as follows:

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-30_10:58:01-fashion_mnist"/ fashion_mnist 0.26 "[256,128,128,128,128,128,128,128,128,128,128,128]" 0.25 ../main 12 500 64

```

As indicated in Table 1, it may take around 20 minutes for the script to complete. It produces output akin to that for MNIST, above.

## CIFAR_10

The results for CIFAR-10 (third row of Table 1) can be reproduced by running the script as follows:

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-28_20:39:32-cifar10"/ cifar10  0.1551 "[512,256,128,128,128,128,128,128]" 0.141 ../main 12 800 256
```

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

