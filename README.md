## About

Globally robust neural networks offer an efficient procedure for certifying the
robustness of neural network outputs. This Dafny project takes this a step
further by formally verifying an implementation of this certification
procedure. Given a text-based representation of a neural network, the program
not only *certifies*, but *verifies* the robustness of a given output vector
with respect to an input error margin (often called an error-ball). The neural
network is assumed to use ReLu activation functions. Convolutional layers are
currently not supported.

More information on globally-robust neural networks can be found in the paper
[*Globally-Robust Neural Networks*
(2021)](https://arxiv.org/pdf/2102.08452.pdf) by Leino, Wang and Fredrikson.

## Assumptions

* We assume the existence of a verified spectral-norm generator for real
rectangular matrices.
* We do not currently account for problems associated with conversions from
real numbers (used in proofs) to floating point numbers (used at compile time).
* Parsing of user input in text files and stdin is assumed to be correct.

## Usage

Verify all files:
`./verify_all.sh`\
Build & run:
`dafny run --unicode-char:false --target:cs main.dfy --input IO/FileIO.cs`

The program expects a neural network schema located at
`Input/neural_network.txt`. This file must strictly contain only the following
characters: `0123456789.,[]`. A neural network schema is a comma-separated list
of matrices. Each matrix is a comma-separated list of vectors, and is enclosed
in square brackets `[]`. These vectors are considered the 'rows' of the matrix.
Each vector is a comma-separated list of real numbers, and is enclosed in
square brackets `[]`. A real number is a non-empty contiguous sequence of
integers, followed by a period `.`, followed by another non-empty contiguous
sequence of integers.

Upon initialisation, the program computes the Lipschitz bounds for each logit
of the output vector of the neural network read from the file above. This is
the most computationally-intensive part of the program. Once these are
computed, the user is repeatedly prompted to enter an output vector and error
margin via stdin. The robustness of this output vector against the given error
margin can be efficiently certified using these Lipschitz bounds.

For stdin input, the program expects an output vector represented by a
comma-separated list of reals, followed by a space, followed by an error margin
represented by a real. Currently, stdin input must be terminated with an EOF
character `^D` rather than a newline. Additionally, the size of the given
output vector must be compatible with the given neural network.

An example of stdin input is: `3.0,3.0,4.0 0.49^D`.
