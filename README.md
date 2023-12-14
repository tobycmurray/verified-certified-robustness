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

## Further Development

Currently, the spectral norms required to compute Lipschitz bounds are
hard-coded as 1.0. An 'assume' statement is implemented here in order to
convince Dafny that these are real spectral norms, hence the verification is
currently unsound. To finish the project, a spectral norm function needs to be
implemented and verified.

Generating spectral norms requires iterative methods that converge on a
solution. To date, the most common and efficient way to generate spectral norms
is the power method. It doesn't guarantee a constant rate of convergence, but
does guarantee that each iteration decreases the distance between the derived
value and the actual spectral norm. This guarantee lends itself to two
verification strategies for the power method.

The most precise method is to partition the vector space on each iteration by
ruling out all the vectors that cannot possibly be the spectral norm. Suppose we
begin with a starting vector [2,2] and on the next iteration we get [4,0]. Then
we can rule out all vectors above the line defined by y = x - 2. We can keep
triangulating this way until we encircle a point. This gives us our error
margin.

A less-precise but easier method is to add a bit of noise to the vector and note
the direction it moves on the next iteration. For example, if we modify a vector
by adding 0.1 to each dimension, and the next iteration decreases every
dimension of this modified vector, then its norm is an upper bound on the
spectral norm. One of the problems here is that, as the number of dimensions
increases, there is a fast increase in the difference in norm of the modified
vector and the old vector. Intuitively, the gap between the outside of a sphere
and the corner of an encompassing cube is larger than the gap between the corner
of a circle and an encompassing square. This may result in an over-conservative
estimation for vectors of high dimension.
