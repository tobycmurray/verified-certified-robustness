include "basic_arithmetic.dfy"
include "linear_algebra.dfy"

module NeuralNetworks {
import opened BasicArithmetic
import opened LinearAlgebra

/* ================================= Types ================================== */

/* A neural network is a non-empty sequence of layers, which are
(functionally) weight matrices, and whose product must be defined. */
type NeuralNetwork = n: seq<Matrix> | |n| > 0 && 
  forall i: int :: 0 <= i < |n| - 1 ==> Rows(n[i]) == Cols(n[i + 1])
  witness [NonEmptyMatrix()]

/** Required to convince Dafny that [[0.0]] is a Matrix. */
function NonEmptyMatrix(): Matrix {
  [[0.0]]
}

/* ============================ Ghost Functions ============================= */

/** True iff the size of v is compatible as an input-vector to n. */
ghost predicate IsInput(v: Vector, n: NeuralNetwork) {
  |v| == Cols(n[0])
}

/** True iff the size of v is compatible as an output-vector of n. */
ghost predicate CompatibleOutput(v: Vector, n: NeuralNetwork) {
  |v| == Rows(n[|n|-1])
}

/** Applies the ReLu activation function to the given vector. */
ghost opaque function ApplyRelu(v: Vector): (r: Vector)
  ensures |r| == |v|
  ensures forall i: int :: 0 <= i < |r| ==> r[i] == Relu(v[i])
  ensures forall i: int :: 0 <= i < |r| ==> Abs(r[i]) <= Abs(v[i])
{
  Apply(v, Relu)
}

/**
 * The assumed functionality of a neural network layer. Applies matrix-vector
 * multiplication, followed by the relu activation function.
 */
ghost function Layer(m: Matrix, v: Vector): (r: Vector)
  requires |v| == |m[0]|
{
  ApplyRelu(MV(m, v))
}

/**
 * Function representing the assumed behaviour of the neural network. Models
 * how the neural network transforms input vectors into output vectors.
 */
ghost opaque function NN(n: NeuralNetwork, v: Vector): (r: Vector)
  requires IsInput(v, n)
  ensures CompatibleOutput(r, n)
  ensures |n| == 1 ==> r == ApplyRelu(MV(n[0], v))
{
  if |n| == 1 then Layer(n[0], v) else Layer(n[|n|-1], NN(n[..|n|-1], v))
}

/* ================================= Lemmas ================================= */

/**
 * Let n' be the neural network n with all rows except row l removed.
 * Formally, n' == n[..|n|-1] + [[n[|n|-1][l]]].
 * Show that NN(n', v) == [NN(n, v)[l]].
 */
lemma TrimmedNN(n: NeuralNetwork, n': NeuralNetwork, v: Vector, l: int)
  requires 0 <= l < |n[|n|-1]|
  requires IsInput(v, n) && IsInput(v, n')
  requires |n| == |n'|
  requires n' == n[..|n|-1] + [[n[|n|-1][l]]]
  ensures NN(n', v) == [NN(n, v)[l]]
{
  assert n' == n[..|n|-1] + [[n[|n|-1][l]]];
  reveal NN();
  // It's totally ridiculous that Dafny can do this.
  // calc {
  //   NN(n', v);
  //   Layer(n'[|n'|-1], NN(n'[..|n'|-1], v));
  //   ApplyRelu(MV(n'[|n'|-1], NN(n'[..|n'|-1], v)));
  //   ApplyRelu(MV([n[|n|-1][l]], NN(n'[..|n'|-1], v)));
  //   ApplyRelu(MV([n[|n|-1][l]], NN(n[..|n|-1], v)));
  //   ApplyRelu([Dot([n[|n|-1][l]][0], NN(n[..|n|-1], v))]);
  //   ApplyRelu([Dot(n[|n|-1][l], NN(n[..|n|-1], v))]);
  //   // Dot(m[l], v) == MV(m, v)[l] by definition of MV.
  //   ApplyRelu([MV(n[|n|-1], NN(n[..|n|-1], v))[l]]);
  //   // By definition of apply when argument is len 1.
  //   [Relu(MV(n[|n|-1], NN(n[..|n|-1], v))[l])];
  //   // By definition of ApplyRelu.
  //   [ApplyRelu(MV(n[|n|-1], NN(n[..|n|-1], v)))[l]];
  //   [Layer(n[|n|-1], NN(n[..|n|-1], v))[l]];
  //   [NN(n, v)[l]];
  // }
}
}
