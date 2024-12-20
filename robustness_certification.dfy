include "basic_arithmetic.dfy"
include "linear_algebra.dfy"
include "operator_norms.dfy"
include "neural_networks.dfy"

module RobustnessCertification {
import opened BasicArithmetic
import opened LinearAlgebra
import opened OperatorNorms
import opened NeuralNetworks

/* ============================ Ghost Functions ============================= */

/**
 * The robustness property is the key specification for the project. An
 * input-output pair of vectors (v, v') for a neural network n is robust
 * with respect to an error ball e if for all input vectors u within a
 * distance e from v, the classification (i.e., argmax) of the output
 * corresponding to u is equal to the classification of v'.
 */
ghost predicate Robust(v: Vector, v': Vector, e: real, n: NeuralNetwork)
  requires CompatibleInput(v, n)
  requires NN(n, v) == v'
{
  forall u: Vector | |v| == |u| && Distance(v, u) <= e ::
    ArgMax(v') == ArgMax(NN(n, u))
}

/** True iff every L[i] is a Lipschitz bound of the matrix n[i]. */
ghost predicate AreLipBounds(n: NeuralNetwork, L: seq<real>)
  requires |L| == |n[|n|-1]|
{
  forall i | 0 <= i < |L| :: IsLipBound(n, L[i], i)
}

/**
 * A real number l is a Lipschitz bound of an output logit i iff l is an
 * upper bound on the change in i per change in distance of the input vector.
 */
ghost predicate IsLipBound(n: NeuralNetwork, l: real, i: int)
  requires 0 <= i < |n[|n|-1]|
{
  forall v, u: Vector | CompatibleInput(v, n) && CompatibleInput(u, n) ::
    Abs(NN(n, v)[i] - NN(n, u)[i]) <= l * Distance(v, u)
}

/* ================================ Methods ================================= */

/**
 * Certifies the output vector v' against the error ball e and Lipschitz
 * constants L. If certification succeeds (returns true), any input
 * corresponding to v' is verified robust.
 */
method Certify(v': Vector, e: real, L: seq<real>) returns (b: bool)
  requires forall i | 0 <= i < |L| :: 0.0 <= L[i]
  requires |v'| == |L|
  ensures b ==> forall v: Vector, n: NeuralNetwork |
    CompatibleInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  var x := ArgMaxImpl(v');
  var i := 0;
  b := true;
  while i < |v'|
    invariant 0 <= i <= |v'|
    invariant b ==> forall j | 0 <= j < i && j != x ::
      v'[x] - L[x] * e > v'[j] + L[j] * e
  {
    if i == x {
      i := i + 1;
      continue;
    }
    if v'[x] - L[x] * e <= v'[i] + L[i] * e {
      b := false;
      break;
    }
    i := i + 1;
  }
  if b {
    ProveRobust(v', e, L, x);
  }
}

/**
 * Generates the Lipschitz bound for each logit in the output of the neural
 * network n. See GenLipBound for details.
 */
method GenLipBounds(n: NeuralNetwork, s: seq<real>) returns (r: seq<real>)
  requires |s| == |n|
  requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures |r| == |n[|n|-1]|
  ensures forall i | 0 <= i < |r| :: 0.0 <= r[i]
  ensures AreLipBounds(n, r)
{
  r := [];
  var i := 0;
  while i < |n[|n|-1]|
    invariant 0 <= i <= |n[|n|-1]|
    invariant |r| == i
    invariant forall j | 0 <= j < i ::
      0.0 <= r[j] && IsLipBound(n, r[j], j)
  {
    var bound := GenLipBound(n, i, s);
    r := r + [bound];
    i := i + 1;
    assert forall j | 0 <= j < i :: IsLipBound(n, r[j], j) by {
      assert forall j | 0 <= j < i - 1 :: IsLipBound(n, r[j], j);
      assert IsLipBound(n, r[i-1], i-1);
    }
  }
  assert AreLipBounds(n, r);
}

method GenLipBound(n: NeuralNetwork, l: nat, s: seq<real>) returns (r: real)
  requires |s| == |n|
  requires l < Rows(n[|n|-1])
  requires forall i: nat | i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures IsLipBound(n, r, l)
  ensures r >= 0.0
{
  if (|n| > 1) {
    var i := |n| - 1;
    var m: Matrix := [n[i][l]];
    r := GramIterationSimple(m);
    assert IsSpecNormUpperBound(r, m);
    forall v: Vector, u: Vector | |v| == |[n[i][l]][0]| && |u| == |[n[i][l]][0]|
    { Helper1(n, v, u, r, l); }
    assert IsLipBound(n[i..], r, l);
    assume {:axiom} false;
  } else {
    assume {:axiom} false;
  }
}

lemma Helper1(n: NeuralNetwork, v: Vector, u: Vector, r: real, l: nat)
  requires |n| > 1
  requires l < Rows(n[|n|-1])
  requires IsSpecNormUpperBound(r, [n[|n|-1][l]])
  requires |v| == |[n[|n|-1][l]][0]|
  requires |u| == |[n[|n|-1][l]][0]|
  ensures Abs(NN(n[|n|-1..], v)[l] - NN(n[|n|-1..], u)[l]) <= r * Distance(v, u)
{
  var i := |n| - 1;
  var m: Matrix := [n[i][l]];
  assert IsSpecNormUpperBound(r, m);
  SpecNormIsLayerLipBound(m, v, u, r);
  assert Distance(Layer(m, v), Layer(m, u)) <= r * Distance(v, u);
  calc {
    Layer(m, v);
    ==
    Layer([n[i][l]], v);
    ==
    [NN(n[i..], v)[l]];
  }
  calc {
    Layer(m, u);
    ==
    Layer([n[i][l]], u);
    ==
    [NN(n[i..], u)[l]];
  }
  assert Distance([NN(n[i..], v)[l]], [NN(n[i..], u)[l]]) <= r * Distance(v, u);
  calc {
    Distance([NN(n[i..], v)[l]], [NN(n[i..], u)[l]]);
    ==
    {
      NormOfOneDimensionIsAbs();
    }
    Abs(NN(n[i..], v)[l] - NN(n[i..], u)[l]);
  }
  assert Abs(NN(n[i..], v)[l] - NN(n[i..], u)[l]) <= r * Distance(v, u);
}

/**
 * Generates the Lipschitz bound of logit l. This is achieved by taking the
 * product of the spectral norms of the first |n|-1 layers, and multiplying
 * this by the spectral norm of the matrix [v], where v is the vector
 * corresponding to the l'th row of the final layer of n.
 */
// method GenLipBound(n: NeuralNetwork, l: int, s: seq<real>) returns (r: real)
//   requires |s| == |n|
//   requires 0 <= l < |n[|n|-1]|
//   requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
//   ensures IsLipBound(n, r, l)
//   ensures r >= 0.0
// {
//   var trimmedLayer := [n[|n|-1][l]];
//   var trimmedSpecNorm := GramIterationSimple(trimmedLayer);
//   var n' := n[..|n|-1] + [trimmedLayer];
//   var s' := s[..|s|-1] + [trimmedSpecNorm];
//   r := ProductImpl(s');
//   PositiveProduct(s');
//   forall v: Vector, u: Vector | |v| == |u| && CompatibleInput(v, n') {
//     SpecNormProductIsLipBound(n', v, u, s');
//   }
//   forall v: Vector, u: Vector | |v| == |u| && CompatibleInput(v, n') {
//     LogitLipBounds(n, n', v, u, l);
//   }
// }

/* ================================= Lemmas ================================= */

/**
 * Given an output vector v', error ball e and Lipschitz bounds L, if the
 * dominant logit x of v' is reduced by L[x] * e, and all other logits i are
 * increased by L[i] * e, and the dominant logit remains x, then v' is
 * robust. This follows from the fact that the maximum change in any logit i
 * is L[i] * e.
 */
lemma ProveRobust(v': Vector, e: real, L: seq<real>, x: int)
  requires forall i | 0 <= i < |L| :: 0.0 <= L[i]
  requires |v'| == |L|
  requires x == ArgMax(v')
  requires forall i | 0 <= i < |v'| && i != x ::
    v'[x] - L[x] * e > v'[i] + L[i] * e
  ensures forall v: Vector, n: NeuralNetwork |
    CompatibleInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  assume {:axiom} false;
  assert forall n: NeuralNetwork, v: Vector, u: Vector, i |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| &&
    |v| == |u| && CompatibleInput(v, n) && Distance(v, u) <= e ::
    Abs(NN(n, v)[i] - NN(n, u)[i]) <= L[i] * e;
  ProveRobustHelper(v', e, L, x);
  assert forall n: NeuralNetwork, v: Vector, u: Vector |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) && |v| == |u| &&
    CompatibleInput(v, n) && Distance(v, u) <= e && v' == NN(n, v) ::
    NN(n, u)[x] >= v'[x] - L[x] * e &&
    forall i | 0 <= i < |L| && i != x ::
    NN(n, u)[i] <= v'[i] + L[i] * e;
  assert forall n: NeuralNetwork, v: Vector, u: Vector, i |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| && |v| == |u| &&
    CompatibleInput(v, n) && Distance(v, u) <= e && v' == NN(n, v) &&
    i != x ::
    NN(n, u)[i] < NN(n, u)[x];
}

/**
 * Sometimes Dafny needs a separate lemma to prove the obvious. In short,
 * this lemma proves that if this (rather verbose) property holds for all i
 * in S, and x is in S, then it also holds specifically for x.
 */
lemma ProveRobustHelper(v': Vector, e: real, L: seq<real>, x: int)
  requires |v'| == |L|
  requires x == ArgMax(v')
  requires forall n: NeuralNetwork, v: Vector, u: Vector, i |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| &&
    |v| == |u| && CompatibleInput(v, n) && Distance(v, u) <= e ::
    Abs(NN(n, v)[i] - NN(n, u)[i]) <= L[i] * e
  ensures forall n: NeuralNetwork, v: Vector, u: Vector |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) &&
    |v| == |u| && CompatibleInput(v, n) && Distance(v, u) <= e ::
    Abs(NN(n, v)[x] - NN(n, u)[x]) <= L[x] * e
{}

lemma SpecNormProductLipBoundHelper(n: NeuralNetwork, s: seq<real>)
  requires |s| == |n|
  requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures forall v: Vector, u: Vector
    | |v| == |u| && CompatibleInput(v, n) && CompatibleInput(u, n)
    :: Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
{
  assume {:axiom} false;
  forall v: Vector, u: Vector
    | |v| == |u| && CompatibleInput(v, n) && CompatibleInput(u, n) {
    SpecNormProductIsLipBound(n, v, u, s);
  }
}

/**
 * The product of the spectral norms of each matrix of a neural network n is
 * a Lipschitz bound on the l2 norm of the output vector of n.
 */
lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
    s: seq<real>)
  requires |v| == |u| && |s| == |n|
  requires CompatibleInput(v, n) && CompatibleInput(u, n)
  requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
{
  if |n| == 1 {
    SpecNormIsLayerLipBound(n[0], v, u, s[0]);
    reveal Product();
  } else {
    SpecNormIsLayerLipBound(n[0], v, u, s[0]);

    var n0 := n[..|n|-1];
    var s0 := s[..|s|-1];
    assert |s0| == |n0|;
    SpecNormProductIsLipBound(n0, v, u, s0);

    var n' := n[|n|-1];
    var s' := s[|s|-1];
    var v' := NN(n0, v);
    var u' := NN(n0, u);

    SpecNormIsLayerLipBound(n', v', u', s');
    reveal NN();
    assert Distance(NN(n, v), NN(n, u)) <= s' * Distance(v', u');
    assert Distance(v', u') <= Product(s0) * Distance(v, u);
    MultiplicationInequality(n, v, u, v', u', s0, s');
    ProductDef(s, s0, s');
    MultiplyBothSides(s, s0, s', v, u);
  }
}

/** An obvious helper-lemma to SpecNormProductIsLipBound. */
lemma MultiplyBothSides(s: seq<real>, s0: seq<real>, s': real, v: Vector,
    u: Vector)
  requires |v| == |u|
  requires Product(s) == s' * Product(s0)
  ensures Product(s) * Distance(v, u) == s' * Product(s0) * Distance(v, u)
{}

/** An obvious helper-lemma to SpecNormProductIsLipBound. */
lemma MultiplicationInequality(n: NeuralNetwork, v: Vector, u: Vector,
    v': Vector, u': Vector, s0: seq<real>, s': real)
  requires |v| == |u|
  requires |v'| == |u'|
  requires s' >= 0.0
  requires CompatibleInput(v, n) && CompatibleInput(u, n)
  requires Distance(NN(n, v), NN(n, u)) <= s' * Distance(v', u')
  requires Distance(v', u') <= Product(s0) * Distance(v, u)
  ensures Distance(NN(n, v), NN(n, u)) <= s' * Product(s0) * Distance(v, u)
{}

/**
 * As seen in the method GenLipBound, computing a Lipschitz bound on logit l
 * for a neural network n involves 'trimming' all rows out of the final layer
 * of n except for row l, and computing the spectral norm of this new neural
 * network n'. This lemma relates a Lipschitz bound on the output vector of
 * n' to a Lipschitz bound on the logit l in n.
 */
lemma LogitLipBounds(n: NeuralNetwork, n': NeuralNetwork, v: Vector,
    u: Vector, l: int)
  requires |v| == |u|
  requires |n| == |n'|
  requires CompatibleInput(v, n)
  requires CompatibleInput(u, n)
  requires 0 <= l < |n[|n|-1]|
  requires n' == n[..|n|-1] + [[n[|n|-1][l]]]
  ensures Distance(NN(n', v), NN(n', u)) == Abs(NN(n, v)[l] - NN(n, u)[l])
{
  TrimmedNN(n, n', v, l);
  TrimmedNN(n, n', u, l);
  NormOfOneDimensionIsAbs();
}

/**
 * The distance between two vectors can only be decreased when the ReLu
 * function is applied to each one. This is equivalent to stating that the
 * spectral norm of the ReLu layer is 1.
 * ||R(v) - R(u)|| <= ||v - u|| where R applies the ReLu activation function.
 */
lemma SmallerRelu(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Distance(ApplyRelu(v), ApplyRelu(u)) <= Distance(v, u)
{
  SmallerL2Norm(Minus(ApplyRelu(v), ApplyRelu(u)), Minus(v, u));
}

/**
 * A neural network layer consists of matrix-vector multiplication, followed
 * by an application of the ReLu activation function. A Lipschitz bound of a
 * layer with matrix m is the spectral norm of that matrix.
 * ||R(m.v) - R(m.u)|| <= ||m|| * ||v - u||
 * where R applies the ReLu activation function.
 */
lemma SpecNormIsLayerLipBound(m: Matrix, v: Vector, u: Vector, s: real)
  requires |m[0]| == |v| == |u|
  requires IsSpecNormUpperBound(s, m)
  ensures Distance(Layer(m, v), Layer(m, u)) <= s * Distance(v, u)
{
  SpecNormIsMvLipBound(m, v, u, s);
  SmallerRelu(MV(m, v), MV(m, u));
}

/** 
 * A matrix's spectral norm is a Lipschitz bound:
 * ||m.v - m.u|| <= ||m|| * ||v - u||
 */
lemma SpecNormIsMvLipBound(m: Matrix, v: Vector, u: Vector, s: real)
  requires |v| == |u| == |m[0]|
  requires IsSpecNormUpperBound(s, m)
  ensures Distance(MV(m, v), MV(m, u)) <= s * Distance(v, u)
{
  SpecNormPropertyHoldsForDifferenceVectors(m, s, v, u);
  MvIsDistributive(m, v, u);
}

/**
 * Since v - u is just a vector, we have ||m.(v - u)|| <= ||m|| * ||v - u||
 */
lemma SpecNormPropertyHoldsForDifferenceVectors(m: Matrix, s: real,
    v: Vector, u: Vector)
  requires |v| == |u| == |m[0]|
  requires IsSpecNormUpperBound(s, m)
  ensures L2(MV(m, Minus(v, u))) <= s * Distance(v, u)
{}
}
