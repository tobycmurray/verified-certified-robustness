include "basic_arithmetic.dfy"
include "linear_algebra.dfy"

module OperatorNorms {
import opened BasicArithmetic
import opened LinearAlgebra

// maximum number of iterations to run the gram-iteration algorithm for
const GRAM_ITERATIONS := 6

/* ============================ Ghost Functions ============================= */

ghost function FrobeniusNorm(m: Matrix): real {
  Sqrt(SumPositiveMatrix(SquareMatrixElements(m)))
}

ghost function {:axiom} SpecNorm(m: Matrix): (r: real)
  ensures r >= 0.0
  ensures IsSpecNormUpperBound(r, m)
  ensures !exists x: real :: 0.0 <= x < r && IsSpecNormUpperBound(x, m)

ghost predicate IsSpecNormUpperBound(s: real, m: Matrix) {
  s >= 0.0 && forall v: Vector | |v| == |m[0]| :: L2(MV(m, v)) <= s * L2(v)
}

/* ================================ Methods ================================= */

/** Generates the spectral norm for each matrix in n. */
method GenerateSpecNorms(n: NeuralNetwork) returns (r: seq<real>)
  ensures |r| == |n|
  ensures forall i | 0 <= i < |n| :: IsSpecNormUpperBound(r[i], n[i])
{
  var i := 0;
  r := [];
  while i < |n|
    invariant 0 <= i == |r| <= |n|
    invariant forall j | 0 <= j < i :: IsSpecNormUpperBound(r[j], n[j])
  {
    if DEBUG { print "Generating spectral norm ", i, " of ", |n|, "...\n"; }
    var specNorm := GramIterationSimple(n[i]);
    assert specNorm >= SpecNorm(n[i]);
    r := r + [specNorm];
    i := i + 1;
  }
}

method FrobeniusNormUpperBound(m: Matrix) returns (r: real)
  ensures r >= FrobeniusNorm(m)
{
  if DEBUG { print "Computing frobenius norm upper bound for matrix of size ", |m|, "x", |m[0]|, "\n"; }
  r := SqrtUpperBound(SumPositiveMatrix(SquareMatrixElements(m)));
}

method GramIterationSimple(G: Matrix) returns (s: real)
  ensures IsSpecNormUpperBound(s, G)
{
  var i := 0;
  var G' := G;
  while i != GRAM_ITERATIONS
    invariant 0 <= i <= GRAM_ITERATIONS
    invariant SpecNorm(G) <= Power2Root(SpecNorm(G'), i)
  {
    if DEBUG { print "Gram iteration for matrix of size ", |G|, "x", |G[0]|, ". Iteration ", i+1, " of ", GRAM_ITERATIONS, "\n"; }
    Assumption1(G');
    Power2RootMonotonic(SpecNorm(G'), Sqrt(SpecNorm(MM(Transpose(G'), G'))), i);
    G' := MM(Transpose(G'), G');
    Power2RootDef(SpecNorm(G'), i);
    i := i + 1;
  }
  if DEBUG { print "Gram iteration done iterating\n"; }
  Assumption2(G');
  Power2RootMonotonic(SpecNorm(G'), FrobeniusNorm(G'), GRAM_ITERATIONS);
  if DEBUG { print "Gram iteration computing frobenius norm upper bound...\n"; }
  s := FrobeniusNormUpperBound(G');
  Power2RootMonotonic(FrobeniusNorm(G'), s, GRAM_ITERATIONS);
  if DEBUG { print "Gram iteration computing square root upper bound...\n"; }
  s := Power2RootUpperBound(s, GRAM_ITERATIONS);
  SpecNormUpperBoundProperty(s, G);
  if DEBUG { print "Gram iteration done\n"; }
}

  // method GramIterationSimple(G0: Matrix, N: int) returns (r: real)
  //   requires N >= 0
  //   ensures r >= SpecNorm(G0)
  // {
  //   var i := 0;
  //   var G := G0;
  //   while i != N
  //     invariant 0 <= i <= N
  //     invariant SpecNorm(G0) <= Power2Root(SpecNorm(G), i)
  //   {
  //     Assumption1(G);
  //     Power2RootMonotonic(SpecNorm(G), Sqrt(SpecNorm(MM(Transpose(G), G))), i);
  //     G := MM(Transpose(G), G);
  //     Power2RootDef(SpecNorm(G), i);
  //     i := i + 1;
  //   }
  //   Assumption2(G);
  //   Power2RootMonotonic(SpecNorm(G), FrobeniusNorm(G), N);
  //   assert SpecNorm(G0) <= Power2Root(SpecNorm(G), i);
  //   assert SpecNorm(G0) <= Power2Root(FrobeniusNorm(G), i);
  //   r := FrobeniusNormUpperBound(G);
  //   Power2RootMonotonic(FrobeniusNorm(G), r, N);
  //   assert r >= 0.0;
  //   assert SpecNorm(G0) <= Power2Root(r, i);
  //   while i != 0
  //     invariant 0 <= i <= N
  //     invariant SpecNorm(G0) <= Power2Root(r, i)
  //   {
  //     r := SqrtUpperBound(r, SQRT_ERROR_MARGIN);
  //     i := i - 1;
  //   }
  //   assert SpecNorm(G0) <= Power2Root(r, 0);
  //   assert SpecNorm(G0) <= r;
  //   SpecNormUpperBoundProperty(r, G0);
  // }


// ASSUMPTIONS
  lemma {:axiom} Assumption1(m: Matrix)
    ensures SpecNorm(m) <= Sqrt(SpecNorm(MM(Transpose(m), m)))

  lemma {:axiom} Assumption2(m: Matrix)
    ensures SpecNorm(m) <= FrobeniusNorm(m)

  /* We only need these for rescaling gram iteration */

  // lemma {:axiom} Assumption3(m: Matrix, x: real)
  //   requires 0.0 < x
  //   ensures SpecNorm(m) <= SpecNorm(MatrixDiv(m, x)) * x

  // function MatrixDiv(m: Matrix, x: real): (r: Matrix)
  //   requires 0.0 < x
  //   ensures |r| == |m| && |r[0]| == |m[0]|
  // {
  //   if |m| == 1 then [VectorDiv(m[0], x)]
  //   else [VectorDiv(m[0], x)] + MatrixDiv(m[1..], x)
  // }

  // function VectorDiv(v: Vector, x: real): (r: Vector)
  //   requires 0.0 < x
  //   ensures |r| == |v|
  // {
  //   if |v| == 1 then [v[0] / x] else [v[0] / x] + VectorDiv(v[1..], x)
  // }

  lemma SpecNormUpperBoundProperty(s: real, m: Matrix)
    requires s >= SpecNorm(m)
    ensures s >= 0.0
    ensures IsSpecNormUpperBound(s, m)
  {
    PositiveL2();
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
   * If each element in v has a lower absolute value than its counterpart in u,
   * then ||v|| <= ||u||.
   */
  lemma SmallerL2Norm(v: Vector, u: Vector)
    requires |v| == |u|
    requires forall i: int :: 0 <= i < |v| ==> Abs(v[i]) <= Abs(u[i])
    ensures L2(v) <= L2(u)
  {
    reveal L2();
    SmallerApplySquare(v, u);
    MonotonicSum(Apply(v, Square), Apply(u, Square));
    MonotonicSqrt(Sum(Apply(v, Square)), Sum(Apply(u, Square)));
  }

}