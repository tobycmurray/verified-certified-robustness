include "basic_arithmetic.dfy"
include "linear_algebra.dfy"

module OperatorNorms {
import opened BasicArithmetic
import opened LinearAlgebra

// maximum number of iterations to run the gram-iteration algorithm for
const GRAM_ITERATIONS := 11

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
method GenerateSpecNorms(n: seq<Matrix>) returns (r: seq<real>)
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
  var n := SquareMatrixElementsImpl(m);
  var x := SumPositiveMatrixImpl(n);
  r := SqrtUpperBound(x);
}

// method FrobeniusNormUpperBound(m: Matrix) returns (r: real)
//   ensures r >= FrobeniusNorm(m)
// {
//   if DEBUG { print "Computing frobenius norm upper bound for matrix of size ", |m|, "x", |m[0]|, "\n"; }
//   r := 0.0;
//   for i := 0 to |m|
//     invariant r >= 0.0
//     invariant i != 0 ==> r == SumPositiveMatrix(SquareMatrixElements(m[..i]))
//   {
//     for j := 0 to |m[i]|
//       invariant r >= 0.0

//     {
//       r := r + m[i][j] * m[i][j];
//     }
//   }
//   r := SqrtUpperBound(r);
// }

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
    var G_transpose := TransposeImpl(G');
    G' := MMImpl(G_transpose, G');
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

/* ================================= Lemmas ================================= */

// ASSUMPTIONS
lemma {:axiom} Assumption1(m: Matrix)
  ensures SpecNorm(m) <= Sqrt(SpecNorm(MM(Transpose(m), m)))

lemma {:axiom} Assumption2(m: Matrix)
  ensures SpecNorm(m) <= FrobeniusNorm(m)

lemma {:axiom} CauchySchwartz(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Dot(v, u) * Dot(v, u) <= Dot(v, v) * Dot(u, u)

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

// Toby's stuff (with James's edits) ===========================================

// FIXME: move to linear_algebra.dfy
lemma AbsDotL2(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Abs(Dot(v,u)) <= L2(v) * L2(u)
{
  CauchySchwartz(v,u);
  assert 0.0 <= Square(Dot(v, u));
  // (v.u)^2 <= v.v * u.u
  assert Square(Dot(v, u)) <= Dot(v,v) * Dot(u,u);
  // sqrt((v.u)^2) <= sqrt(v.v * u.u)
  assert Sqrt(Square(Dot(v, u))) <= Sqrt(Dot(v, v) * Dot(u, u)) by {
    var x := Square(Dot(v, u));
    var y := Dot(v, v) * Dot(u, u);
    MonotonicSqrt(x, y);
    assert Sqrt(x) <= Sqrt(y);
    calc {
      Sqrt(Square(Dot(v, u)));
      ==
      Sqrt(x);
      <=
      Sqrt(y);
      ==
      {
        assert y == Dot(v, v) * Dot(u, u);
      }
      Sqrt(Dot(v, v) * Dot(u, u));
    }
    assert Sqrt(Square(Dot(v, u))) <= Sqrt(Dot(v, v) * Dot(u, u));
  } // WTF Dafny !?!
  // |(v.u)| <= sqrt(v.v * u.u)
  assert Abs(Dot(v,u)) <= Sqrt(Dot(v,v) * Dot(u,u)) by {  SqrtOfSquare(); } // WTF Dafny !?!
  
  DotSelfIsNonNegative(v);
  DotSelfIsNonNegative(u);
  assert Sqrt(Dot(v,v) * Dot(u,u)) == Sqrt(Dot(v,v)) * Sqrt(Dot(u,u)) by {
    var x := Dot(v, v);
    var y := Dot(u, u);
    SqrtOfMult(x, y);
    assert Sqrt(x * y) == Sqrt(x) * Sqrt(y);
    calc {
      Sqrt(Dot(v, v) * Dot(u, u));
      ==
      {
        assert x * y == Dot(v, v) * Dot(u, u);
      }
      Sqrt(x * y);
      ==
      Sqrt(x) * Sqrt(y);
      ==
      Sqrt(Dot(v, v)) * Sqrt(Dot(u, u));
    }
    assert Sqrt(Dot(v, v) * Dot(u, u)) == Sqrt(Dot(v, v)) * Sqrt(Dot(u, u));
  } // WTF Dafny !?!

  calc {
    Abs(Dot(v,u));
    <=
    Sqrt(Dot(v,v) * Dot(u,u));
    ==
    Sqrt(Dot(v,v)) * Sqrt(Dot(u,u));
  }

  // |(v.u)| <= sqrt(v.v) * sqrt(u.u)
  assert Abs(Dot(v,u)) <= Sqrt(Dot(v,v)) * Sqrt(Dot(u,u)); // WTF Dafny !?!
  
  L2IsSqrtDot(v);
  L2IsSqrtDot(u);
  assert Abs(Dot(v,u)) <= L2(v) * L2(u);  // WTF Dafny !?!
}

lemma L2IsSpecNormUpperBound(s: real, m: Matrix)
  requires |m| == 1
  requires s >= L2(m[0])
  ensures IsSpecNormUpperBound(s, m)
{
  PositiveL2();
  forall u: Vector | |u| == |m[0]|
    ensures L2(MV(m, u)) <= s * L2(u)
  {
    AbsDotL2(m[0], u);
    assert Abs(Dot(m[0], u)) <= L2(m[0]) * L2(u);
    MultIsMono(L2(m[0]), s, L2(u));
    assert s * L2(u) >= L2(m[0]) * L2(u);
    assert Abs(Dot(m[0],u)) <= s * L2(u);
    assert L2(MV(m, u)) == Abs(Dot(m[0],u)) by {
      L2MVIsAbsDot(m[0],u);
      assert L2(MV([m[0]], u)) == Abs(Dot(m[0], u));
      assert L2(MV(m, u)) == Abs(Dot(m[0], u)) by {
        assert [m[0]] == m;
      }
    }
    assert L2(MV(m, u)) <= s * L2(u);
  }
}

}
