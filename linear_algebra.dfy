include "basic_arithmetic.dfy"

module LinearAlgebra {
import opened BasicArithmetic

/* ================================= Types ================================== */

/* A vector is a non-empty sequence of reals. */
type Vector = v: seq<real> | |v| > 0 witness [0.0]

/* A matrix is a non-empty sequence of vectors of equal length. This is
equivalent to a non-empty rectangle of reals. Matrices are indexed row-first.
That is, |m| represents the number of rows, whereas |m[0]| represents the
number of columns. */
type Matrix = m: seq<Vector> | |m| > 0 && |m[0]| > 0 && 
  forall i, j: int :: 0 <= i < |m| && 0 <= j < |m| ==> |m[i]| == |m[j]| 
  witness [NonEmptyVector()]

/** Required to convince Dafny that [0.0] is a Vector and not a seq<real>. */
function NonEmptyVector(): Vector {
  [0.0]
}

/* ============================ Ghost Functions ============================= */

/** Element-wise subtraction of vector u from vector v. */
ghost opaque function Minus(v: Vector, u: Vector): (r: Vector)
  requires |v| == |u|
  ensures |r| == |v|
  ensures forall i: int :: 0 <= i < |r| ==> r[i] == v[i] - u[i]
{
  if |v| == 1 then [v[0] - u[0]] else [v[0] - u[0]] + Minus(v[1..], u[1..])
}

/** Dot product. */
ghost opaque function Dot(v: Vector, u: Vector): real
  requires |v| == |u|
{
  if |v| == 1 then v[0] * u[0] else v[0] * u[0] + Dot(v[1..], u[1..])
}

/** Matrix vector product. */
ghost opaque function MV(m: Matrix, v: Vector): (r: Vector)
  requires |m[0]| == |v|
  ensures |r| == |m|
  ensures forall i: int :: 0 <= i < |r| ==> r[i] == Dot(m[i], v)
{
  if |m| == 1 then [Dot(m[0], v)]
  else [Dot(m[0], v)] + MV(m[1..], v)
}

/** Matrix-matrix multiplication. */
ghost function MM(m: Matrix, n: Matrix): Matrix
  requires |m[0]| == |n|
{
  if |m| == 1 then [MMGetRow(m[0], n)]
  else [MMGetRow(m[0], n)] + MM(m[1..], n)
}

ghost function MMGetRow(v: Vector, n: Matrix): (r: Vector)
  requires |v| == |n|
  ensures |r| == |n[0]|
  decreases |n[0]|
{
  if |n[0]| == 1 then [Dot(v, GetFirstColumn(n))]
  else [Dot(v, GetFirstColumn(n))] + MMGetRow(v, RemoveFirstColumn(n))
}

/** 
 * Applies the given function to all elements of the given vector.
 * Similar to Haskell's 'map' function.
 */
ghost opaque function Apply(v: Vector, f: real -> real): (r: Vector)
  ensures |v| == |r|
  ensures forall i: int :: 0 <= i < |r| ==> r[i] == f(v[i])
{
  if |v| == 1 then [f(v[0])] else [f(v[0])] + Apply(v[1..], f)
}

/**
 * L2 norm of the given vector.
 * L2([v_1, v_2,..., v_n]) = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
 */
ghost opaque function L2(v: Vector): (r: real)
{
  Sqrt(Sum(Apply(v, Square)))
}

/**
 * The 'distance' between two vectors is the norm of their difference-vector.
 */
ghost function Distance(v: Vector, u: Vector): real
  requires |v| == |u|
{
  L2(Minus(v, u))
}

ghost function SquareMatrixElements(m: Matrix): (r: Matrix)
  ensures forall i, j | 0 <= i < |r| && 0 <= j < |r[0]| :: 0.0 <= r[i][j]
{
  if |m| == 1 then [Apply(m[0], Square)]
  else [Apply(m[0], Square)] + SquareMatrixElements(m[1..])
}

ghost function GetFirstColumn(m: Matrix): (r: Vector)
  ensures |r| == |m|
{
  if |m| == 1 then [m[0][0]]
  else [m[0][0]] + GetFirstColumn(m[1..])
}

ghost function RemoveFirstColumn(m: Matrix): (r: Matrix)
  requires |m[0]| > 1
  ensures |r| == |m|
{
  if |m| == 1 then [m[0][1..]]
  else [m[0][1..]] + RemoveFirstColumn(m[1..])
}

ghost function Transpose(m: Matrix): (r: Matrix)
  decreases |m[0]|
{
  if |m[0]| == 1 then [GetFirstColumn(m)]
  else [GetFirstColumn(m)] + Transpose(RemoveFirstColumn(m))
}

ghost predicate IsColumn(r: Vector, n: nat, m: Matrix)
  requires n < Cols(m)
{
  |r| == Rows(m) && forall i: nat | i < |r| :: r[i] == m[i][n]
}

/* =========================== Concrete Functions =========================== */

function Rows(m: Matrix): nat {
  |m|
}

function Cols(m: Matrix): nat {
  |m[0]|
}

/* ================================ Methods ================================= */

method DotImpl(v: Vector, u: Vector) returns (r: real)
  requires |v| == |u|
  ensures r == Dot(v, u)
{
  r := 0;
  for i := 0 to |v| {
    r := r + v[i] * u[i];
  }
}

method MVImpl(m: Matrix, v: Vector) returns (r: Vector)
  requires |m[0]| == |v|
  ensures r == MV(m, v)
{
  var x := DotImpl(m[0], v);
  r := [x];
  for i := 1 to |m| {
    x := DotImpl(m[i], v);
    r := r + [x]
  }
}

method MMImpl(m: Matrix, n: Matrix) returns (r: Matrix)
  requires Cols(m) == Rows(n)
  ensures r == MM(m, n)
{
  // todo: this could be simplified with arrays
  var s: seq<seq<real>> := [];
  for i := 0 to Rows(m) {
    var v: seq<real> := [];
    for j := 0 to Cols(n) {
      var c: seq<real> := ColumnImpl(j, n);
      var x: real := DotImpl(m[i], c);
      v := v + [x]
    }
    s := s + [v]
  }
  r := s;
}

method ApplyImpl(v: Vector, f: real -> real) returns (r: Vector)
  ensures r == Apply(v, f)
{
  r := v;
  for i := 0 to |r| {
    r[i] := f(r[i]);
  }
}

method SquareMatrixElementsImpl(m: Matrix) returns (r: Matrix)
  ensures r == SquareMatrixElements(m)
{
  r := m;
  for i := 0 to |r| {
    for j := 0 to |r[i]| {
      r[i][j] := r[i][j] * r[i][j]
    }
  }
}

method ColumnImpl(n: nat, m: Matrix) returns (r: Vector)
  requires n < Cols(m)
  ensures IsColumn(r, n, m)
{
  r := [m[0][n]];
  for i := 0 to Rows(m) {
    r := r + [m[i][n]];
  }
}

method TransposeImpl(m: Matrix) returns (r: Matrix)
  ensures r == Transpose(m)
{
  // todo: this could be simplified with arrays
  var v: Vector := ColumnImpl(0, m);
  r := [v];
  for i := 1 to Cols(m) {
    v := ColumnImpl(i, m);
    r := r + [v];
  }
}

/* ================================= Lemmas ================================= */

/**
 * If each element in vector v has a lower absolute value than its
 * counterpart in u, then the square of each element in v is lower than the
 * square of its counterpart in u.
 */
lemma SmallerApplySquare(v: Vector, u: Vector)
  requires |v| == |u|
  requires forall i: int | 0 <= i < |v| :: Abs(v[i]) <= Abs(u[i])
  ensures forall i: int | 0 <= i < |v| ::
    Apply(v, Square)[i] <= Apply(u, Square)[i]
{
  var i := 0;
  while i < |v|
    invariant i <= |v|
    invariant forall j: int | 0 <= j < i ::
      Apply(v, Square)[j] <= Apply(u, Square)[j]
  {
    MonotonicSquare(v[i], u[i]);
    i := i + 1;
  }
}

/** 
 * Matrix-vector products distribute over subtraction: 
 * m.(v - u) == m.v - m.u
 */
lemma MvIsDistributive(m: Matrix, v: Vector, u: Vector)
  requires |m[0]| == |v| == |u|
  ensures MV(m, Minus(v, u)) == Minus(MV(m, v), MV(m, u))
{
  for i := 0 to |m|
    invariant forall j: int | 0 <= j < i ::
      Dot(m[j], Minus(v, u)) == Dot(m[j], v) - Dot(m[j], u)
  {
    DotIsDistributive(m[i], v, u);
  }
}

/** Dot products distribute over subtraction: v.(u - w) == v.u - v.w */
lemma DotIsDistributive(v: Vector, u: Vector, w: Vector)
  requires |v| == |u| == |w|
  ensures Dot(v, Minus(u, w)) == Dot(v, u) - Dot(v, w)
{
  reveal Dot();
  if |v| == 1 {
  } else {
    DotIsDistributive(v[1..], u[1..], w[1..]);
    assert Minus(u, w)[1..] == Minus(u[1..], w[1..]);
  }
}

/** The norm of a vector with one element d is the absolute value of d. */
lemma NormOfOneDimensionIsAbs()
  ensures forall v: Vector | |v| == 1 :: L2(v) == Abs(v[0])
{
  reveal L2();
  reveal Sum();
  assert forall v: Vector | |v| == 1 :: Sum(Apply(v, Square)) == v[0] * v[0];
  SqrtOfSquare();
}

/** 
 * If each element in vector v has a lower absolute value than its
 * counterpart in vector u, then ||v|| < ||u||.
 */
lemma MonotonicL2(v: Vector, u: Vector)
  requires |v| == |u|
  requires forall i: int | 0 <= i < |v| :: Abs(v[i]) <= Abs(u[i])
  ensures L2(v) <= L2(u)
{
  reveal L2();
  for i := 0 to |v|
    invariant forall j: int | 0 <= j < i :: Square(v[j]) <= Square(u[j])
  {
    MonotonicSquare(v[i], u[i]);
  }
  MonotonicSum(Apply(v, Square), Apply(u, Square));
  MonotonicSqrt(Sum(Apply(v, Square)), Sum(Apply(u, Square)));
}


lemma PositiveL2()
  ensures forall v: Vector :: L2(v) >= 0.0
{
  reveal L2();
}
}
