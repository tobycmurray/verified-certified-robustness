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
  // if |v| == 1 then [v[0] - u[0]] else Minus(v[..|v|-1], u[..|u|-1]) + [v[|v|-1] - u[|u|-1]]
  if |v| == 1 then [v[0] - u[0]] else [v[0] - u[0]] + Minus(v[1..], u[1..])
}

/** Dot product. */
ghost opaque function Dot(v: Vector, u: Vector): real
  requires |v| == |u|
{
  // if |v| == 1 then v[0] * u[0] else Dot(v[..|v|-1], u[..|u|-1]) + v[|v|-1] * u[|u|-1]
  if |v| == 1 then v[0] * u[0] else v[0] * u[0] + Dot(v[1..], u[1..])
}

/** Matrix vector product. */
ghost opaque function MV(m: Matrix, v: Vector): (r: Vector)
  requires |m[0]| == |v|
  ensures |r| == |m|
  ensures forall i: int :: 0 <= i < |r| ==> r[i] == Dot(m[i], v)
{
  // if |m| == 1 then [Dot(m[0], v)] else MV(m[..|m|-1], v) + [Dot(m[|m|-1], v)]
  if |m| == 1 then [Dot(m[0], v)] else [Dot(m[0], v)] + MV(m[1..], v)
}

/** Matrix-matrix multiplication. */
ghost function MM(m: Matrix, n: Matrix): Matrix
  requires |m[0]| == |n|
{
  if |m| == 1 then [MMGetRow(m[0], n, 0)]
  else [MMGetRow(m[0], n, 0)] + MM(m[1..], n)
}

ghost function MMGetRow(v: Vector, n: Matrix, x: nat): (r: Vector)
  requires x < |n[0]|
  requires |v| == |n|
  ensures |r| == |n[0]| - x
  decreases |n[0]| - x
{
  if x == |n[0]| - 1 then [Dot(v, Column(x, n))]
  else [Dot(v, Column(x, n))] + MMGetRow(v, n, x + 1)
}

ghost function Column(x: nat, m: Matrix): (r: Vector)
  requires x < |m[0]|
  ensures |r| == |m|
{
  if |m| == 1 then [m[0][x]] else [m[0][x]] + Column(x, m[1..])
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

ghost function SumPositiveMatrix(m: Matrix): (r: real)
  requires forall i, j | 0 <= i < |m| && 0 <= j < |m[0]| :: 0.0 <= m[i][j]
  ensures 0.0 <= r
{
  if |m| == 1 then SumPositive(m[0])
  else SumPositive(m[0]) + SumPositiveMatrix(m[1..])
}

ghost function SumPositive(v: Vector): (r: real)
  requires forall i | 0 <= i < |v| :: 0.0 <= v[i]
  ensures 0.0 <= r
{
  if |v| == 1 then v[0] else v[0] + SumPositive(v[1..])
}

ghost function SquareMatrixElements(m: Matrix): (r: Matrix)
  ensures forall i, j | 0 <= i < |r| && 0 <= j < |r[0]| :: 0.0 <= r[i][j]
{
  if |m| == 1 then [Apply(m[0], Square)]
  else [Apply(m[0], Square)] + SquareMatrixElements(m[1..])
}

ghost function Transpose(m: Matrix, x: nat := 0): (r: Matrix)
  requires x < |m[0]|
  decreases |m[0]| - x
{
  if x == |m[0]| - 1 then [Column(x, m)]
  else [Column(x, m)] + Transpose(m, x + 1)
}

/* =========================== Concrete Functions =========================== */

function Rows(m: Matrix): nat {
  |m|
}

function Cols(m: Matrix): nat {
  |m[0]|
}

/* ================================ Methods ================================= */

method MinusImpl(v: Vector, u: Vector) returns (r: Vector)
  requires |v| == |u|
  ensures r == Minus(v, u)
{
  reveal Minus();
  var i := |v| - 1;
  r := [v[i] - u[i]];
  while i > 0
    invariant r == Minus(v[i..], u[i..])
  {
    i := i - 1;
    r := [v[i] - u[i]] + r;
  }
}

method DotImpl(v: Vector, u: Vector) returns (r: real)
  requires |v| == |u|
  ensures r == Dot(v, u)
{
  reveal Dot();
  var i := |v| - 1;
  r := v[i] * u[i];
  while i > 0
    invariant r == Dot(v[i..], u[i..])
  {
    i := i - 1;
    r := r + v[i] * u[i];
  }
}

method MVImpl(m: Matrix, v: Vector) returns (r: Vector)
  requires |m[0]| == |v|
  ensures r == MV(m, v)
{
  reveal MV();
  var i := |m| - 1;
  var x := DotImpl(m[i], v);
  r := [x];
  while i > 0
    invariant r == MV(m[i..], v)
  {
    i := i - 1;
    x := DotImpl(m[i], v);
    r := [x] + r;
  }
}

/** Matrix-matrix multiplication. */
// ghost function MM(m: Matrix, n: Matrix): Matrix
//   requires |m[0]| == |n|
// {
//   if |m| == 1 then [MMGetRow(m[0], n)]
//   else [MMGetRow(m[0], n)] + MM(m[1..], n)
// }

// ghost function MMGetRow(v: Vector, n: Matrix): (r: Vector)
//   requires |v| == |n|
//   ensures |r| == |n[0]|
//   decreases |n[0]|
// {
//   if |n[0]| == 1 then [Dot(v, GetFirstColumn(n))]
//   else [Dot(v, GetFirstColumn(n))] + MMGetRow(v, RemoveFirstColumn(n))
// }

method MMImpl(m: Matrix, n: Matrix) returns (r: Matrix)
  requires Cols(m) == Rows(n)
  ensures r == MM(m, n)
{
  var i := |m| - 1;
  var v: Vector := MMGetRowImpl(m[i], n);
  r := [v];
  while i > 0
    invariant r == MM(m[i..], n)
  {
    i := i - 1;
    v := MMGetRowImpl(m[i], n);
    r := [v] + r;
  }
}

/*
  if x == |n[0]| - 1 then [Dot(v, Column(n[x]))]
  else [Dot(v, Column(n[x]))] + MMGetRow(v, n, x + 1)
*/

method MMGetRowImpl(v: Vector, n: Matrix) returns (r: Vector)
  requires |v| == |n|
  ensures r == MMGetRow(v, n, 0)
{
  var i := |n[0]| - 1;
  var c := ColumnImpl(i, n);
  assert c == Column(i, n);
  var x := DotImpl(v, c);
  r := [x];
  while i > 0
    invariant 0 <= i < |n[0]|
    invariant r == MMGetRow(v, n, i)
  {
    i := i - 1;
    c := ColumnImpl(i, n);
    assert c == Column(i, n);
    x := DotImpl(v, c);
    r := [x] + r;
  }
}

/*
if |v| == 1 then [f(v[0])] else [f(v[0])] + Apply(v[1..], f)
*/

method ApplyImpl(v: Vector, f: real -> real) returns (r: Vector)
  ensures r == Apply(v, f)
{
  var i := |v| - 1;
  r := [f(v[i])];
  while i > 0
    invariant r == Apply(v[i..], f)
  {
    i := i - 1;
    r := [f(v[i])] + r;
  }
}

method SquareMatrixElementsImpl(m: Matrix) returns (r: Matrix)
  ensures r == SquareMatrixElements(m)
{
  var i := |m| - 1;
  var v := ApplyImpl(m[i], Square);
  r := [v];
  while i > 0
    invariant r == SquareMatrixElements(m[i..])
  {
    i := i - 1;
    v := ApplyImpl(m[i], Square);
    r := [v] + r;
  }
}

method SumPositiveMatrixImpl(m: Matrix) returns (r: real)
  requires forall i, j | 0 <= i < |m| && 0 <= j < |m[0]| :: 0.0 <= m[i][j]
  ensures r == SumPositiveMatrix(m)
{
  var i := |m| - 1;
  var x := SumPositiveImpl(m[i]);
  r := x;
  while i > 0
    invariant r == SumPositiveMatrix(m[i..])
  {
    i := i - 1;
    x := SumPositiveImpl(m[i]);
    r := x + r;
  }
}

method ColumnImpl(x: nat, m: Matrix) returns (r: Vector)
  requires x < Cols(m)
  ensures r == Column(x, m)
{
  var i := |m| - 1;
  r := [m[i][x]];
  while i > 0
    invariant r == Column(x, m[i..])
  {
    i := i - 1;
    r := [m[i][x]] + r;
  }
}

method TransposeImpl(m: Matrix) returns (r: Matrix)
  ensures r == Transpose(m)
{
  var i := |m[0]| - 1;
  var c := ColumnImpl(i, m);
  r := [c];
  while i > 0
    invariant 0 <= i < |m[0]|
    invariant r == Transpose(m, i)
  {
    i := i - 1;
    c := ColumnImpl(i, m);
    r := [c] + r;
  }
}

method SumPositiveImpl(v: Vector) returns (r: real)
  requires forall i | 0 <= i < |v| :: 0.0 <= v[i]
  ensures r == SumPositive(v)
{
  var i := |v| - 1;
  r := v[i];
  while i > 0
    invariant r == SumPositive(v[i..])
  {
    i := i - 1;
    r := v[i] + r;
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
