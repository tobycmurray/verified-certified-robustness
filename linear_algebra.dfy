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
ghost opaque function Dot(v: Vector, u: Vector): (r: real)
  requires |v| == |u|
  ensures v == u ==> r >= 0.0
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

function ArrayRow(M: array2<real>, acc: seq<real>, row: nat) : (r: Vector)
  reads M
  requires M.Length0 > 0 && M.Length1 > 0
  requires row < M.Length0 && |acc| <= M.Length1
  requires forall j | 0 <= j < |acc| :: acc[j] == M[row,j]
  ensures |r| == M.Length1
  ensures forall j | 0 <= j < |r| :: r[j] == M[row,j]
  decreases M.Length1 - |acc|
{
  if |acc| == M.Length1 then acc
  else ArrayRow(M, acc+[M[row,|acc|]], row)
}

/* Machinery to convert between Matrices as Dafny sequences, to two-dimensional arrays */
function ArrayToMatrix(M: array2<real>, acc: seq<seq<real>>) : (r: Matrix)
  reads M
  requires M.Length0 > 0 && M.Length1 > 0
  requires |acc| <= M.Length0 && (forall j | 0 <= j < |acc| :: |acc[j]| == M.Length1)
  ensures Rows(r) == M.Length0 && Cols(r) == M.Length1
  requires forall i,j | 0 <= i < |acc| && 0 <= j < M.Length1 :: acc[i][j] == M[i,j]
  ensures forall i,j | 0 <= i < M.Length0 && 0 <= j < M.Length1 :: r[i][j] == M[i,j]
  decreases M.Length0 - |acc|
{
  if |acc| == M.Length0 then acc
  else ArrayToMatrix(M, acc+[ArrayRow(M, [], |acc|)])
}

function A2M(M: array2<real>) : (r: Matrix)
  reads M
  requires M.Length0 > 0 && M.Length1 > 0
  ensures Rows(r) == M.Length0 && Cols(r) == M.Length1
  ensures forall i,j | 0 <= i < M.Length0 && 0 <= j < M.Length1 :: r[i][j] == M[i,j]
{
  ArrayToMatrix(M, [])
}

method M2A(M: Matrix, a: array2<real>)
  modifies a
  requires a.Length0 == Rows(M)
  requires a.Length1 == Cols(M)
  ensures forall i,j | 0 <= i < Rows(M) && 0 <= j < Cols(M) :: a[i, j] == M[i][j]
{
  var row := 0;
  while row < Rows(M)
    invariant 0 <= row && row <= Rows(M)
    invariant forall i | 0 <= i < row :: (forall j | 0 <= j < Cols(M) :: a[i, j] == M[i][j]) 
  {
    var col := 0;
    while col < Cols(M)
      invariant 0 <= row && row <= Rows(M)
      invariant 0 <= col && col <= Cols(M)
      invariant forall i,j | 0 <= i < row && 0 <= j < Cols(M) :: a[i, j] == M[i][j]
      invariant forall j | 0 <= j < col :: a[row, j] == M[row][j]
    {
      a[row, col] := M[row][col];
      col := col + 1;
    }
    row := row + 1;
  }
  assert forall i,j | 0 <= i < Rows(M) && 0 <= j < Cols(M) :: a[i, j] == M[i][j];
}

lemma MatrixEquality(M1: Matrix, M2: Matrix)
  requires Rows(M1) == Rows(M2)
  requires Cols(M1) == Cols(M2)
  requires forall i,j | 0 <= i < Rows(M1) && 0 <= j < Cols(M1) :: M1[i][j] == M2[i][j]
  ensures M1 == M2
{
  assert |M1| == |M2|;
  assert forall i | 0 <= i < |M1| :: |M1[i]| == |M2[i]|;
  assert forall i | 0 <= i < |M1| :: (forall j | 0 <= j < |M1[i]| :: M1[i][j] == M2[i][j]);
  assert forall i | 0 <= i < |M1| :: M1[i] == M2[i];
  assert M1 == M2;
}

// TODO: remove this?
method Test(M: Matrix) returns (r: Matrix)
  ensures r == M 
{
  var a := new real[Rows(M), Cols(M)];
  M2A(M, a);
  assert forall i,j | 0 <= i < Rows(M) && 0 <= j < Cols(M) :: a[i, j] == M[i][j];
  r := A2M(a);
  MatrixEquality(r,M);
}

/* lemmas to reason about the behaviour of MM(Transpose(M),M)
 *
 * first we need a bunch of lemmas about the basic functions
 */
lemma ColumnCorrect(M: Matrix, col: nat, i: nat)
  requires col < Cols(M)
  requires i < Rows(M)
  ensures Column(col,M)[i] == M[i][col]
{
  if |M| > 1 && i > 0 {
    ColumnCorrect(M[1..], col, i-1);
  }
}

lemma MMGetRowElements(v: Vector, M2: Matrix, col: nat, i: nat)
  requires col < Cols(M2)
  requires |v| == Rows(M2)
  requires i < Cols(M2) - col
  ensures MMGetRow(v, M2, col)[i] == Dot(v, Column(i+col, M2))
  decreases i
{
  if col == Cols(M2)-1 {
    return;
  } else {
    if i == 0 {
      return;
    } else {
      assert i > 0;
      MMGetRowElements(v, M2, col+1, i-1);
      return;
    }
  }
}

/* each element of matrix multiplication is the corresponding dot product */
lemma MMElements(M1: Matrix, M2: Matrix, M3: Matrix, i: nat, j:nat)
  requires Cols(M1) == Rows(M2)
  requires M3 == MM(M1,M2)
  requires i < Rows(M1) && j < Cols(M2)
  ensures Rows(M3) == Rows(M1)
  ensures Cols(M3) == Cols(M2)
  ensures M3[i][j] == Dot(M1[i],Column(j,M2))
{
  if |M1| == 1 {
    MMGetRowElements(M1[0], M2, 0, j);
  } else {
    MMGetRowElements(M1[0], M2, 0, j);
    if i > 0 {
      MMElements(M1[1..], M2, M3[1..], i-1, j);
    } else {
      // need a recursive call to prove the Rows conclusion from the induction hypothesis
      MMElements(M1[1..], M2, M3[1..], 0, 0);
      return;
    }
  }
}

/* each row of a transposed matrix is the corresponding column of the original matrix */
lemma TransposeTransposes(M: Matrix, i: nat, j: nat)
  requires i < Cols(M) - j
  requires j < Cols(M)
  ensures Rows(Transpose(M,j)) == Cols(M) - j
  ensures Cols(Transpose(M,j)) == Rows(M)
  ensures Transpose(M,j)[i] == Column(i+j, M)
  decreases Cols(M) - j
{
  if j == Cols(M) - 1 {
  } else {
    if i == 0 {
    } else {
      TransposeTransposes(M,i-1,j+1);
    }
  }
}

/* finally we characterise each element of MM(Transpose(M),M) as the dot
 * product of the corresponding column vectors of M 
 */
lemma MTMElement(M: Matrix, i: nat, j: nat)
  requires i < Cols(M) && j < Cols(M)
  ensures Rows(MM(Transpose(M),M)) == Cols(M)
  ensures Cols(MM(Transpose(M),M)) == Cols(M)
  ensures MM(Transpose(M),M)[i][j] == Dot(Column(i,M),Column(j,M))
{
  TransposeTransposes(M, i, 0);
  MMElements(Transpose(M),M,MM(Transpose(M),M),i,j); 
}

/* dot product is symmetric */
lemma DotSym(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Dot(v,u) == Dot(u,v)
{
  reveal Dot();
  if |v| == 1 {
  } else {
    DotSym(v[1..], u[1..]);
  }
}

/* therefore, MM(Transpose(M),M) is a symmetric matrix.
 * later, we can take advantage of this to save having to recompute entries already computed
 */
lemma MTMSym(M: Matrix, i: nat, j: nat)
  requires i < Cols(M) && j < Cols(M)
  ensures Rows(MM(Transpose(M),M)) == Cols(M)
  ensures Cols(MM(Transpose(M),M)) == Cols(M)
  ensures MM(Transpose(M),M)[i][j] == MM(Transpose(M),M)[j][i]
{
  MTMElement(M, i, j);
  MTMElement(M, j, i);
  DotSym(Column(i,M), Column(j,M));
}

lemma DotColumnsInit(M: Matrix, i: nat, j: nat, k: nat, dot: real)
  requires i < Cols(M) && j < Cols(M) && k == Rows(M)-1
  requires dot == M[k][i] * M[k][j]
  ensures dot == Dot(Column(i,M[k..]), Column(j,M[k..]))
{
  reveal Dot();
}

lemma DotColumnsInductive(M: Matrix, i: nat, j: nat, k: nat, dot: real)
  requires i < Cols(M) && j < Cols(M) && 0 < k < Rows(M)
  requires dot == Dot(Column(i,M[k..]), Column(j,M[k..]))
  ensures dot + M[k-1][i] * M[k-1][j] == Dot(Column(i,M[k-1..]), Column(j,M[k-1..]))
{
  reveal Dot(); 
}

lemma ComputedMTM(r: Matrix, M: Matrix)
  requires Rows(r) == Cols(M)
  requires Cols(r) == Cols(M)
  requires (forall y | 0 <= y < Cols(M) :: forall x | y <= x < Cols(M) :: r[y][x] == Dot(Column(y,M),Column(x,M)) && r[x][y] == r[y][x])
  ensures r == MM(Transpose(M),M)
{
  assert Rows(r) == Rows(MM(Transpose(M),M)) by { MTMElement(M,0,0); } 
  forall i,j | 0 <= i < Rows(r) && 0 <= j < Cols(r) 
    ensures r[i][j] == MM(Transpose(M),M)[i][j] {
    if j >= i {
      assert r[i][j] == Dot(Column(i,M),Column(j,M));
      MTMElement(M, i, j);
      assert r[i][j] == MM(Transpose(M),M)[i][j];
    } else {
      assert r[j][i] == Dot(Column(j,M),Column(i,M));
      assert r[i][j] == Dot(Column(j,M),Column(i,M));
      DotSym(Column(j,M),Column(i,M));
      MTMElement(M, i, j);
    }
  }
  MatrixEquality(r,MM(Transpose(M),M));
}

method MTM(M: Matrix) returns (r: Matrix)
  ensures r == MM(Transpose(M),M)
{
  var aM := new real[Rows(M), Cols(M)];
  M2A(M, aM);
  var rM := new real[Cols(M), Cols(M)]; // results go in here
  var i: nat := 0;
  while i < rM.Length0
    invariant i >= 0 && i <= Cols(M) &&
              (forall x,y | 0 <= x < Rows(M) && 0 <= y < Cols(M) :: aM[x,y] == M[x][y]) &&
              (forall y | 0 <= y < i :: forall x | y <= x < Cols(M) :: rM[y,x] == Dot(Column(y,M),Column(x,M))) &&
              (forall y,x | 0 <= y < i && 0 <= x < y :: rM[y,x] == rM[x,y])
  {
    print "{ \"debug_msg\": \"MTM outer loop, i: ", i, " of : ", Cols(M), "\" },\n";

    // copy the existing elements over, ensuring we write each row sequentially to maximise cache hit rate
    var c := 0;
    while c < i
      invariant c <= i && i >= 0 && i <= Cols(M) &&
                (forall x | 0 <= x < c :: rM[i,x] == rM[x,i]) &&
                (forall x,y | 0 <= x < Rows(M) && 0 <= y < Cols(M) :: aM[x,y] == M[x][y]) &&
                (forall y | 0 <= y < i :: forall x | y <= x < Cols(M) :: rM[y,x] == Dot(Column(y,M),Column(x,M))) &&
                (forall y,x | 0 <= y < i && 0 <= x < y :: rM[y,x] == rM[x,y])
    {
      rM[i,c] := rM[c,i];
      c := c + 1;
    }

    // compute the new elements, again ensuring we write sequentially to each row
    var j: nat := i;
    while j < rM.Length1 
      invariant i >= 0 && i <= Cols(M) && 
                j >= i && j <= Cols(M) && 
                (forall y,x | 0 <= y < i && 0 <= x < y :: rM[y,x] == rM[x,y]) &&
                (forall y | 0 <= y < i :: forall x | y <= x < Cols(M) :: rM[y,x] == Dot(Column(y,M),Column(x,M))) &&
                (forall x | i <= x < j :: rM[i,x] == Dot(Column(i,M),Column(x,M))) &&
                (forall x | 0 <= x < i :: rM[i,x] == rM[x,i]) &&
                (forall x,y | 0 <= x < Rows(M) && 0 <= y < Cols(M) :: aM[x,y] == M[x][y])
    {
      var k: nat := aM.Length0-1;
      var dot: real := aM[k,i] * aM[k,j];
      DotColumnsInit(M,i,j,k,dot);
      while k != 0 
        invariant k < Rows(M) && k >= 0 && dot == Dot(Column(i,M[k..]),Column(j,M[k..])) && 
                  (forall x,y | 0 <= x < Rows(M) && 0 <= y < Cols(M) :: aM[x,y] == M[x][y]) &&
                  (forall y,x | 0 <= y < i && 0 <= x < y :: rM[y,x] == rM[x,y]) &&
                  (forall y | 0 <= y < i :: forall x | y <= x < Cols(M) :: rM[y,x] == Dot(Column(y,M),Column(x,M))) &&
                  (forall x | i <= x < j :: rM[i,x] == Dot(Column(i,M),Column(x,M))) &&
                  (forall x | 0 <= x < i :: rM[i,x] == rM[x,i])
        decreases k
      {
        DotColumnsInductive(M,i,j,k,dot);
        k := k - 1;
        dot := dot + aM[k,i] * aM[k,j];
        //assert dot == Dot(Column(i,M[k..]),Column(j,M[k..]));
      }
      assert dot == Dot(Column(i,M), Column(j,M));
      rM[i,j] := dot;
      j := j + 1;
    }
    i := i + 1;
  }
  r := A2M(rM);
  assert Rows(r) == Cols(M) && Cols(r) == Cols(M);
  ComputedMTM(r, M);
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
    MinusIsDistributive(u, w);
    assert Minus(u, w)[1..] == Minus(u[1..], w[1..]);
  }
}

lemma MinusIsDistributive(v: Vector, u: Vector)
  requires 1 < |v| == |u|
  ensures Minus(v, u)[1..] == Minus(v[1..], u[1..])
{}

lemma DotIsCommutative(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Dot(v, u) == Dot(u, v)
{
  reveal Dot();
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

// Toby's work =================================================================

lemma MVIsDot(v: Vector, u: Vector)
  requires |v| == |u|
  ensures MV([v], u) == [Dot(v, u)]
{
}

lemma L2MVIsAbsDot(v: Vector, u: Vector)
  requires |v| == |u|
  ensures L2(MV([v], u)) == Abs(Dot(v, u))
{
  MVIsDot(v, u);
  NormOfOneDimensionIsAbs();
}

lemma PlusNonNegative(a: real, b: real)
  requires a >= 0.0
  requires b >= 0.0
  ensures a + b >= 0.0
{
}

lemma DotSelfIsNonNegative(v: Vector)
  ensures Dot(v, v) >= 0.0
{
  reveal Dot();

  if |v| == 1 {
  } else {
    DotSelfIsNonNegative(v[1..]);
    PlusNonNegative(v[0] * v[0], Dot(v[1..], v[1..]));
  } 
}

lemma L2IsSqrtDot(v: Vector)
  ensures L2(v) == Sqrt(Dot(v, v))
{
  DotIsSumApplySquare(v);
  assert Dot(v, v) == Sum(Apply(v, Square));
  reveal L2();
}

lemma DotIsSumApplySquare(v: Vector)
  ensures Dot(v, v) == Sum(Apply(v, Square))
{
  reveal Dot();
  reveal Sum();
  reveal Apply();
  reveal Square();
  if (|v| == 1) {
  } else {
    calc {
      Dot(v, v);
      ==
      v[0] * v[0] + Dot(v[1..], v[1..]);
      ==
      {
        DotIsSumApplySquare(v[1..]);
      }
      v[0] * v[0] + Sum(Apply(v[1..], Square));
      ==
      Square(v[0]) + Sum(Apply(v[1..], Square));
      ==
      Apply([v[0]], Square)[0] + Sum(Apply(v[1..], Square));
      ==
      calc {
        Apply([v[0]], Square)[0];
        ==
        Apply(v, Square)[0];
      }
      Apply(v, Square)[0] + Sum(Apply(v[1..], Square));
      ==
      calc {
        Sum(Apply(v[1..], Square));
        ==
        Sum(Apply(v, Square)[1..]);
      }
      Apply(v, Square)[0] + Sum(Apply(v, Square)[1..]);
      ==
      {
        ReverseSum(Apply(v, Square));
      }
      Sum(Apply(v, Square));
    }
  }
}

}
