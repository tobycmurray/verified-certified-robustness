include "basic_arithmetic.dfy"
include "linear_algebra.dfy"

module OperatorNorms {
import opened BasicArithmetic
import opened LinearAlgebra

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
method GenerateSpecNorms(n: seq<Matrix>, GRAM_ITERATIONS: int) returns (r: seq<real>)
  requires GRAM_ITERATIONS >= 0
  ensures |r| == |n|  
  ensures forall i | 0 <= i < |n| :: IsSpecNormUpperBound(r[i], n[i])
{
  var i := 0;
  r := [];
  while i < |n|
    invariant 0 <= i == |r| <= |n|
    invariant forall j | 0 <= j < i :: IsSpecNormUpperBound(r[j], n[j])
  {
    if DEBUG { print "{ \"debug_msg\": \"Generating spectral norm ", i, " of ", |n|, "...\" },\n"; }
    var specNorm := GramIterationSimple(n[i], GRAM_ITERATIONS);
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


lemma SumPositiveIsZero(v: Vector)
  requires forall i | 0 <= i < |v| :: 0.0 <= v[i]
  requires SumPositive(v) == 0.0
  ensures forall i | 0 <= i < |v| :: v[i] == 0.0
{}

lemma SumPositiveMatrixIsZero(m: Matrix)
  requires forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: 0.0 <= m[i][j]
  requires SumPositiveMatrix(m) == 0.0
  ensures forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
{
  SumPositiveIsZero(m[0]);
}
  
lemma MultIsZero(x: real, y: real)
  requires x * y == 0.0
  ensures x == 0.0 || y == 0.0
{}

lemma MultNonZero(x: real, y: real)
  requires x * y != 0.0
  ensures x != 0.0 && y != 0.0
{}

lemma SquareIsZero(x: real)
  requires Square(x) == 0.0
  ensures x == 0.0
{
  MultIsZero(x,x);
}

lemma ApplySquareIsZero(v: Vector)
  requires forall i | 0 <= i < |v| :: v[i] * v[i] == 0.0
  ensures  forall i | 0 <= i < |v| :: v[i] == 0.0
{
  reveal Apply();
  if |v| == 1 {
    SquareIsZero(v[0]);
  } else {
    SquareIsZero(v[0]);
    ApplySquareIsZero(v[1..]);
  }
}

lemma SquareMatrixElementsSize(m: Matrix, r: Matrix)
  requires r == SquareMatrixElements(m)
  ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
{
  if |m| == 1 {
  } else {
    SquareMatrixElementsSize(m[1..],r[1..]);
  }
}
lemma SquareMatrixElementsElements(m: Matrix, r: Matrix, i: nat, j: nat)
  requires r == SquareMatrixElements(m)
  requires i <= i < |r| && 0 <= j < |r[0]|
  ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
  ensures r[i][j] == m[i][j] * m[i][j]
{
  if |m| == 1 {
  } else {
    if i == 0 {
      SquareMatrixElementsElements(m[1..],r[1..],0,j);
    } else {
      SquareMatrixElementsElements(m[1..],r[1..],i-1,j);
    }
  }
}

lemma SquareMatrixElementsIsZero(m: Matrix, r: Matrix)
  requires r == SquareMatrixElements(m)
  requires forall i,j | 0 <= i < |r| && 0 <= j < |r[0]| :: r[i][j] == 0.0
  ensures forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
{
  SquareMatrixElementsSize(m,r);
  forall i:nat,j:nat | 0 <= i < |m| && 0 <= j < |m[0]| ensures m[i][j] == 0.0 {
    SquareMatrixElementsElements(m,r,i,j);
    SquareIsZero(m[i][j]);
  }
}

function VectorIsZero(v: Vector) : (b: bool)
  ensures b <==> forall i | 0 <= i < |v| :: v[i] == 0.0
{
  if |v| == 1 then v[0] == 0.0 else v[0] == 0.0 && VectorIsZero(v[1..])
}

function MatrixIsZero(m: Matrix) : (b: bool)
  ensures b <==> forall i,j | 0 <= i < |m| && 0 <= j < |m[0]| :: m[i][j] == 0.0
{
  if |m| == 1 then VectorIsZero(m[0]) else VectorIsZero(m[0]) && MatrixIsZero(m[1..])
}

lemma SqrtIsZero(x: real)
  requires x >= 0.0
  requires Sqrt(x) == 0.0
  ensures x == 0.0
{}

lemma FrobeniusNormIsZero(m: Matrix)
  requires FrobeniusNorm(m) == 0.0
  ensures MatrixIsZero(m)
{
  SqrtIsZero(SumPositiveMatrix(SquareMatrixElements(m)));
  SumPositiveMatrixIsZero(SquareMatrixElements(m));
  SquareMatrixElementsIsZero(m,SquareMatrixElements(m));
}

lemma SafeFrobeniusNorm(m: Matrix)
  requires !MatrixIsZero(m)
  ensures FrobeniusNorm(m) != 0.0
{
  forall n:Matrix | FrobeniusNorm(n) == 0.0 ensures MatrixIsZero(n) { FrobeniusNormIsZero(n); }
}



method Truncate(m: Matrix) returns (r: Matrix, e: Matrix)
  requires Rows(m) == Cols(m)
  requires forall i,j | 0 <= i < Rows(m) && 0 <= j < Cols(m) :: m[i][j] == m[j][i]
  ensures SpecNorm(m) <= SpecNorm(r) + SpecNorm(e)
{
  var rM := new real[Rows(m), Rows(m)];
  var eM := new real[Rows(m), Rows(m)];
  var i := 0;
  while i < rM.Length0
    invariant 0 <= i <= Rows(m)
    invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
    invariant forall x,y | 0 <= x < i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
  {
    var j := 0;
    while j < i
      invariant 0 <= i <= Rows(m) && 0 <= j <= i
      invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
      invariant forall x,y | 0 <= x < i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
      invariant forall y | 0 <= y < j :: rM[i,y] == rM[y,i] && eM[i,y] == eM[y,i]
    {
      rM[i,j] := rM[j,i];
      eM[i,j] := eM[j,i];
      j := j + 1;
    }
    assert forall y | 0 <= y < i :: rM[i,y] == rM[y,i];

    while j < rM.Length1
      invariant 0 <= i <= Rows(m) && 0 <= j <= Cols(m)
      invariant forall x,y | 0 <= x < i && 0 <= y < Cols(m) :: rM[x,y] == m[x][y] + eM[x,y]
      invariant forall x,y | 0 <= x <= i && 0 <= y < x :: rM[x,y] == rM[y,x] && eM[x,y] == eM[y,x]
      invariant forall y | 0 <= y < j :: rM[i,y] == m[i][y] + eM[i,y]
    {
      if m[i][j] > 0.0 {
        rM[i,j] := RoundUp(m[i][j]);
      } else if m[i][j] < 0.0 {
        rM[i,j] := RoundDown(m[i][j]);
      } else {
        rM[i,j] := m[i][j];
      }
      eM[i,j] := rM[i,j] - m[i][j];
      j := j + 1;
    }
    i := i + 1;

  }
  // FIXME: these two lines cause Dafny to time out (set --verification-time to 30 to verify this)
  r := A2M(rM);
  e := A2M(eM);
  Assumption4(m,r,e);
}

ghost function Expand(a: seq<(real,real)>, v: real) : (r: real)
  requires forall i | 0 <= i < |a| :: a[i].0 >= 0.0 && a[i].1 >= 0.0
  requires v >= 0.0
{
  if a == [] then v
  else Expand(a[1..],Sqrt((v + a[0].1)*a[0].0))
}

method ExpandImpl(a: seq<(real,real)>, v: real) returns (r: real)
  requires forall i | 0 <= i < |a| :: a[i].0 >= 0.0 && a[i].1 >= 0.0
  requires v >= 0.0
  ensures r >= Expand(a,v)
{
  reveal Expand();
  if a == [] { 
    r := v;
    return;
  }
  else {
    var b := SqrtUpperBound((v + a[0].1)*a[0].0);
    r := ExpandImpl(a[1..],b); 
    assert r >= Expand(a[1..],b);
    ExpandMono(a[1..],Sqrt((v + a[0].1)*a[0].0),b);
  }
}

lemma ExpandMono(a: seq<(real,real)>, v1: real, v2: real)
  requires 0.0 <= v1 <= v2
  requires forall i | 0 <= i < |a| :: a[i].0 >= 0.0 && a[i].1 >= 0.0
  ensures Expand(a,v1) <= Expand(a,v2)
{
  if a == [] {
  } else {
    MonotonicSqrt(a[0].0*(v1 + a[0].1),a[0].0*(v2 + a[0].1));
    ExpandMono(a[1..],Sqrt(a[0].0*(v1 + a[0].1)),Sqrt(a[0].0*(v2 + a[0].1)));
  }
}

lemma ExtendExWellformed(r: real, e: real, ex: seq<(real,real)>)
  requires r >= 0.0 && e >= 0.0
  requires forall i | 0 <= i < |ex| :: ex[i].0 >= 0.0 && ex[i].1 >= 0.0
  ensures forall i | 0 <= i < |[(r,e)]+ex| :: ([(r,e)]+ex)[i].0 >= 0.0 && ([(r,e)]+ex)[i].1 >= 0.0
{}

method Gram(m: Matrix) returns (r: Matrix)
  ensures Sqrt(SpecNorm(r)) >= SpecNorm(m)
  ensures Rows(r) == Cols(r)
  ensures forall i,j | 0 <= i < Rows(r) && 0 <= j < Cols(r) :: r[i][j] == r[j][i]
{
  r := MTM(m);
  Assumption1(m);
}

method Normalise(m: Matrix) returns (r: Matrix, scale_factor: real)
  requires Rows(m) == Cols(m)
  requires forall i,j | 0 <= i < Rows(m) && 0 <= j < Cols(m) :: m[i][j] == m[j][i]
  ensures Rows(r) == Cols(r) && Rows(r) == Rows(m) && Cols(r) == Cols(m)
  ensures scale_factor > 0.0
  ensures forall i,j | 0 <= i < Rows(r) && 0 <= j < Cols(r) :: r[i][j] == m[i][j]/scale_factor
  ensures SpecNorm(m) <= SpecNorm(r)*scale_factor
  ensures forall i,j | 0 <= i < Rows(r) && 0 <= j < Cols(r) :: r[i][j] == r[j][i]  
{
  scale_factor := 1.0;
  if !MatrixIsZero(m) {
    SafeFrobeniusNorm(m);
    scale_factor := FrobeniusNormUpperBound(m);
  }
  r := MatrixDivImpl(m, scale_factor);
  Assumption3(m, scale_factor); // SpecNorm(m) <= SpecNorm(m/scale_factor)*scale_factor
  calc {
    SpecNorm(m)
    <=
    SpecNorm(MatrixDiv(m,scale_factor))*scale_factor
    <=
    SpecNorm(r)*scale_factor;
  }
  assert SpecNorm(m) <= SpecNorm(r) * scale_factor; // WTF Dafny ?!?
}
  

method GramIterationSimple(G: Matrix, GRAM_ITERATIONS: int) returns (s: real)
  requires GRAM_ITERATIONS >= 0
  ensures IsSpecNormUpperBound(s, G)
{
  var i := 0;
  var G' := G;
  var ex: seq<(real,real)> := [];
  while i != GRAM_ITERATIONS
    invariant 0 <= i <= GRAM_ITERATIONS
    invariant forall i | 0 <= i < |ex| :: ex[i].0 >= 0.0 && ex[i].1 >= 0.0
    invariant SpecNorm(G) <= Expand(ex, SpecNorm(G'))
  {
    var G1: Matrix := Gram(G');
    ExpandMono(ex,SpecNorm(G'),Sqrt(SpecNorm(G1)));
    var G2: Matrix, scale_factor: real := Normalise(G1);
    MonotonicSqrt(SpecNorm(G1),SpecNorm(G2)*scale_factor);
    ExpandMono(ex,Sqrt(SpecNorm(G1)),Sqrt(SpecNorm(G2)*scale_factor));
    
    var Trunc: Matrix;
    var E: Matrix;
    Trunc,E := Truncate(G2);

    Assumption2(E);
    var f := FrobeniusNormUpperBound(E);
    MultIsMono(SpecNorm(G2),SpecNorm(Trunc) + f,scale_factor); // SpecNorm(G2)*scale_factor <= (SpecNorm(Trunc) + f)*scale_factor
    ghost var a := SpecNorm(G2)*scale_factor;
    ghost var b := (SpecNorm(Trunc) + f)*scale_factor;
    calc {
      a
      <=
      SpecNorm(G2)*scale_factor
      <=
       (SpecNorm(Trunc) + f)*scale_factor
      <=
      b;
    }
    assert a <= b; // WTF Dafny!?!
    MonotonicSqrt(a,b);
    ExpandMono(ex,Sqrt(a),Sqrt(b));    
    G' := Trunc;
    i := i + 1;
    ExtendExWellformed(scale_factor,f,ex);
    calc {
      SpecNorm(G)
      <=
      Expand(ex, SpecNorm(G'))
      <=
      Expand(ex, Sqrt(SpecNorm(G1)))
      <=
      Expand(ex,Sqrt(SpecNorm(G2)*scale_factor))
      <=
      Expand(ex,a)
      <=
      Expand(ex,b)
      <=
      Expand(ex,(SpecNorm(Trunc) + f)*scale_factor)
      ==
      Expand([(scale_factor,f)]+ex,SpecNorm(G'));
    }
    ghost var ex_old := ex;
    ex := [(scale_factor,f)]+ex;
    calc {
      SpecNorm(G)
      <= 
      Expand([(scale_factor,f)]+ex_old,SpecNorm(G'))
      <=
      Expand(ex,SpecNorm(G'));
    }
    assert SpecNorm(G) == Expand(ex,SpecNorm(G')); // WTF Dafny ?!?
  }
  Assumption2(G');
  ExpandMono(ex,SpecNorm(G'), FrobeniusNorm(G'));
  if DEBUG { print "{ \"debug_msg\": \"Gram iteration computing frobenius norm upper bound...\" },\n"; }
  s := FrobeniusNormUpperBound(G');
  ExpandMono(ex,FrobeniusNorm(G'),s);
  if DEBUG { print "{ \"debug_msg\": \"Gram iteration expanding...\" },\n"; }
  var ret := ExpandImpl(ex,s);
  calc {
    SpecNorm(G)
    <=
    Expand(ex,SpecNorm(G'))
    <=
    FrobeniusNorm(G')
    <=
    s
    <=
    ret;
  }
  SpecNormUpperBoundProperty(ret, G);
  s := ret;
  if DEBUG { print "{ \"debug_msg\": \"Gram iteration done\" },\n"; }  
}


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

lemma {:axiom} Assumption3(m: Matrix, x: real)
  requires 0.0 < x
  ensures SpecNorm(m) <= SpecNorm(MatrixDiv(m, x)) * x

lemma {:axiom} Assumption4(m: Matrix, n: Matrix, e: Matrix)
  requires Rows(m) == Rows(n) == Rows(e) && Cols(m) == Cols(n) == Cols(e) && Rows(n) == Cols(n)
  requires forall i,j | 0 <= i < Rows(m) && 0 <= j < Cols(m) :: m[i][j] == m[j][i] && e[i][j] == e[i][j]
  requires forall i,j | 0 <= i < Rows(n) && 0 <= j < Cols(n) :: n[i][j] == m[i][j] + e[i][j]
  ensures Abs(SpecNorm(n)-SpecNorm(m)) <= SpecNorm(e)
  

ghost function MatrixDiv(m: Matrix, x: real): (r: Matrix)
  requires 0.0 < x
  ensures |r| == |m| && |r[0]| == |m[0]|
  ensures forall i,j | 0 <= i < |r| && 0 <= j < |r[0]| :: r[i][j] == m[i][j]/x
{
  if |m| == 1 then [Apply(m[0], a => a/x)]
  else [Apply(m[0], a => a/x)] + MatrixDiv(m[1..], x)
}

method MatrixDivImpl(m: Matrix, x: real) returns (r: Matrix)
  requires 0.0 < x
  ensures r == MatrixDiv(m, x)
  ensures Rows(r) == Rows(m) && Cols(r) == Cols(m)
  ensures forall i,j | 0 <= i < |r| && 0 <= j < |r[0]| :: r[i][j] == m[i][j]/x
{
  var i := |m| - 1;
  var b := ApplyImpl(m[i], a => a/x);
  r := [b];
  while i > 0
    invariant r == MatrixDiv(m[i..], x)
  {
    i := i - 1;
    b := ApplyImpl(m[i], a => a/x);
    r := [b] + r;
  }
}

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

  // sqrt((v.u)^2) <= sqrt(v.v * u.u)
  assert Sqrt(Square(Dot(v, u))) <= Sqrt(Dot(v, v) * Dot(u, u));

  // |(v.u)| <= sqrt(v.v * u.u)
  assert Abs(Dot(v,u)) <= Sqrt(Dot(v,v) * Dot(u,u)) by {  SqrtOfSquare(); }
  
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
  }

  calc {
    Abs(Dot(v,u));
    <=
    Sqrt(Dot(v,v) * Dot(u,u));
    ==
    Sqrt(Dot(v,v)) * Sqrt(Dot(u,u));
  }

  // |(v.u)| <= sqrt(v.v) * sqrt(u.u)
  assert Abs(Dot(v,u)) <= Sqrt(Dot(v,v)) * Sqrt(Dot(u,u));
  
  L2IsSqrtDot(v);
  L2IsSqrtDot(u);
  assert Sqrt(Dot(v,v)) * Sqrt(Dot(u,u)) == L2(v) * L2(u);
  assert Abs(Dot(v,u)) <= L2(v) * L2(u);
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
