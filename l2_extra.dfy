include "basic_arithmetic.dfy"
include "operator_norms.dfy"
include "linear_algebra.dfy"

module L2Extra {
import opened BasicArithmetic
import opened LinearAlgebra
import opened OperatorNorms

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