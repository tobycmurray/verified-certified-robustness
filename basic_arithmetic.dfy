module BasicArithmetic {

  /** The ReLu activation function. */
  ghost opaque function Relu(x: real): (r: real)
    ensures x >= 0.0 ==> r == x
    ensures x < 0.0 ==> r == 0.0
  {
    if x >= 0.0 then x else 0.0
  }

  /** Positive square root (abstract). */
  ghost opaque function Sqrt(x: real): (r: real)
    requires x >= 0.0
    ensures {:axiom} r >= 0.0
    ensures {:axiom} r * r == x

  /** The square root of 0 is 0. */
  lemma SqrtZeroIsZero()
    ensures Sqrt(0.0) == 0.0
  {
    if Sqrt(0.0) != 0.0 {
      calc {
        0.0;
        < Sqrt(0.0) * Sqrt(0.0);
        == 0.0;
      }
    }
  }

  /** For reals x and y where 0 <= y, if x^2 <= y^2 then x <= y. */
  lemma MonotonicSqrt(x: real, y: real)
    requires 0.0 <= y
    requires x * x <= y * y
    ensures x <= y
  {
    if x > y {
      if y == 0.0 {
        Increase(x, 0.0, x);
      } else {
        Increase(x, y, x);
        Increase(x, y, y);
      }
      assert false;
    }
  }

  /** For reals x and y, and some real z > 0, if x > y then x * z > y * z. */
  lemma Increase(x: real, y: real, z: real)
    requires z > 0.0
    requires x > y
    ensures x * z > y * z
  {}

  lemma SmallerDenominator(x: real, y: real, z: real)
    requires 0.0 <= x
    requires 0.0 < y <= z
    ensures x / z <= x / y
  {}

  lemma PositiveSquare(x: real)
    ensures 0.0 <= x * x
  {}

  method SqrtUpperBound(x: real, e: real) returns (r: real)
    requires x >= 0.0
    requires e > 0.0
    ensures Sqrt(x) <= r <= Sqrt(x) + e
  {
    if x == 0.0 {
      SqrtZeroIsZero();
      return 0.0;
    }
    r := if x < 1.0 then 1.0 else x;
    while r >= e && (r - e) * (r - e) > x
      invariant 0.0 < Sqrt(x) <= r
      decreases 2.0 / e * r
    {
      ghost var R := r;

      // lower bound on R, necessary for termination proof
      assert R >= Sqrt(x) + e by {
        assert (r - e) * (r - e) > x;
        PositiveSquare(r - e);
        MonotonicSqrt(Sqrt(x), r - e);
        assert r - e >= Sqrt(x);
        assert R >= Sqrt(x) + e;
      }

      // proof for the update of r
      assert Sqrt(x) <= 0.5 * (r + x / r) by {
        PositiveSquare(r - Sqrt(x));
        assert 0.0 <= (r - Sqrt(x)) * (r - Sqrt(x)); // 0.0 <= any square
        assert 0.0 <= r * r - 2.0 * r * Sqrt(x) + x; // distribute
        assert 0.0 <= (r * r - 2.0 * r * Sqrt(x) + x) / r; // divide by r
        assert 0.0 <=  r     - 2.0 * Sqrt(x)     + x / r; // simplify
        assert 2.0 * Sqrt(x) <= r + x / r; // rearrange
        assert Sqrt(x) <= 0.5 * (r + x / r); // divide by 2
      }
      
      // update r
      r := 0.5 * (r + x / r);
      assert Sqrt(x) <= r;
      
      // termination
      assert 2.0 / e * R - 2.0 / e * r >= 1.0 by {
        calc >= {
          R - r;
          calc {
            r;
            (R + x / R) / 2.0;
          }
          R - (R + x / R) / 2.0;
          { SmallerDenominator(x, Sqrt(x), R); }
          R - (R + x / Sqrt(x)) / 2.0;
          (R - Sqrt(x)) / 2.0;
          (Sqrt(x) + e - Sqrt(x)) / 2.0;
          e / 2.0;
        }
        assert R - r >= e / 2.0;
        assert 2.0 * R - 2.0 * r >= e;
        assert 2.0 / e * R - 2.0 / e * r >= 1.0;
      }
    }
    if (r >= e) {
      PositiveSquare(r - e);
      MonotonicSqrt(r - e, Sqrt(x));
    }
  }
}
