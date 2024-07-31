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

  /** Positive 2^i'th root (abstract). */
  ghost function Power2Root(x: real, i: nat): (r: real)
    requires x >= 0.0
  {
    if i == 0 then x else Sqrt(Power2Root(x, i - 1))
  }

  method Power2RootUpperBound(x: real, i: nat, e: real) returns (r: real)
    requires e > 0.0
    requires x >= 0.0
    ensures r >= Power2Root(x, i)
  {
    var j := i;
    r := x;
    while j > 0
      invariant 0 <= j <= i
      invariant r >= Power2Root(x, i - j)
    {
      MonotonicSqrt(Power2Root(x, i - j), r);
      r := SqrtUpperBound(r, e);
      j := j - 1;
    }
  }

  lemma Power2RootDef(x: real, i: nat)
    requires x >= 0.0
    ensures Power2Root(x, i + 1) == Power2Root(Sqrt(x), i)
  {}

  lemma Power2RootMonotonic(x: real, y: real, i: nat)
    requires 0.0 <= x <= y
    ensures Power2Root(x, i) <= Power2Root(y, i)
  {
    if i != 0 {
      Power2RootMonotonic(x, y, i - 1);
      MonotonicSqrt(Power2Root(x, i - 1), Power2Root(y, i - 1));
    }
  }

  /** Absolute value of the given number. */
  ghost function Abs(x: real): real
  {
    if x >= 0.0 then x else -x
  }

  /** Square of the given number. */
  function Square(x: real): (r: real)
    ensures r >= 0.0
  {
    x * x
  }

  /** 
   * The Sqrt function used in this file returns the positive square root.
   * Therefore, Sqrt(x^2) == |x|.
   */
  lemma SqrtOfSquare()
    ensures forall x: real :: Sqrt(Square(x)) == Abs(x)
  {
    assert forall x: real ::
      Sqrt(Square(x)) * Sqrt(Square(x)) == Abs(x) * Abs(x);
    forall x: real {
      PositiveSquaresEquality(Sqrt(Square(x)), Abs(x));
    }
  }

  /** For non-negative reals x and y, if x^2 == y^2 then x == y. */
  lemma PositiveSquaresEquality(x: real, y: real)
    requires x >= 0.0 && y >= 0.0
    requires x * x == y * y
    ensures x == y
  {
    if x > y {
      IncreaseSquare(x, y);
    } else if x < y {
      IncreaseSquare(y, x);
    }
  }

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

  /** For non-negative reals x and y, if x <= y then sqrt(x) <= sqrt(y). */
  lemma MonotonicSqrt(x: real, y: real)
    requires 0.0 <= x <= y
    ensures Sqrt(x) <= Sqrt(y)
  {
    if Sqrt(x) > Sqrt(y) {
      // Then sqrt(x) * sqrt(x) > sqrt(y) * sqrt(y)
      IncreaseSquare(Sqrt(x), Sqrt(y));
      // Hence x > y, which is a contradiction
      assert false;
    }
  }

  /** For reals x and y where 0 <= y, if x^2 <= y^2 then x <= y. */
  lemma MonotonicSqrt2(x: real, y: real)
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

  /** For any real number x, if x * x == 0 then x == 0. */
  lemma Zero(x: real)
    requires x * x == 0.0
    ensures x == 0.0
  {
    // Assume the conclusion is false
    if x != 0.0 {
      // Case 1: x > 0
      if x > 0.0 {
        // Then x * x > x * 0
        Increase(x, 0.0, x);
        // This violates the requires clause
        assert false;
      // Case 2: x < 0
      } else if x < 0.0 {
        // Then -x * -x > -x * 0
        Increase(-x, 0.0, -x);
        // This violates the requires clause
        assert false;
      }
    }
  }

  /** For non-negative reals x and y, if y < x then y^2 < x^2. */
  lemma IncreaseSquare(x: real, y: real)
    requires 0.0 <= y < x
    ensures y * y < x * x
  {
    if y == 0.0 {
      // x * x > x * 0
      Increase(x, 0.0, x);
      // Since x * 0 == 0 == y * y, we have x * x > y * y
    } else {
      // x * x > x * y
      Increase(x, y, x);
      // x * y > y * y
      Increase(x, y, y);
      // Thus x * x > y * y by transitivity of '>'
    }
  }

  /** For any non-negative reals x and y, x^2 <= y^2. */
  lemma MonotonicSquarePositive(x: real, y: real)
    requires 0.0 <= x <= y
    ensures Square(x) <= Square(y)
  {
    assert 0.0 <= y;
  }

  /** For any real number x, we have |x|^2 == x^2. */
  lemma AbsoluteSquare(x: real)
    ensures Square(Abs(x)) == Square(x)
  {}

  /** For any real numbers x and y, if |x| <= |y| then x^2 <= y^2. */
  lemma MonotonicSquare(x: real, y: real)
    requires Abs(x) <= Abs(y)
    ensures Square(x) <= Square(y)
  {
    // |x| <= |y|
    MonotonicSquarePositive(Abs(x), Abs(y));
    // 1: |x|^2 <= |y|^2
    AbsoluteSquare(x);
    // 2: |x|^2 == x^2
    AbsoluteSquare(y);
    // 3: |y|^2 == y^2
    // From 1, 2, 3: x^2 <= y^2
  }

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
        MonotonicSqrt2(Sqrt(x), r - e);
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
      MonotonicSqrt2(r - e, Sqrt(x));
    }
  }
}
