module BasicArithmetic {

  /** The ReLu activation function. */
  ghost opaque function Relu(x: real): (r: real)
    ensures x >= 0.0 ==> r == x
    ensures x < 0.0 ==> r == 0.0
  {
    if x >= 0.0 then x else 0.0
  }

  /** Positive square root (abstract). */
  ghost opaque function {:axiom} Sqrt(x: real): (r: real)
    requires x >= 0.0
    ensures r >= 0.0
    ensures r * r == x

  /** Positive 2^i'th root (abstract). */
  ghost function Power2Root(x: real, i: nat): (r: real)
    requires x >= 0.0
  {
    if i == 0 then x else Sqrt(Power2Root(x, i - 1))
  }

  /** Computes an upper bound on the positive square root of x. */
  method SqrtUpperBound(a: real) returns (r: real)
    requires a >= 0.0
    ensures r >= Sqrt(a)
  {
    r := 0.5 * (a + 1.0);
    var i := 1;
    while i < 10 // fixme: we'd like to stop when r * r <= a + e for some e
      invariant r >= Sqrt(a) && r > 0.0
    {
      assert Square((r - Sqrt(a))) >= 0.0;
      r := 0.5 * (r + a / r);
      i := i + 1;
    }
  }

  /** Computes an upper bound on the positive square root of x. */
  method Power2RootUpperBound(x: real, i: nat) returns (r: real)
    requires x >= 0.0
    ensures r >= Power2Root(x, i)
  {
    r := 1.0;
    assume r >= Power2Root(x, i); // todo
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
}
