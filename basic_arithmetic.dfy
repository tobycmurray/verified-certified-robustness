module BasicArithmetic {

// number of decimal places to round reals to, for efficiency purposes
const ROUNDING_PRECISION := 16
// maximum number of iterations to run the square-root algorithm for
const SQRT_ITERATIONS := 20000
// satisfactory error margin for square roots, to optimise the algorithm
const SQRT_ERR_MARGIN := 0.0000001
// print debug messages
const DEBUG := true

/* ============================ Ghost Functions ============================= */

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

/** Absolute value of the given number. */
ghost function Abs(x: real): real
{
  if x >= 0.0 then x else -x
}

/** Represents x^y. */
ghost function Pow(x: real, y: nat): (r: real)
  ensures x != 0.0 ==> r != 0.0
{
  if y == 0 then 1.0 else x * Pow(x, y - 1)
}

/** Product of all vector elements. */
ghost opaque function Product(s: seq<real>): (r: real)
{
  if |s| == 0 then 1.0 else Product(s[..|s|-1]) * s[|s|-1]
}

/** Sum of all vector elements. */
ghost opaque function Sum(s: seq<real>): (r: real)
  ensures (forall i | 0 <= i < |s| :: 0.0 <= s[i]) ==> r >= 0.0
{
  if |s| == 0 then 0.0 else Sum(s[..|s|-1]) + s[|s|-1]
}

lemma ReverseSum(s: seq<real>)
  requires |s| > 1
  ensures Sum(s) == s[0] + Sum(s[1..])
{
  reveal Sum();
  if |s| == 2 {
    calc {
      Sum(s);
      ==
      Sum(s[..|s|-1]) + s[|s|-1];
      ==
      s[1] + s[0];
      ==
      s[0] + s[1];
      ==
      s[0] + Sum(s[1..]);
    }
  } else {
    calc {
      Sum(s);
      ==
      Sum(s[..|s|-1]) + s[|s|-1];
      ==
      {
        ReverseSum(s[..|s|-1]);
      }
      s[..|s|-1][0] + Sum(s[..|s|-1][1..]) + s[|s|-1];
      ==
      s[0] + Sum(s[1..|s|-1]) + s[|s|-1];
      ==
      calc {
        Sum(s[1..]);
        ==
        Sum(s[1..][..|s[1..]|-1]) + s[1..][|s[1..]|-1];
        ==
        calc {
          Sum(s[1..][..|s[1..]|-1]);
          ==
          calc {
            s[1..][..|s[1..]|-1];
            ==
            s[1..|s|-1];
          }
          Sum(s[1..|s|-1]);
        }
        Sum(s[1..|s|-1]) + s[1..][|s[1..]|-1];
        ==
        Sum(s[1..|s|-1]) + s[|s|-1];
      }
      s[0] + Sum(s[1..]);
    }
  }
}

ghost function ArgMax(v: seq<real>): (r: nat)
  requires |v| > 0
  ensures 0 <= ArgMax(v) < |v|
{
  if |v| == 1 then 0
  else if v[ArgMax(v[..|v|-1])] >= v[|v|-1] then ArgMax(v[..|v|-1])
  else |v|-1
}

/* =========================== Concrete Functions =========================== */

/** Square of the given number. */
function Square(x: real): (r: real)
  ensures r >= 0.0
{
  x * x
}

/* ================================ Methods ================================= */

method ProductImpl(s: seq<real>) returns (r: real)
  ensures r == Product(s)
{
  reveal Product();
  r := 1.0;
  for i := 0 to |s|
    invariant r == Product(s[..i])
  {
    assert s[..i+1][..i] == s[..i];
    r := r * s[i];
  }
  assert s[..|s|] == s;
}

method SumImpl(s: seq<real>) returns (r: real)
  ensures r == Sum(s)
{
  reveal Sum();
  r := 0.0;
  for i := 0 to |s|
    invariant r == Sum(s[..i])
  {
    assert s[..i+1][..i] == s[..i];
    r := r + s[i];
  }
  assert s[..|s|] == s;
}

method ArgMaxImpl(s: seq<real>) returns (r: nat)
  requires |s| > 0
  ensures r == ArgMax(s)
{
  r := 0;
  var max := s[0];
  for i := 1 to |s|
    invariant s[ArgMax(s[..i])] == max
    invariant r == ArgMax(s[..i])
  {
    assert s[..i+1][..i] == s[..i];
    if s[i] > max {
      r := i;
      max := s[i];
    }
  }
  assert s[..|s|] == s;
}

// /**
//  * Returns the index of the maximum element in xs. If there is a tie, the
//  * lowest index is returned.
//  */
// function ArgMax(xs: Vector): (r: int)
//   // r is a valid index.
//   ensures 0 <= r < |xs|
//   // The element at index r is greater than or equal to all other elements.
//   ensures forall i: int :: 0 <= i < |xs| ==> xs[r] >= xs[i]
//   // When there is a tie, the lowest index is returned.
//   ensures forall i: int :: 0 <= i < |xs| ==> 
//     xs[r] == xs[i] ==> r <= i
// {
//   ArgMaxHelper(xs).0
// }

// /** Recursive helper function for ArgMax. */
// function ArgMaxHelper(xs: Vector): (r: (int, real))
//   requires |xs| > 0
//   // r.0 is a valid index.
//   ensures 0 <= r.0 < |xs|
//   // r is a corresponding (index, value) pair.
//   ensures xs[r.0] == r.1
//   // r.1 is greater than or equal to all preceding elements.
//   ensures forall i: int :: 0 <= i < |xs| ==> r.1 >= xs[i]
//   // If a tie is found, r.0 is the lowest index amongst the tied indices.
//   ensures forall i: int :: 0 <= i < |xs| ==> r.1 == xs[i] ==> r.0 <= i
// {
//   if |xs| == 1 || ArgMaxHelper(xs[0..|xs|-1]).1 < xs[|xs|-1]
//   then (|xs|-1, xs[|xs|-1])
//   else ArgMaxHelper(xs[0..|xs|-1])
// }

/**
 * Computes an upper bound on the 2^i'th root of x by repeatedly taking its
 * square root.
 */
method Power2RootUpperBound(x: real, i: nat) returns (r: real)
  requires x >= 0.0
  ensures r >= Power2Root(x, i)
{
  var j := i;
  r := x;
  while j > 0
    invariant 0 <= j <= i
    invariant r >= Power2Root(x, i - j)
  {
    if DEBUG { print "Power2RootUpperBound iteration ", i-j+1, " of ", i, "\n"; }
    MonotonicSqrt(Power2Root(x, i - j), r);
    r := SqrtUpperBound(r);
    j := j - 1;
  }
}

/**
 * Applies the Babylonian method SQRT_ITERATIONS times to yield an upper bound
 * on the square root of x.
 */
method SqrtUpperBound(x: real) returns (r: real)
  requires x >= 0.0
  ensures r >= Sqrt(x)
{
  if x == 0.0 {
    SqrtZeroIsZero();
    return 0.0;
  }
  r := if x < 1.0 then 1.0 else x;
  var i := 0;
  while i < SQRT_ITERATIONS
    invariant r >= Sqrt(x) > 0.0
  {
    if DEBUG { print "SqrtUpperBound iteration ", i+1, " of a maximum of ", SQRT_ITERATIONS, "\n"; }
    var old_r := r;
    assert Sqrt(x) <= (r + x / r) / 2.0 by {
      assert 0.0 <= (r - Sqrt(x)) * (r - Sqrt(x)); // 0.0 <= any square
      assert 0.0 <= r * r - 2.0 * r * Sqrt(x) + x; // distribute
      assert 0.0 <= (r * r - 2.0 * r * Sqrt(x) + x) / r; // divide by r
      assert ((r * r - 2.0 * r * Sqrt(x) + x) / r) == (r - 2.0 * Sqrt(x) + x / r);
      assert 0.0 <= r - 2.0 * Sqrt(x) + x / r; // simplify
      assert 2.0 * Sqrt(x) <= r + x / r; // rearrange
      assert Sqrt(x) <= (r + x / r) / 2.0; // divide by 2
    }
    r := RoundUp((r + x / r) / 2.0);
    i := i + 1;
    if (old_r - r < SQRT_ERR_MARGIN) { return; }
  }
  print "WARNING: Sqrt algorithm terminated early after reaching ", SQRT_ITERATIONS, " iterations.\n";
}

/**
 * Rounds up x to ROUNDING_PRECISION decimal places.
 */
method RoundUp(x: real) returns (r: real)
  requires x >= 0.0
  ensures r >= x
{
  var i := 0;
  r := x;
  while r != r.Floor as real && i < ROUNDING_PRECISION
    decreases ROUNDING_PRECISION - i
    invariant r == x * Pow(10.0, i)
  {
    r := r * 10.0;
    i := i + 1;
  }
  if r != r.Floor as real {
    r := r + 1.0;
  }
  // this line must be executed even if r == r.Floor because otherwise Dafny
  // will continue to store trailing zeros after the decimal point in r
  r := r.Floor as real;
  while i > 0
    invariant r >= x * Pow(10.0, i)
  {
    r := r / 10.0;
    i := i - 1;
  }
}

/**
 * Prints the first n digits of x.
 */
method PrintReal(x: real, n: nat) {
  var z: int := x.Floor;
  print z;
  print '.';
  var y: real := x;
  var i: nat := 0;
  while i < n {
    y := y * 10.0;
    z := z * 10;
    i := i + 1;
  }
  print y.Floor - z;
}

/* ================================= Lemmas ================================= */

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

/** For reals x and y, and some real z > 0, if x > y then x * z > y * z. */
lemma Increase(x: real, y: real, z: real)
  requires z > 0.0
  requires x > y
  ensures x * z > y * z
{}

/** For non-negative reals x and y, if y < x then y^2 < x^2. */
lemma IncreaseSquare(x: real, y: real)
  requires 0.0 <= y < x
  ensures y * y < x * x
{
  if y == 0.0 {
    // x * x > x * 0
    Increase(x, 0.0, x);
    calc {
      x * x;
      >
      x * 0.0;
      ==
      0.0;
      ==
      y * y;
    }
    // Since x * 0 == 0 == y * y, we have x * x > y * y
  } else {
    // x * x > x * y
    Increase(x, y, x);
    // x * y > y * y
    Increase(x, y, y);
    // Thus x * x > y * y by transitivity of '>'
  }
}

lemma MultIsMono(x: real, y: real, m: real)
  requires x <= y
  requires m >= 0.0
  ensures x * m <= y * m
{
}

/** For any non-negative reals x and y, x^2 <= y^2. */
lemma MonotonicSquarePositive(x: real, y: real)
  requires 0.0 <= x <= y
  ensures Square(x) <= Square(y)
{
  assert 0.0 <= y;
  MultIsMono(x, y, x); // x * x <= y * x
  MultIsMono(x, y, y); // x * y <= y * y;
  assert x * x <= y * y;
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

/** 
 * If every element in vector v is less than its counterpart in vector u,
 * the sum of all elements in v is less than the sum of all elements in u.
 */
lemma MonotonicSum(s1: seq<real>, s2: seq<real>)
  requires |s1| == |s2|
  requires forall i: int | 0 <= i < |s1| :: s1[i] <= s2[i]
  ensures Sum(s1) <= Sum(s2)
{
  reveal Sum();
}

/** The product of non-negative reals is non-negative. */
lemma PositiveProduct(s: seq<real>)
  requires forall i | 0 <= i < |s| :: 0.0 <= s[i]
  ensures 0.0 <= Product(s)
{
  reveal Product();
}

lemma PositiveMultiplication(x: real, y: real)
  requires x >= 0.0
  requires y >= 0.0
  ensures x * y >= 0.0
{}

/** The product of S is equal to the product of S[..|S|-1] times S[|S|-1]. */
lemma ProductDef(s: seq<real>, s0: seq<real>, s': real)
  requires |s| > 0
  requires s0 == s[..|s|-1]
  requires s' == s[|s|-1]
  ensures Product(s) == s' * Product(s0)
{
  reveal Product();
}

lemma MultiplyBothSidesPositive(x: real, y: real, z: real)
  requires x >= 0.0
  requires y >= 0.0
  requires z >= 0.0
  requires x <= y
  ensures z * x <= z * y
{}

lemma SqrtOfMult(x: real, y: real)
  requires x >= 0.0
  requires y >= 0.0
  ensures Sqrt(x * y) == Sqrt(x) * Sqrt(y)
{
  assert (Sqrt(x) * Sqrt(x)) * (Sqrt(y) * Sqrt(y)) == (Sqrt(x) * Sqrt(y)) * (Sqrt(x) * Sqrt(y));
  var r: real := (Sqrt(x) * Sqrt(y)) * (Sqrt(x) * Sqrt(y));
  assert x * y == r;
  assert Sqrt(r * r) == r by {   SqrtOfSquare2(r); }
  assert Sqrt(x * y) == Sqrt(r) by { SqrtUnique(x * y, r); }
  assert Sqrt(r) == Sqrt(x) * Sqrt(y) by {
    var w := Sqrt(x) * Sqrt(y);
    SqrtOfSquare2(w);
    assert Sqrt(w * w) == w;
    calc {
      Sqrt(x) * Sqrt(y);
      ==
      w;
      ==
      Sqrt(w * w);
      ==
      {
        assert w * w == (Sqrt(x) * Sqrt(y)) * (Sqrt(x) * Sqrt(y));
      }
      Sqrt((Sqrt(x) * Sqrt(y)) * (Sqrt(x) * Sqrt(y)));
    }
    assert Sqrt((Sqrt(x) * Sqrt(y)) * (Sqrt(x) * Sqrt(y))) == Sqrt(x) * Sqrt(y);
    assert Sqrt(r) == Sqrt(x) * Sqrt(y);
  }  
}

lemma SqrtUnique(x: real, y: real)
  requires x >= 0.0
  requires y >= 0.0
  requires x == y
  ensures Sqrt(x) == Sqrt(y)
{
}

lemma SqrtOfSquare2(x : real)
  requires x >= 0.0
  ensures Sqrt(x * x) == x
{
  SqrtOfSquare();
  assert Sqrt(Square(x)) == Abs(x);
  assert Sqrt(Square(x)) == Sqrt(x * x);
  assert Abs(x) == x;
}

}
