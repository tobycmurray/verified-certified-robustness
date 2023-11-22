include "basic_arithmetic.dfy"

module LinearAlgebra {
  import opened BasicArithmetic

  /* ================================ Types ================================ */

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

  /* A neural network is a non-empty sequence of layers, which are (functionally)
  weight matrices, and whose product must be defined. */
  type NeuralNetwork = n: seq<Matrix> | |n| > 0 && 
    forall i: int :: 0 <= i < |n| - 1 ==> |n[i]| == |n[i + 1][0]|
    witness [NonEmptyMatrix()]

  /** Required to convince Dafny that [[0.0]] is a Matrix. */
  function NonEmptyMatrix(): Matrix {
    [[0.0]]
  }

  /* ============================= Operations ============================== */

  /** Product of all vector elements. */
  ghost opaque function Product(v: Vector): (r: real)
    // ensures (forall i | 0 <= i < |v| :: v[i] >= 0.0) ==> r >= 0.0
    ensures |v| == 1 ==> r == v[0]
  {
    if |v| == 1 then v[0] else v[0] * Product(v[1..])
  }

  /** Sum of all vector elements. */
  ghost opaque function Sum(v: Vector): (r: real)
    ensures (forall i: int | 0 <= i < |v| :: v[i] >= 0.0) ==> r >= 0.0
  {
    if |v| == 1 then v[0] else v[0] + Sum(v[1..])
  }

  /** Element-wise subtraction of vector u from vector v. */
  ghost opaque function Minus(v: Vector, u: Vector): (r: Vector)
    requires |v| == |u|
    ensures |r| == |v|
    ensures forall i: int :: 0 <= i < |v| ==> r[i] == v[i] - u[i]
  {
    if |v| == 1 then [v[0] - u[0]]
      else [v[0] - u[0]] + Minus(v[1..], u[1..])
  }

  /**
   * L2 norm of the given vector.
   * L2([v_1, v_2,..., v_n]) = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
   */
  ghost opaque function L2(v: Vector): real
  {
    Sqrt(Sum(Apply(v, Square)))
  }

  /**
   * The distance between two vectors is the norm of their difference-vector.
   */
  ghost function Distance(v: Vector, u: Vector): real
    requires |v| == |u|
  {
    L2(Minus(v, u))
  }

  /** 
   * Applies the given function to all elements of the given vector.
   * Similar to Haskell's 'map' function.
   */
  ghost opaque function Apply(v: Vector, f: real -> real): (r: Vector)
    ensures |v| == |r|
    ensures forall i: int :: 0 <= i < |v| ==> r[i] == f(v[i])
  {
    if |v| == 1 then [f(v[0])] else [f(v[0])] + Apply(v[1..], f)
  }

  /** Dot product. */
  ghost opaque function Dot(v: Vector, u: Vector): real
    requires |v| == |u|
  {
    if |v| == 1 then v[0] * u[0]
      else v[0] * u[0] + Dot(v[1..], u[1..])
  }

  /** Matrix vector product. */
  ghost opaque function MV(m: Matrix, v: Vector): (r: Vector)
    requires |m[0]| == |v|
    ensures |r| == |m|
    ensures forall i: int :: 0 <= i < |m| ==> r[i] == Dot(m[i], v)
  {
    if |m| == 1 then [Dot(m[0], v)] 
      else [Dot(m[0], v)] + MV(m[1..], v)
  }

  /**
   * Returns the index of the maximum element in xs.
   * If there is a tie, the lowest index is returned.
   * 
   * Todo: As this is only used in specifications, perhaps it can be declared
   * ghost and the implementation can be removed.
   */
  opaque function ArgMax(xs: Vector): (r: int)
    // r is a valid index.
    ensures 0 <= r < |xs|
    // The element at index r is greater than or equal to all other elements.
    ensures forall i: int :: 0 <= i < |xs| ==> xs[r] >= xs[i]
    // When there is a tie, the lowest index is returned.
    ensures forall i: int :: 0 <= i < |xs| ==> 
      xs[r] == xs[i] ==> r <= i
  {
    ArgMaxHelper(xs).0
  }

  function ArgMaxHelper(xs: Vector): (r: (int, real))
    requires |xs| > 0
    // r.0 is a valid index.
    ensures 0 <= r.0 < |xs|
    // r is a corresponding (index, value) pair.
    ensures xs[r.0] == r.1
    // r.1 is greater than or equal to all preceding elements.
    ensures forall i: int :: 0 <= i < |xs| ==> r.1 >= xs[i]
    // If a tie is found, r.0 is the lowest index amongst the tied indices.
    ensures forall i: int :: 0 <= i < |xs| ==> r.1 == xs[i] ==> r.0 <= i
  {
    if |xs| == 1 || ArgMaxHelper(xs[0..|xs|-1]).1 < xs[|xs|-1]
    then (|xs|-1, xs[|xs|-1])
    else ArgMaxHelper(xs[0..|xs|-1])
  }

  /** Applies the ReLu activation function to the given vector. */
  ghost opaque function ApplyRelu(v: Vector): (r: Vector)
    ensures |r| == |v|
    ensures forall i: int :: 0 <= i < |v| ==> r[i] == Relu(v[i])
    ensures forall i: int :: 0 <= i < |v| ==> Abs(r[i]) <= Abs(v[i])
  {
    Apply(v, Relu)
  }

  /** Spectral norm of the given matrix (external implementation). */
  method {:extern} SpecNorm(m: Matrix) returns (r: real)
    ensures IsSpecNorm(r, m)

  lemma SpecNormLemma(m: Matrix, v: Vector, s: real)
    requires IsSpecNorm(s, m)
    requires |v| == |m[0]|
    ensures L2(MV(m, v)) <= s * L2(v)
  {}

  ghost predicate IsSpecNorm(s: real, m: Matrix) {
    s >= 0.0 && forall v: Vector | |v| == |m[0]| :: L2(MV(m, v)) <= s * L2(v)
  }

  /** 
   * Function representing the assumed behaviour of the neural network. Models how
   * the neural network transforms input vectors into output vectors.
   */
  ghost opaque function NN(n: NeuralNetwork, v: Vector): (r: Vector)
    requires |n[0][0]| == |v|
    ensures |r| == |n[|n| - 1]|
    ensures |n| == 1 ==> r == ApplyRelu(MV(n[0], v))
  {
    if |n| == 1 then Layer(n[0], v) else NN(n[1..], Layer(n[0], v))
  }

  ghost function Layer(m: Matrix, v: Vector): (r: Vector)
    requires |v| == |m[0]|
  {
    ApplyRelu(MV(m, v))
  }


  /* =============================== Lemmas ================================ */

  /** 
   * If every element in vector v is less than its counterpart in vector u,
   * the sum of all elements in v is less than the sum of all elements in u.
   */
  lemma MonotonicSum(v: Vector, u: Vector)
    requires |v| == |u|
    requires forall i: int | 0 <= i < |v| :: v[i] <= u[i]
    ensures Sum(v) <= Sum(u)
  {
    reveal Sum();
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

  /** Dot products distribute over subtraction: v.(u - w) == v.u - v.w */
  lemma DotIsDistributive(v: Vector, u: Vector, w: Vector)
    requires |v| == |u| == |w|
    ensures Dot(v, Minus(u, w)) == Dot(v, u) - Dot(v, w)
  {
    reveal Dot();
    if |v| == 1 {
      // v.(u - w) 
      // == v[0] * (u[0] - w[0])
      // == v[0] * u[0] - v[0] * w[0] 
      // == v.u - v.w
    } else {
      // 1. Assume v[1..].(u[1..] - w[1..]) == v[1..].u[1..] - v[1..].w[1..]
      DotIsDistributive(v[1..], u[1..], w[1..]);
      // 2: (u - w)[1..] == u[1..] - w[1..]
      assert Minus(u, w)[1..] == Minus(u[1..], w[1..]);
      // v.(u - w) 
      // == v[0] * (u - w)[0] + v[1..].(u - w)[1..]
      // == v[0] * u[0] - v[0] * w[0] + v[1..].(u - w)[1..]
      // From 2: == v[0] * u[0] - v[0] * w[0] + v[1..].(u[1..] - w[1..])
      // From 1: == v[0] * u[0] - v[0] * w[0] + v[1..].u[1..] - v[1..].w[1..]
      // == v[0] * u[0] + v[1..].u[1..] - (v[0] * w[0] + v[1..].w[1..])
      // == v.u - v.w
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
      // 1. m.(v - u)[i] == m[i].(v - u) by definition
      DotIsDistributive(m[i], v, u);
      // 2. m[i].(v - u) == m[i].v - m[i].u
      // From 1, 2: m.(v - u)[i] == m[i].v - m[i].u
    }
    // For all i: m.(v - u)[i] == m[i].v - m[i].u
    // Hence m.(v - u) == m.v - m.u
  }

  /**
   * Since v - u is just a vector, we have ||m.(v - u)|| <= ||m|| * ||v - u||
   */
  lemma SpecNormPropertyHoldsForDifferenceVectors(m: Matrix, s: real,
      v: Vector, u: Vector)
    requires |v| == |u| == |m[0]|
    requires IsSpecNorm(s, m)
    ensures L2(MV(m, Minus(v, u))) <= s * Distance(v, u)
  {}

  /** 
   * A matrix's spectral norm is a Lipschitz bound:
   * ||m.v - m.u|| <= ||m|| * ||v - u||
   */
  lemma SpecNormIsMvLipBound(m: Matrix, v: Vector, u: Vector, s: real)
    requires |v| == |u| == |m[0]|
    requires IsSpecNorm(s, m)
    ensures Distance(MV(m, v), MV(m, u)) <= s * Distance(v, u)
  {
    SpecNormPropertyHoldsForDifferenceVectors(m, s, v, u);
    MvIsDistributive(m, v, u);
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
    // |v[i]| <= |u[i]| for each i
    SmallerApplySquare(v, u);
    // v[i]^2 <= u[i]^2 for each i
    MonotonicSum(Apply(v, Square), Apply(u, Square));
    // v[0]^2 + ... + v[n]^2 <= u[0]^2 + ... + u[n]^2
    MonotonicSqrt(Sum(Apply(v, Square)), Sum(Apply(u, Square)));
    // sqrt(v[0]^2 + ... + v[n]^2) <= sqrt(u[0]^2 + ... + u[n]^2)
    // ||v|| <= ||u||
  }

  /**
   * The distance between two vectors can only be decreased when the ReLu function
   * is applied to each one. This is equivalent to stating that the spectral norm
   * of the ReLu layer is 1.
   * ||R(v) - R(u)|| <= ||v - u|| where R applies the ReLu activation function.
   */
  lemma SmallerRelu(v: Vector, u: Vector)
    requires |v| == |u|
    ensures Distance(ApplyRelu(v), ApplyRelu(u)) <= Distance(v, u)
  {
    // Dafny can infer that |(R(v) - R(u))[i]| <= |(v - u)[i]| for each i
    SmallerL2Norm(Minus(ApplyRelu(v), ApplyRelu(u)), Minus(v, u));
    // ||R(v) - R(u)|| <= ||v - u||
  }

  /**
   * A neural network layer consists of matrix-vector multiplication, followed by
   * an application of the ReLu activation function. A Lipschitz bound of a layer
   * with matrix m is the spectral norm of that matrix.
   * ||R(m.v) - R(m.u)|| <= ||m|| * ||v - u||
   * where R applies the ReLu activation function.
   */
  lemma SpecNormIsLayerLipBound(m: Matrix, v: Vector, u: Vector, s: real)
    requires |m[0]| == |v| == |u|
    requires IsSpecNorm(s, m)
    ensures Distance(Layer(m, v), Layer(m, u)) <= s * Distance(v, u)
  {
    SpecNormIsMvLipBound(m, v, u, s);
    // 1. ||m.v - m.v|| <= ||m|| * ||v - u||
    SmallerRelu(MV(m, v), MV(m, u));
    // 2. ||R(m.v) - R(m.u)|| <= ||m.v - m.u||
    // From 1, 2: ||R(m.v) - R(m.u)|| <= ||m|| * ||v - u||
  }

  /**
   * If each element in vector v has a lower absolute value than its counterpart
   * in u, then the square of each element in v is lower than the square of its
   * counterpart in u.
   */
  lemma SmallerApplySquare(v: Vector, u: Vector)
    requires |v| == |u|
    requires forall i: int :: 0 <= i < |v| ==> Abs(v[i]) <= Abs(u[i])
    ensures forall i: int :: 0 <= i < |v| ==>  Apply(v, Square)[i] <= Apply(u, Square)[i]
  {
    var i := 0;
    while i < |v|
      invariant i <= |v|
      invariant forall j: int :: 0 <= j < i ==> Apply(v, Square)[j] <= Apply(u, Square)[j]
    {
      // |v[i]| <= |u[i]|
      MonotonicSquare(v[i], u[i]);
      // v[i]^2 <= u[i]^2
      i := i + 1;
    }
    // v[i]^2 <= u[i]^2 for each i
  }

  // lemma Z2(s0: real, s: seq<real>, v: Vector, u: Vector)
  //   requires |s| > 0
  //   requires s0 == s[0]
  //   requires |v| == |u|
  //   ensures s0 * Distance(v, u) == s[0] * Distance(v, u)
  // {}

  // lemma Z3(s1: seq<real>, v: Vector, u: Vector, v': Vector, u': Vector)
  //   requires |v| == |u|
  //   requires v' == v
  //   requires u' == u
  //   requires |s1| > 0
  //   ensures Product(s1) * Distance(v', u') == Product(s1) * Distance(v, u)
  // {}

  // lemma Idk(n: NeuralNetwork, v: Vector, u: Vector, s: seq<real>, x: real)
  //   requires |v| == |u| == |n[0][0]|
  //   requires |s| > 0
  //   requires x == s[0]
  //   requires Distance(Layer(n[0], v), Layer(n[0], u)) <= x * Distance(v, u)
  //   ensures Distance(Layer(n[0], v), Layer(n[0], u)) <= s[0] * Distance(v, u)
  // {}

  // lemma Again(x: real, y: real, z: real, s: seq<real>)
  //   requires x <= y * z
  //   requires |s| > 0
  //   requires s[0] == y
  //   ensures x <= s[0] * z
  // {}

  // lemma AndAgain(a: real, s: seq<real>, c: real, d: real)
  //   requires |s| > 0
  //   requires a <= Product(s) * c
  //   requires c == d
  //   ensures a <= Product(s) * d
  // {}

  // lemma Another(n: NeuralNetwork, v': Vector, u': Vector, b: seq<real>, x: real, d: real)
  //   requires |b| > 0
  //   requires |v'| == |u'| == |n[0][0]|
  //   requires Distance(NN(n, v'), NN(n, u')) <= Product(b) * x
  //   requires d == x
  //   ensures Distance(NN(n, v'), NN(n, u')) <= Product(b) * d
  // {}

  // lemma Boom(s1: seq<real>, nv: Vector, nu: Vector, d: real)
  //   requires |s1| > 0
  //   requires |nv| == |nu|
  //   requires d == Distance(nv, nu)
  //   ensures Product(s1) * Distance(nv, nu) == Product(s1) * d

  // lemma OtherBoom(s0: real, s1: seq<real>, v: Vector, u: Vector, d: real)
  //   requires |v| == |u|
  //   requires |s1| > 0
  //   requires d <= s0 * Distance(v, u)
  //   requires Product(s1) >= 0.0
  //   ensures Product(s1) * d <= Product(s1) * s0 * Distance(v, u)

  // lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
  //     s: seq<real>)
  //   requires |v| == |u| == |n[0][0]|
  //   requires |s| == |n|
  //   requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
  //   ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
  // {
  //   if |n| == 1 {
  //     SpecNormIsLayerLipBound(n[0], v, u, s[0]);
  //   } else {
  //     var s0 := s[0];
  //     SpecNormIsLayerLipBound(n[0], v, u, s0);
  //     assert Distance(Layer(n[0], v), Layer(n[0], u)) <= s0 * Distance(v, u);
  //     var d := Distance(Layer(n[0], v), Layer(n[0], u));
  //     assert d <= s0 * Distance(v, u);
      
  //     var nv := Layer(n[0], v);
  //     var nu := Layer(n[0], u);
  //     var s1 := s[1..];
  //     SpecNormProductIsLipBound(n[1..], nv, nu, s1);
  //     assert Distance(NN(n[1..], nv), NN(n[1..], nu)) <= Product(s1) * Distance(nv, nu);
  //     reveal NN();

  //     assert Distance(NN(n, v), NN(n, u)) <= Product(s1) * Distance(nv, nu);
  //     assert Distance(nv, nu) == Distance(Layer(n[0], v), Layer(n[0], u));
  //     assert d == Distance(nv, nu);
  //     Boom(s1, nv, nu, d);
  //     assert Product(s1) * Distance(nv, nu) == Product(s1) * d;
  //     assert Distance(NN(n, v), NN(n, u)) <= Product(s1) * d;

  //     ProductOfPositivesIsPositive(s1);
  //     assert Product(s1) >= 0.0;
  //     OtherBoom(s0, s1, v, u, d);
  //     assert Product(s1) * d <= Product(s1) * s0 * Distance(v, u);
  //     assert Distance(NN(n, v), NN(n, u)) <= Product(s1) * s0 * Distance(v, u);
  //     WhoKnows(s, s1, s0, Distance(v, u));
  //     assert Distance(NN(n, v), NN(n, u)) <= Product(s[1..]) * s[0] * Distance(v, u);
  //     assume false;
  //   }
  // }

  // lemma WhoKnows(s: seq<real>, s1: seq<real>, s0: real, x: real)
  //   requires |s| > 1
  //   requires s1 == s[1..]
  //   requires s0 == s[0]
  //   ensures Product(s1) * s0 * x == Product(s[1..]) * s[0] * x
  // {}

  // lemma ProductProperty(s: seq<real>)
  //   requires |s| > 1
  //   ensures Product(s) == s[0] * Product(s[1..])
  // {
  //   reveal Product();
  // }

  // lemma Final(e: real, n: NeuralNetwork, v: Vector, u: Vector, k: real, s: seq<real>, h: real)
  //   requires |v| == |u| == |n[0][0]|
  //   requires e == Distance(NN(n, v), NN(n, u))
  //   requires |s| >= 1
  //   requires k == Product(s)
  //   requires h == Distance(v, u)
  //   requires e <= k * h
  //   ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
  // {}

  // lemma MultiplicationSubstitution(e: real, f: real, s0: real, h: real, k: real)
  //   requires e <= f * s0 * h
  //   requires f * s0 == k
  //   ensures e <= k * h
  // {}

  // lemma ProductDef(s: seq<real>, f: real, s0: real)
  //   requires |s| >= 2
  //   requires f == Product(s[1..])
  //   requires s0 == s[0]
  //   ensures f * s0 == Product(s)
  // {
  //   reveal Product();
  // }

  // lemma Increase2(e: real, f: real, g: real, s0: real, h: real)
  //   requires e <= f * g
  //   requires g <= s0 * h
  //   requires f >= 0.0
  //   ensures e <= f * s0 * h
  // {}

  lemma ProductOfPositivesIsPositive(s: seq<real>)
    requires forall i | 0 <= i < |s| :: s[i] >= 0.0
    requires |s| >= 1
    ensures Product(s) >= 0.0
  {
    reveal Product();
  }

  /* ============================= NEW LEMMAS ============================== */

  method GenLipBound(n: NeuralNetwork, specNorms: seq<real>,
    logitSpecNorm: real, logit: int) returns (r: real)
    requires |specNorms| == |n|-1
    requires 0 <= logit < |n[|n|-1]|
    requires forall i | 0 <= i < |specNorms| :: IsSpecNorm(specNorms[i], n[i])
    requires IsSpecNorm(logitSpecNorm, [n[|n|-1][logit]])
    ensures IsLogitLipBound(r, n, logit)
  {
    if |n| == 1 {
      assert forall v: Vector | |v| == |[n[|n|-1][logit]][0]| :: L2(MV([n[|n|-1][logit]], v)) <= logitSpecNorm * L2(v);
      assert forall v: Vector | |v| == |n[|n|-1][logit]| :: L2(MV([n[|n|-1][logit]], v)) <= logitSpecNorm * L2(v);
      assume false;
      return logitSpecNorm;
    }
    assume false;
  }

  ghost predicate IsLogitLipBound(l: real, n: NeuralNetwork, x: int)
    requires 0 <= x < |n[|n|-1]|
  {
    forall v, u: Vector | |v| == |u| == |n[0][0]| ::
      Abs(NN(n, v)[x] - NN(n, u)[x]) <= Distance(v, u) * l
  }

  ghost predicate IsLipBound(l: real, n: NeuralNetwork) {
    forall v, u: Vector | |v| == |u| == |n[0][0]| ::
      Distance(NN(n, v), NN(n, u)) <= Distance(v, u) * l
  }
}
