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

  lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
      s: seq<real>)
    requires |v| == |u| == |n[0][0]|
    requires |s| == |n|
    requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
    ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
  {
    if |n| == 1 {
      SpecNormIsLayerLipBound(n[0], v, u, s[0]);
    } else {
      var n0 := n[0];
      var s0 := s[0];
      SpecNormIsLayerLipBound(n0, v, u, s0);
      assert Distance(Layer(n0, v), Layer(n0, u)) <= s0 * Distance(v, u);
      var a := Distance(Layer(n0, v), Layer(n0, u));
      var c := Distance(v, u);
      var d := s[0];
      A2(n0, v, u, s0, a, c);
      assert a <= s0 * c;
      A1(a, s0, c, d);
      assert a <= d * c;
      A3(a, d, c, n0, v, u, s);
      assert Distance(Layer(n0, v), Layer(n0, u)) <= s[0] * Distance(v, u);
      var n1 := n[1..];
      var s1 := s[1..];
      var nv := Layer(n0, v); // *
      var nu := Layer(n0, u);
      SpecNormProductIsLipBound(n1, nv, nu, s1);
      assert Distance(NN(n1, nv), NN(n1, nu)) <= Product(s1) * Distance(nv, nu);
      
      assume false;
    }
  }

  lemma A3(a: real, d: real, c: real, n0: Matrix, v: Vector, u: Vector, s: seq<real>)
    requires |v| == |u| == |n0[0]|
    requires a <= d * c
    requires |s| >= 1
    requires a == Distance(Layer(n0, v), Layer(n0, u))
    requires d == s[0]
    requires c == Distance(v, u)
    ensures Distance(Layer(n0, v), Layer(n0, u)) <= s[0] * Distance(v, u)
  {}

  lemma A2(n0: Matrix, v: Vector, u: Vector, s0: real, a: real, c: real)
    requires |v| == |u| == |n0[0]|
    requires Distance(Layer(n0, v), Layer(n0, u)) <= s0 * Distance(v, u)
    requires a == Distance(Layer(n0, v), Layer(n0, u))
    requires c == Distance(v, u)
    ensures a <= s0 * c
  {}

  lemma A1(a: real, b: real, c: real, d: real)
    requires a <= b * c
    requires b == d
    ensures a <= d * c
  {}

  // lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
  //     s: seq<real>)
  //   requires |v| == |u| == |n[0][0]|
  //   requires |s| == |n|
  //   requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
  //   ensures L2(Minus(NN(n, v), NN(n, u))) <= Product(s) * L2(Minus(v, u))
  // {
  //   if |n| == 1 {
  //     SpecNormIsLayerLipBound(n[0], v, u, s[0]);
  //   } else {
  //     // 1. n(v) == n[1..](R(n[0].v))
  //     // 2. n(u) == n[1..](R(n[0].u))
  //     SpecNormIsLayerLipBound(n[0], v, u, s[0]);
  //     // 3. ||R(n[0].v) - R(n[0].u)|| <= s[0] * ||v - u||
  //     var nv := ApplyRelu(MV(n[0], v));
  //     var nu := ApplyRelu(MV(n[0], u));
  //     assert L2(Minus(nv, nu)) <= s[0] * L2(Minus(v, u));
  //     SpecNormProductIsLipBound(n[1..], nv, nu, s[1..]);
  //     // 4. Assume ||n[1..](R(n[0].v)) - n[1..](R(n[0].u))|| <= s[1..] * ||R(n[0].v) - R(n[0].u)||
  //     assert L2(Minus(NN(n[1..], nv), NN(n[1..], nu))) <= Product(s[1..]) * L2(Minus(nv, nu));
  //     // 5. n[1..](R(n[0].v)) == n(v)
  //     assert NN(n[1..], nv) == NN(n, v);
  //     // 6. n[1..](R(n[0].u)) == n(u)
  //     assert NN(n[1..], nu) == NN(n, u);
  //     // 7: From 4, 5, 6: ||n(v) - n(u)|| <= s[1..] * ||R(n[0].v) - R(n[0].u)||
  //     K1(NN(n[1..], nv), NN(n, v), NN(n[1..], nu), NN(n, u));
  //     K2(L2(Minus(NN(n[1..], nv), NN(n[1..], nu))), L2(Minus(NN(n, v), NN(n, u))), Product(s[1..]), L2(Minus(nv, nu)));
  //     assert L2(Minus(NN(n, v), NN(n, u))) <= Product(s[1..]) * L2(Minus(nv, nu));
  //     K3(L2(Minus(NN(n, v), NN(n, u))), Product(s[1..]), L2(Minus(nv, nu)), s[0], L2(Minus(v, u)));
  //     // 8: From 3, 7: ||n(v) - n(u)|| <= s[1..] * s[0] * ||v - u||
  //     assert L2(Minus(NN(n, v), NN(n, u))) <= Product(s[1..]) * s[0] * L2(Minus(v, u));
  //     // 9: S(n[1..]) * ||n[0]|| == S(n)
  //     // From 8, 9: ||n(v) - n(u)|| <= S(n) * ||v - u||
  //     assert L2(Minus(NN(n, v), NN(n, u))) <= Product(s) * L2(Minus(v, u));
  //   }
  // }

  // /** 
  //  * The product of the spectral norms of each layer in a neural network is a
  //  * Lipschitz bound of the network.
  //  * ||n(v) - n(u)|| <= S(n) * ||v - u||
  //  * where:
  //  * - n(v) applies the vector v to the neural network n
  //  * - S(n) is the product of the spectral norms of each layer in n
  //  */
  // lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector, s)
  //   requires |n[0][0]| == |v| == |u|
  //   ensures L2(Minus(ApplyNN(n, v), ApplyNN(n, u))) <= 
  //     SpecNormProduct(n) * L2(Minus(v, u))
  // {
  //   if |n| == 1 {
  //     // 1. n(v) == R(n[0].v)
  //     // 2. n(u) == R(n[0].u)
  //     SpecNormIsLayerLipBound(n[0], v, u);
  //     // 3. ||R(n[0].v) - R(n[0].u)|| <= ||n[0]|| * ||v - u||
  //     // From 1, 2, 3: ||n(v) - n(u)|| <= ||n[0]|| * ||v - u||
  //     // 4. S(n) == ||n[0]||
  //     // From 3, 4: ||n(v) - n(u)|| <= S(n) * ||v - u||
  //   } else {
  //     // 1. n(v) == n[1..](R(n[0].v))
  //     // 2. n(u) == n[1..](R(n[0].u))
  //     SpecNormIsLayerLipBound(n[0], v, u);
  //     // 3. ||R(n[0].v) - R(n[0].u)|| <= ||n[0]|| * ||v - u||
  //     assert L2(Minus(ApplyRelu(MV(n[0], v)), ApplyRelu(MV(n[0], u)))) <=
  //       SpecNorm(n[0]) * L2(Minus(v, u));
  //     var nv := ApplyRelu(MV(n[0], v));
  //     var nu := ApplyRelu(MV(n[0], u));
  //     SpecNormProductIsLipBound(n[1..], nv, nu);
  //     // 4. Assume ||n[1..](R(n[0].v)) - n[1..](R(n[0].u))|| <= S(n[1..]) * ||R(n[0].v) - R(n[0].u)||
  //     // 5. n[1..](R(n[0].v)) == n(v)
  //     // 6. n[1..](R(n[0].u)) == n(u)
  //     // 7: From 4, 5, 6: ||n(v) - n(u)|| <= S(n[1..]) * ||R(n[0].v) - R(n[0].u)||
  //     // 8: From 3, 7: ||n(v) - n(u)|| <= S(n[1..]) * ||n[0]|| * ||v - u||
  //     // 9: S(n[1..]) * ||n[0]|| == S(n)
  //     // From 8, 9: ||n(v) - n(u)|| <= S(n) * ||v - u||
  //   }
  // }
}
