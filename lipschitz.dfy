include "basic_arithmetic.dfy"

module Lipschitz {
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
  ghost opaque function Product(s: seq<real>): (r: real)
  {
    if |s| == 0 then 1.0 else Product(s[..|s|-1]) * s[|s|-1]
  }

  lemma PositiveProduct(s: seq<real>)
    requires forall i | 0 <= i < |s| :: 0.0 <= s[i]
    ensures 0.0 <= Product(s)
  {
    reveal Product();
  }

  lemma ProductDef(s: seq<real>, s0: seq<real>, s': real)
    requires |s| > 0
    requires s0 == s[..|s|-1]
    requires s' == s[|s|-1]
    ensures Product(s) == s' * Product(s0)
  {
    reveal Product();
  }

  /** Sum of all vector elements. */
  ghost opaque function Sum(s: seq<real>): (r: real)
    ensures (forall i | 0 <= i < |s| :: 0.0 <= s[i]) ==> r >= 0.0
  {
    if |s| == 0 then 0.0 else Sum(s[..|s|-1]) + s[|s|-1]
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

  /** Element-wise subtraction of vector u from vector v. */
  ghost opaque function Minus(v: Vector, u: Vector): (r: Vector)
    requires |v| == |u|
    ensures |r| == |v|
    ensures forall i: int :: 0 <= i < |r| ==> r[i] == v[i] - u[i]
  {
    if |v| == 1 then [v[0] - u[0]] else [v[0] - u[0]] + Minus(v[1..], u[1..])
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
    ensures forall i: int :: 0 <= i < |r| ==> r[i] == f(v[i])
  {
    if |v| == 1 then [f(v[0])] else [f(v[0])] + Apply(v[1..], f)
  }

  /** Dot product. */
  ghost opaque function Dot(v: Vector, u: Vector): real
    requires |v| == |u|
  {
    if |v| == 1 then v[0] * u[0] else v[0] * u[0] + Dot(v[1..], u[1..])
  }

  /** Matrix vector product. */
  ghost opaque function MV(m: Matrix, v: Vector): (r: Vector)
    requires |m[0]| == |v|
    ensures |r| == |m|
    ensures forall i: int :: 0 <= i < |r| ==> r[i] == Dot(m[i], v)
  {
    if |m| == 1 then [Dot(m[0], v)]
      else [Dot(m[0], v)] + MV(m[1..], v)
  }

  /** Applies the ReLu activation function to the given vector. */
  ghost opaque function ApplyRelu(v: Vector): (r: Vector)
    ensures |r| == |v|
    ensures forall i: int :: 0 <= i < |r| ==> r[i] == Relu(v[i])
    ensures forall i: int :: 0 <= i < |r| ==> Abs(r[i]) <= Abs(v[i])
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
    requires CompatibleInput(v, n)
    ensures CompatibleOutput(r, n)
    ensures |n| == 1 ==> r == ApplyRelu(MV(n[0], v))
  {
    if |n| == 1 then Layer(n[0], v) else Layer(n[|n|-1], NN(n[..|n|-1], v))
  }

  ghost function Layer(m: Matrix, v: Vector): (r: Vector)
    requires |v| == |m[0]|
  {
    ApplyRelu(MV(m, v))
  }

  ghost predicate CompatibleInput(v: Vector, n: NeuralNetwork) {
    |v| == |n[0][0]|
  }

  ghost predicate CompatibleOutput(v: Vector, n: NeuralNetwork) {
    |v| == |n[|n|-1]|
  }

  /* ========================= Core Specification ========================== */

  /**
   * Certifies the output vector v' against the error ball e and Lipschitz
   * constant l. If certification succeeds (returns true), any input
   * corresponding to v' is verified robust.
   */
  method Certify(v': Vector, e: real, L: seq<real>) returns
      (b: bool)
    requires forall i | 0 <= i < |L| :: 0.0 <= L[i]
    requires |v'| == |L|
    ensures b ==> forall v: Vector, n: NeuralNetwork |
      CompatibleInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
      Robust(v, v', e, n)
  {
    var x := ArgMax(v');
    var i := 0;
    b := true;
    while i < |v'|
      invariant 0 <= i <= |v'|
      invariant b ==> forall j | 0 <= j < i && j != x ::
        v'[x] - L[x] * e > v'[j] + L[j] * e
    {
      if i == x {
        i := i + 1;
        continue;
      }
      if v'[x] - L[x] * e <= v'[i] + L[i] * e {
        b := false;
        break;
      }
      i := i + 1;
    }
    if b {
      ProveRobust(v', e, L, x);
    }
  }

  lemma ProveRobust(v': Vector, e: real, L: seq<real>, x: int)
    requires forall i | 0 <= i < |L| :: 0.0 <= L[i]
    requires |v'| == |L|
    requires x == ArgMax(v')
    requires forall i | 0 <= i < |v'| && i != x ::
      v'[x] - L[x] * e > v'[i] + L[i] * e
    ensures forall v: Vector, n: NeuralNetwork |
      CompatibleInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
      Robust(v, v', e, n)
  {
    A1(v', e, L, x);
    assert forall n: NeuralNetwork, v: Vector, u: Vector |
      |n[|n|-1]| == |L| && AreLipBounds(n, L) && |v| == |u| &&
      CompatibleInput(v, n) && Distance(v, u) <= e && v' == NN(n, v) ::
      NN(n, u)[x] >= v'[x] - L[x] * e &&
      forall i | 0 <= i < |L| && i != x ::
      NN(n, u)[i] <= v'[i] + L[i] * e;
    assert forall n: NeuralNetwork, v: Vector, u: Vector, i |
      |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| && |v| == |u| &&
      CompatibleInput(v, n) && Distance(v, u) <= e && v' == NN(n, v) &&
      i != x ::
      NN(n, u)[i] < NN(n, u)[x];
  }

  lemma A1(v': Vector, e: real, L: seq<real>, x: int)
    requires |v'| == |L|
    requires x == ArgMax(v')
    requires forall n: NeuralNetwork, v: Vector, u: Vector, i |
      |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| &&
      |v| == |u| && CompatibleInput(v, n) && Distance(v, u) <= e ::
      Abs(NN(n, v)[i] - NN(n, u)[i]) <= L[i] * e
    ensures forall n: NeuralNetwork, v: Vector, u: Vector |
      |n[|n|-1]| == |L| && AreLipBounds(n, L) &&
      |v| == |u| && CompatibleInput(v, n) && Distance(v, u) <= e ::
      Abs(NN(n, v)[x] - NN(n, u)[x]) <= L[x] * e
  {}

  ghost predicate Robust(v: Vector, v': Vector, e: real, n: NeuralNetwork)
    requires CompatibleInput(v, n)
    requires NN(n, v) == v'
  {
    forall u: Vector | |v| == |u| && Distance(v, u) <= e ::
      Classification(v') == Classification(NN(n, u))
  }

  /**
   * The classification of an output vector is the index of the maximum logit.
   */
  function Classification(v: Vector): int {
    ArgMax(v)
  }

  /**
   * Returns the index of the maximum element in xs.
   * If there is a tie, the lowest index is returned.
   * 
   * Todo: As this is only used in specifications, perhaps it can be declared
   * ghost and the implementation can be removed.
   */
  function ArgMax(xs: Vector): (r: int)
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

  /* ========================== Lipschitz Bounds =========================== */

  ghost predicate AreLipBounds(n: NeuralNetwork, L: seq<real>)
    requires |L| == |n[|n|-1]|
  {
    forall i | 0 <= i < |L| :: IsLipBound(n, L[i], i)
  }

  ghost predicate IsLipBound(n: NeuralNetwork, l: real, i: int)
    requires 0 <= i < |n[|n|-1]|
  {
    forall v, u: Vector | CompatibleInput(v, n) && CompatibleInput(u, n) ::
      Abs(NN(n, v)[i] - NN(n, u)[i]) <= l * Distance(v, u)
  }

  // method GenerateLipschitzBound(n: NeuralNetwork, n': Matrix, s: seq<real>,
  //     s': real, l: int) returns (r: real)
  //   requires |s| == |n|
  //   requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
  //   requires IsSpecNorm(s', n')
  //   ensures forall v, u: Vector ::
  //     Abs(Minus(NN(n, v)[l], NN(n, u)[l])) <= r * Distance(v, u)
  // {
  //   var new_s := s + [s'];
  //   var new_n := n + [[n[|n|-1][l]]];
  //   assert forall i | 0 <= i < new_s :: IsSpecNorm(new_s[i], new_n[i]);
  //   var prod := Product(new_s);
  //   SpecNormProductIsLipBound(new_n, v, u)
  // }

  // lemma Test(n: NeuralNetwork, s: seq<real>)
  //   requires |s| == |n|
  //   requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
  //   ensures forall v, u: Vector |
  //     CompatibleInput(v, n) && CompatibleInput(u, n) ::
  //     Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
  // {
  //   var v: Vector :| assume CompatibleInput(v, n);
  //   var u: Vector :| assume CompatibleInput(u, n);
  //   assert |v| == |u|;
  //   SpecNormProductIsLipBound(n, v, u, s);
  // }

  lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
      s: seq<real>)
    requires |v| == |u| && |s| == |n|
    requires CompatibleInput(v, n) && CompatibleInput(u, n)
    requires forall i | 0 <= i < |s| :: IsSpecNorm(s[i], n[i])
    ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
  {
    if |n| == 1 {
      BaseCase(n, v, u, s);
      reveal Product();
    } else {
      BaseCase(n, v, u, s);

      var n0 := n[..|n|-1];
      var s0 := s[..|s|-1];
      assert |s0| == |n0|;
      SpecNormProductIsLipBound(n0, v, u, s0);

      var n' := n[|n|-1];
      var s' := s[|s|-1];
      var v' := NN(n0, v);
      var u' := NN(n0, u);

      SpecNormIsLayerLipBound(n', v', u', s');
      reveal NN();
      assert Distance(NN(n, v), NN(n, u)) <= s' * Distance(v', u');
      assert Distance(v', u') <= Product(s0) * Distance(v, u);
      MultiplicationInequality(n, v, u, v', u', s0, s');
      ProductDef(s, s0, s');
      MultiplyBothSides(s, s0, s', v, u);
    }
  }

  lemma MultiplyBothSides(s: seq<real>, s0: seq<real>, s': real, v: Vector,
      u: Vector)
    requires |v| == |u|
    requires Product(s) == s' * Product(s0)
    ensures Product(s) * Distance(v, u) == s' * Product(s0) * Distance(v, u)
  {}

  lemma MultiplicationInequality(n: NeuralNetwork, v: Vector, u: Vector,
      v': Vector, u': Vector, s0: seq<real>, s': real)
    requires |v| == |u|
    requires |v'| == |u'|
    requires s' >= 0.0
    requires CompatibleInput(v, n) && CompatibleInput(u, n)
    requires Distance(NN(n, v), NN(n, u)) <= s' * Distance(v', u')
    requires Distance(v', u') <= Product(s0) * Distance(v, u)
    ensures Distance(NN(n, v), NN(n, u)) <= s' * Product(s0) * Distance(v, u)
  {}

  lemma BaseCase(n: NeuralNetwork, v: Vector, u: Vector, s: seq<real>)
    requires |v| == |u| && |s| == |n|
    requires CompatibleInput(v, n) && CompatibleInput(u, n)
    requires IsSpecNorm(s[0], n[0])
    ensures Distance(Layer(n[0], v), Layer(n[0], u)) <= s[0] * Distance(v, u)
  {
    SpecNormIsLayerLipBound(n[0], v, u, s[0]);
  }

  lemma LogitLipBounds(n: NeuralNetwork, n': NeuralNetwork, v: Vector,
      u: Vector, l: int)
    requires |v| == |u|
    requires 1 < |n| == |n'|
    requires CompatibleInput(v, n)
    requires CompatibleInput(u, n)
    requires 0 <= l < |n[|n|-1]|
    requires n' == n[..|n|-1] + [[n[|n|-1][l]]]
    requires Distance(NN(n', v), NN(n', u)) == Abs(NN(n, v)[l] - NN(n, u)[l])
  {
    TrimmedNN(n, n', v, l);
    TrimmedNN(n, n', u, l);
  }

  /**
   * Let n' be the neural network n with all rows except row l removed.
   * Formally, n' == n[..|n|-1] + [[n[|n|-1][l]]].
   * Show that NN(n', v) == [NN(n, v)[l]].
   * Hence, NN(n', v)[0] == NN(n, v)[l].
   */
  lemma TrimmedNN(n: NeuralNetwork, n': NeuralNetwork, v: Vector, l: int)
    requires 0 <= l < |n[|n|-1]|
    requires CompatibleInput(v, n) && CompatibleInput(v, n')
    requires 1 < |n| == |n'|
    requires n' == n[..|n|-1] + [[n[|n|-1][l]]]
    ensures NN(n', v) == [NN(n, v)[l]]
  {
    assert n' == n[..|n|-1] + [[n[|n|-1][l]]];
    reveal NN();
    // It's totally ridiculous that Dafny can do this.
    // calc {
    //   NN(n', v);
    //   Layer(n'[|n'|-1], NN(n'[..|n'|-1], v));
    //   ApplyRelu(MV(n'[|n'|-1], NN(n'[..|n'|-1], v)));
    //   ApplyRelu(MV([n[|n|-1][l]], NN(n'[..|n'|-1], v)));
    //   ApplyRelu(MV([n[|n|-1][l]], NN(n[..|n|-1], v)));
    //   ApplyRelu([Dot([n[|n|-1][l]][0], NN(n[..|n|-1], v))]);
    //   ApplyRelu([Dot(n[|n|-1][l], NN(n[..|n|-1], v))]);
    //   // Dot(m[l], v) == MV(m, v)[l] by definition of MV.
    //   ApplyRelu([MV(n[|n|-1], NN(n[..|n|-1], v))[l]]);
    //   // By definition of apply when argument is len 1.
    //   [Relu(MV(n[|n|-1], NN(n[..|n|-1], v))[l])];
    //   // By definition of ApplyRelu.
    //   [ApplyRelu(MV(n[|n|-1], NN(n[..|n|-1], v)))[l]];
    //   [Layer(n[|n|-1], NN(n[..|n|-1], v))[l]];
    //   [NN(n, v)[l]];
    // }
  }

  /* =============================== End New =============================== */

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
    SmallerRelu(MV(m, v), MV(m, u));
  }

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
   * Since v - u is just a vector, we have ||m.(v - u)|| <= ||m|| * ||v - u||
   */
  lemma SpecNormPropertyHoldsForDifferenceVectors(m: Matrix, s: real,
      v: Vector, u: Vector)
    requires |v| == |u| == |m[0]|
    requires IsSpecNorm(s, m)
    ensures L2(MV(m, Minus(v, u))) <= s * Distance(v, u)
  {}

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
    SmallerL2Norm(Minus(ApplyRelu(v), ApplyRelu(u)), Minus(v, u));
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
      MonotonicSquare(v[i], u[i]);
      i := i + 1;
    }
  }
}