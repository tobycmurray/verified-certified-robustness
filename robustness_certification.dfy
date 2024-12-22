include "basic_arithmetic.dfy"
include "linear_algebra.dfy"
include "operator_norms.dfy"
include "neural_networks.dfy"

module RobustnessCertification {
import opened BasicArithmetic
import opened LinearAlgebra
import opened OperatorNorms
import opened NeuralNetworks

/* ============================ Ghost Functions ============================= */

/**
 * The robustness property is the key specification for the project. An
 * input-output pair of vectors (v, v') for a neural network n is robust
 * with respect to an error ball e if for all input vectors u within a
 * distance e from v, the classification (i.e., argmax) of the output
 * corresponding to u is equal to the classification of v'.
 */
ghost predicate Robust(v: Vector, v': Vector, e: real, n: NeuralNetwork)
  requires IsInput(v, n)
  requires NN(n, v) == v'
{
  forall u: Vector | |v| == |u| && Distance(v, u) <= e ::
    ArgMax(v') == ArgMax(NN(n, u))
}

/** True iff every L[i] is a Lipschitz bound of the matrix n[i]. */
ghost predicate AreLipBounds(n: NeuralNetwork, L: seq<real>)
  requires |L| == |n[|n|-1]|
{
  forall i | 0 <= i < |L| :: IsLipBound(n, L[i], i)
}

/**
 * A real number l is a Lipschitz bound of an output logit i iff l is an
 * upper bound on the change in i per change in distance of the input vector.
 */
ghost predicate IsLipBound(n: NeuralNetwork, r: real, l: int)
  requires 0 <= l < |n[|n|-1]|
{
  forall v, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs(NN(n, v)[l] - NN(n, u)[l]) <= r * Distance(v, u)
}

/* ============================= Margin Bounds ============================== */

ghost predicate IsMarginLipBound(n: NeuralNetwork, r: real, p: nat, q: nat)
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]|
{
  forall v, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u)
}

ghost predicate AreMarginLipBounds(n: NeuralNetwork, L: Matrix)
  requires |L| == |L[0]| == |n[|n|-1]|
{
  forall p: nat, q: nat | p < |L| && q < |L[0]| :: IsMarginLipBound(n, L[p][q], p, q)
}

method CertifyMargin(v': Vector, e: real, L: Matrix) returns (b: bool)
  requires |L| == |L[0]| == |v'|
  requires forall i: nat, j: nat | i < |L| && j < |L[0]| :: 0.0 <= L[i][j]
  ensures b ==> forall v: Vector, n: NeuralNetwork |
    IsInput(v, n) && NN(n, v) == v' && AreMarginLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  var x := ArgMaxImpl(v');
  var i := 0;
  b := true;
  while i < |v'|
    invariant 0 <= i <= |v'|
    invariant b ==> forall j | 0 <= j < i && j != x :: L[j][x] * e < v'[x] - v'[j]
  {
    if i == x {
      i := i + 1;
      continue;
    }
    if L[i][x] * e >= v'[x] - v'[i] {
      b := false;
      break;
    }
    i := i + 1;
  }
  if b {
    assert forall j | 0 <= j < |v'| && j != x :: L[j][x] * e < v'[x] - v'[j];
    ProveRobustMargin(v', e, L, x);
  }
}

method ProveRobustMargin(v': Vector, e: real, L: Matrix, x: nat)
  requires P1: |L| == |L[0]| == |v'|
  requires P2: x == ArgMax(v')
  requires P3: forall i: nat, j: nat | i < |L| && j < |L[0]| :: 0.0 <= L[i][j]
  requires P4: forall j: nat | j < |v'| && j != x :: L[j][x] * e < v'[x] - v'[j]
  ensures forall v: Vector, n: NeuralNetwork |
    IsInput(v, n) && NN(n, v) == v' && AreMarginLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  reveal P1;
  reveal P2;
  reveal P3;
  forall v: Vector, u: Vector, n: NeuralNetwork |
    IsInput(v, n) && IsInput(u, n) && NN(n, v) == v' && AreMarginLipBounds(n, L)
    && Distance(v, u) <= e
    ensures ArgMax(v') == ArgMax(NN(n, u))
  {
    forall j: nat | j < |L|
      ensures Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j])) <= L[j][x] * e
    {
      ghost var y := Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j]));
      ghost var z := L[j][x];
      DistanceInequality(y, z, e, v, u);
      assert y <= z * e;
      assert Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j])) <= L[j][x] * e by {
        assert y == Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j]));
        assert z == L[j][x];
      }
    }
    forall j: nat | j < |L| && j != x
      ensures NN(n, u)[j] < NN(n, u)[x]
    {
      assert L[j][x] * e < v'[x] - v'[j] by { reveal P4; }
      assert Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j])) <= L[j][x] * e;
      assert Abs((NN(n, v)[x] - NN(n, v)[j]) - (NN(n, u)[x] - NN(n, u)[j])) < v'[x] - v'[j];
      assert Abs((v'[x] - v'[j]) - (NN(n, u)[x] - NN(n, u)[j])) < v'[x] - v'[j];
      assert v'[x] - v'[j] >= 0.0;
      assert (v'[x] - v'[j]) - (NN(n, u)[x] - NN(n, u)[j]) < v'[x] - v'[j];
      assert v'[x] - v'[j] - (NN(n, u)[x] - NN(n, u)[j]) < v'[x] - v'[j];
      assert v'[x] - (NN(n, u)[x] - NN(n, u)[j]) < v'[x];
      assert -(NN(n, u)[x] - NN(n, u)[j]) < 0.0;
      assert - NN(n, u)[x] + NN(n, u)[j] < 0.0;
      assert NN(n, u)[j] < NN(n, u)[x];
    }
    assert forall j: nat | j < |L| && j != x :: NN(n, u)[j] < NN(n, u)[x];
    ArgMaxDef(NN(n, u), x);
    assert ArgMax(NN(n, u)) == x;
    assert ArgMax(v') == ArgMax(NN(n, u));
  }
  assert forall v: Vector, u: Vector, n: NeuralNetwork |
    IsInput(v, n) && IsInput(u, n) && NN(n, v) == v' && AreMarginLipBounds(n, L)
    && Distance(v, u) <= e :: ArgMax(v') == ArgMax(NN(n, u));
}

lemma DistanceInequality(y: real, z: real, e: real, v: Vector, u: Vector)
  requires |v| == |u|
  requires y <= z * Distance(v, u)
  requires Distance(v, u) <= e
  requires 0.0 <= z
  ensures y <= z * e
{}


/* ================================ Methods ================================= */

/**
 * Certifies the output vector v' against the error ball e and Lipschitz
 * constants L. If certification succeeds (returns true), any input
 * corresponding to v' is verified robust.
 */
method Certify(v': Vector, e: real, L: seq<real>) returns (b: bool)
  requires forall i | 0 <= i < |L| :: 0.0 <= L[i]
  requires |v'| == |L|
  ensures b ==> forall v: Vector, n: NeuralNetwork |
    IsInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  var x := ArgMaxImpl(v');
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
    IsInput(v, n) && NN(n, v) == v' && AreLipBounds(n, L) ::
    Robust(v, v', e, n)
{
  forall v: Vector, u: Vector, n: NeuralNetwork |
    IsInput(v, n) && IsInput(u, n) && NN(n, v) == v' && AreLipBounds(n, L)
    && Distance(v, u) <= e
    ensures ArgMax(v') == ArgMax(NN(n, u))
  {
    ghost var k: Vector := [L[0] * e];
    ghost var i := 1;
    ghost var old_L := L;
    ghost var old_v' := v';
    assert forall j: nat | j < i :: k[j] == L[j] * e by {
      assert k == [L[0] * e];
    }
    while i < |L|
      invariant 0 <= i <= |L|
      invariant |k| == i
      invariant forall j: nat | j < i :: k[j] == L[j] * e
      invariant L == old_L
      invariant v' == old_v'
    {
      k := k + [L[i] * e];
      assert forall j: nat | j <= i :: k[j] == L[j] * e by {
        H2(i, k, L, e);
      }
      i := i + 1;
      assert forall j: nat | j < i :: k[j] == L[j] * e;
    }
    assert |k| == |L|;
    assert forall j: nat | j < |k| :: k[j] == L[j] * e;
    assert forall j: nat | j < |v'| :: Abs(v'[j] - NN(n, u)[j]) <= k[j] by {
      assert forall j: nat | j < |v'| :: Abs(v'[j] - NN(n, u)[j]) <= L[j] * e by {
        assert |L| == |v'|;
        assert forall j: nat | j < |v'| :: IsLipBound(n, L[j], j);
        assert forall j: nat | j < |v'| :: Abs(NN(n, v)[j] - NN(n, u)[j]) <= L[j] * Distance(v, u);
        assert Distance(v, u) <= e;
        H3(v', L, v, u, n, e);
        assert forall j: nat | j < |v'| :: Abs(NN(n, v)[j] - NN(n, u)[j]) <= L[j] * e;
      }
    }
    assert forall j: nat | j < |v'| && j != x :: v'[x] - k[x] > v'[j] + k[j] by {
      assert forall j | 0 <= j < |v'| :: k[j] == L[j] * e;
      assert forall j | 0 <= j < |v'| && j != x ::
        v'[x] - L[x] * e > v'[j] + L[j] * e;
      H4(v', L, k, e, x);
    }
    SameArgMax(v', NN(n, u), x, k);
    assert ArgMax(v') == ArgMax(NN(n, u));
  }
  assert forall v: Vector, u: Vector, n: NeuralNetwork |
    IsInput(v, n) && IsInput(u, n) && NN(n, v) == v' && AreLipBounds(n, L)
    && Distance(v, u) <= e ::
    ArgMax(v') == ArgMax(NN(n, u));
}

lemma H4(v': Vector, L: Vector, k: Vector, e: real, x: int)
  requires 0 <= x < |v'|
  requires |v'| == |L| == |k|
  requires forall j | 0 <= j < |v'| :: k[j] == L[j] * e
  requires forall j | 0 <= j < |v'| && j != x :: v'[x] - L[x] * e > v'[j] + L[j] * e
  ensures forall j: nat | j < |v'| && j != x :: v'[x] - k[x] > v'[j] + k[j]
{
  
}


lemma H3(v': Vector, L: Vector, v: Vector, u: Vector, n: NeuralNetwork, e: real)
  requires IsInput(v, n) && IsInput(u, n)
  requires |v'| == |L|
  requires forall j: nat | j < |L| :: L[j] >= 0.0
  requires v' == NN(n, v)
  requires forall j: nat | j < |v'| :: Abs(NN(n, v)[j] - NN(n, u)[j]) <= L[j] * Distance(v, u)
  requires Distance(v, u) <= e
  ensures forall j: nat | j < |v'| :: Abs(NN(n, v)[j] - NN(n, u)[j]) <= L[j] * e
{
  reveal L2();
  assert forall j: nat | j < |v'| :: L[j] * Distance(v, u) >= 0.0;
  assert Distance(v, u) >= 0.0;
  assert forall j: nat | j < |v'| :: L[j] >= 0.0;
  assert forall j: nat | j < |v'| :: L[j] * Distance(v, u) <= L[j] * e;
}

lemma H2(i: int, k: Vector, L: Vector, e: real)
  requires 0 <= i < |L|
  requires |k| <= |L|
  requires |k| == i + 1
  requires forall j: nat | j < i :: k[j] == L[j] * e
  requires k[i] == L[i] * e
  ensures forall j: nat | j <= i :: k[j] == L[j] * e
{}

lemma SameArgMax(p: Vector, q: Vector, x: int, k: Vector)
  requires P1: |p| == |q| == |k|
  requires P2: 0 <= x < |p|
  requires P3: ArgMax(p) == x
  requires P4: forall i: nat | i < |p| :: Abs(p[i] - q[i]) <= k[i]
  requires P5: forall i: nat | i < |p| && i != x :: p[x] - k[x] > p[i] + k[i]
  ensures ArgMax(p) == ArgMax(q)
{
  reveal P1;
  forall i: nat | i < |p|
    ensures q[i] <= p[i] + k[i]
  {
    assert Q1: k[i] >= 0.0 by { reveal P4; }
    assert q[i] <= p[i] + k[i] by {
      reveal P4;
      assert Abs(p[i] - q[i]) <= k[i];
      assert Abs(q[i] - p[i]) <= k[i];
      reveal Q1;
      assert q[i] - p[i] <= k[i];
      assert q[i] <= k[i] + p[i];
    }
  }
  assert forall i: nat | i < |p| :: q[i] <= p[i] + k[i];
  reveal P2;
  assert q[x] >= p[x] - k[x] by {
    reveal P4;
    assert k[x] >= 0.0;
    assert Abs(p[x] - q[x]) <= k[x];
    assert p[x] - q[x] <= k[x];
    assert q[x] >= p[x] - k[x];
  }
  reveal P5;
  forall i: nat | i < |p| && i != x
    ensures q[x] > q[i]
  {
    calc {
      q[x];
      >=
      p[x] - k[x];
      >
      p[i] + k[i];
      >=
      q[i];
    }
    assert q[x] > q[i];
  }
  assert forall i: nat | i < |p| && i != x :: q[x] > q[i];
  ArgMaxDef(q, x);
  assert ArgMax(q) == x;
  reveal P3;
  assert ArgMax(q) == ArgMax(p);
}

lemma ArgMaxDef(q: Vector, x: int)
  requires 0 <= x < |q|
  requires forall i: nat | i < |q| && i != x :: q[x] > q[i]
  ensures ArgMax(q) == x
{}

/**
 * Generates the Lipschitz bound for each logit in the output of the neural
 * network n. See GenLipBound for details.
 */
method GenLipBounds(n: NeuralNetwork, s: seq<real>) returns (r: seq<real>)
  requires |s| == |n|
  requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures |r| == |n[|n|-1]|
  ensures forall i | 0 <= i < |r| :: 0.0 <= r[i]
  ensures AreLipBounds(n, r)
{
  r := [];
  var i := 0;
  while i < |n[|n|-1]|
    invariant 0 <= i <= |n[|n|-1]|
    invariant |r| == i
    invariant forall j | 0 <= j < i ::
      0.0 <= r[j] && IsLipBound(n, r[j], j)
  {
    var bound := GenLipBound(n, i, s);
    r := r + [bound];
    i := i + 1;
    assert forall j | 0 <= j < i :: IsLipBound(n, r[j], j) by {
      assert forall j | 0 <= j < i - 1 :: IsLipBound(n, r[j], j);
      assert IsLipBound(n, r[i-1], i-1);
    }
  }
  assert AreLipBounds(n, r);
}

method GenLipBound(n: NeuralNetwork, l: nat, s: seq<real>) returns (r: real)
  requires P1: |s| == |n|
  requires P2: l < Rows(n[|n|-1])
  requires P3: forall i: nat | i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures IsLipBound(n, r, l)
  ensures r >= 0.0
{
  if (|n| > 1) {
    reveal P1;
    reveal P2;
    reveal P3;
    var i := |n| - 1;
    var m: Matrix := [n[i][l]];
    r := GramIterationSimple(m);
    assert IsSpecNormUpperBound(r, m);
    assert r >= 0.0 by {
      assert IsSpecNormUpperBound(r, m);
    }
    forall v: Vector, u: Vector | |v| == |[n[i][l]][0]| && |u| == |[n[i][l]][0]|
    { Helper1(n, v, u, r, l); }
    assert IsLipBound(n[i..], r, l);
    while i > 0
      invariant 0 <= i < |n|
      invariant IsLipBound(n[i..], r, l)
      invariant r >= 0.0
    {
      ghost var old_i := i;
      i := i - 1;
      var r1 := r;
      var r2 := s[i];
      r := r2 * r1;
      assert r >= 0.0 by {
        assert r2 >= 0.0;
        assert r1 >= 0.0;
        PositiveMultiplication(r2, r1);
        assert r2 * r1 >= 0.0;
        assert r == r2 * r1;
      }
      ghost var n' := n[i..];
      assert i < |n| - 1;
      assert IsLipBound(n[i..][1..], r1, l) by {
        assert IsLipBound(n[old_i..], r1, l);
        assert IsLipBound(n[old_i..], r1, l) ==> IsLipBound(n[i+1..], r1, l);
        assert IsLipBound(n[i+1..], r1, l);
      }
      assert IsLipBound(n', r, l) by {
        assert |n'| > 1 by {
          assert n' == n[i..];
          assert |n'| == |n[i..]|;
          assert i < |n| - 1;
          assert |n[i..]| > 1;
        }
        assert l < |n'[|n'|-1]|;
        assert IsLipBound(n'[1..], r1, l);
        assert IsSpecNormUpperBound(r2, n'[0]);
        assert r == r2 * r1;
        LipBoundsRecursiveCase(n', r, r1, r2, l);
        assert IsLipBound(n', r, l);
      }
      assert IsLipBound(n[i..], r, l) by {
        assert IsLipBound(n', r, l);
      }
    }
    assert IsLipBound(n[0..], r, l);
    assert IsLipBound(n, r, l) by {
      assert IsLipBound(n[0..], r, l);
    }
    assert r >= 0.0;
  } else {
    reveal P1;
    reveal P2;
    reveal P3;
    var i := |n| - 1;
    var m: Matrix := [n[i][l]];
    r := GramIterationSimple(m);
    assert IsSpecNormUpperBound(r, m);
    assert r >= 0.0 by {
      assert IsSpecNormUpperBound(r, m);
    }
    forall v: Vector, u: Vector | |v| == |[n[i][l]][0]| && |u| == |[n[i][l]][0]|
    { Helper1(n, v, u, r, l); }
    assert IsLipBound(n[i..], r, l);
    assert IsLipBound(n, r, l) by {
      assert IsLipBound(n[i..], r, l);
    }
    assert r >= 0.0 by {
      assert IsSpecNormUpperBound(r, m);
    }
  }
  assert IsLipBound(n, r, l);
  assert r >= 0.0;
}

lemma LipBoundsRecursiveCase(n: NeuralNetwork, r: real, r1: real, r2: real, l: nat)
  requires P1: |n| > 1
  requires P2: l < |n[|n|-1]|
  requires P3: r1 >= 0.0
  requires P4: IsLipBound(n[1..], r1, l)
  requires P5: IsSpecNormUpperBound(r2, n[0])
  requires P6: r == r2 * r1
  ensures IsLipBound(n, r, l)
{
  reveal L2();
  reveal P1;
  reveal P2;
  forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n)
    ensures Abs(NN(n, v)[l] - NN(n, u)[l]) <= r * Distance(v, u)
  {
    var v': Vector := Layer(n[0], v);
    var u': Vector := Layer(n[0], u);
    assert Q1: Abs(NN(n[1..], v')[l] - NN(n[1..], u')[l]) <= r1 * Distance(v', u') by {
      reveal P4;
      assert IsInput(v', n[1..]);
      assert IsInput(u', n[1..]);
    }
    assert Q2: Abs(NN(n, v)[l] - NN(n, u)[l]) <= r1 * Distance(v', u') by {
      reveal Q1;
      NeuralNetDefinition(n, v);
      NeuralNetDefinition(n, u);
    }
    assert Q3: r1 * Distance(v', u') <= r1 * r2 * Distance(v, u) by {
      reveal P5;
      SpecNormIsLayerLipBound(n[0], v, u, r2);
      assert Distance(Layer(n[0], v), Layer(n[0], u)) <= r2 * Distance(v, u);
      assert Distance(v', u') <= r2 * Distance(v, u);
      assert r1 * Distance(v', u') <= r1 * r2 * Distance(v, u) by {
        reveal P3;
        H(v, u, v', u', r1, r2);
      }
    }
    assert Q4: Abs(NN(n, v)[l] - NN(n, u)[l]) <= r1 * r2 * Distance(v, u) by {
      calc {
        r1 * r2 * Distance(v, u);
        >=
        {
          reveal Q3;
        }
        r1 * Distance(v', u');
        >=
        {
          reveal Q2;
        }
        Abs(NN(n, v)[l] - NN(n, u)[l]);
      }
    }
    assert Abs(NN(n, v)[l] - NN(n, u)[l]) <= r * Distance(v, u) by {
      reveal Q4;
      reveal P6;
    }
  }
  assert forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n)
    :: Abs(NN(n, v)[l] - NN(n, u)[l]) <= r * Distance(v, u);
  assert IsLipBound(n, r, l);
}

lemma H(v: Vector, u: Vector, v': Vector, u': Vector, r1: real, r2: real)
  requires |v'| == |u'|
  requires |v| == |u|
  requires r1 >= 0.0
  requires r2 >= 0.0
  requires Distance(v', u') <= r2 * Distance(v, u)
  ensures r1 * Distance(v', u') <= r1 * r2 * Distance(v, u)
{
  reveal L2();
}

lemma Helper1(n: NeuralNetwork, v: Vector, u: Vector, r: real, l: nat)
  requires |n| >= 1
  requires l < Rows(n[|n|-1])
  requires IsSpecNormUpperBound(r, [n[|n|-1][l]])
  requires |v| == |[n[|n|-1][l]][0]|
  requires |u| == |[n[|n|-1][l]][0]|
  ensures Abs(NN(n[|n|-1..], v)[l] - NN(n[|n|-1..], u)[l]) <= r * Distance(v, u)
{
  var i := |n| - 1;
  var m: Matrix := [n[i][l]];
  assert IsSpecNormUpperBound(r, m);
  SpecNormIsLayerLipBound(m, v, u, r);
  assert Distance(Layer(m, v), Layer(m, u)) <= r * Distance(v, u);
  calc {
    Layer(m, v);
    ==
    Layer([n[i][l]], v);
    ==
    [NN(n[i..], v)[l]];
  }
  calc {
    Layer(m, u);
    ==
    Layer([n[i][l]], u);
    ==
    [NN(n[i..], u)[l]];
  }
  assert Distance([NN(n[i..], v)[l]], [NN(n[i..], u)[l]]) <= r * Distance(v, u);
  calc {
    Distance([NN(n[i..], v)[l]], [NN(n[i..], u)[l]]);
    ==
    {
      NormOfOneDimensionIsAbs();
    }
    Abs(NN(n[i..], v)[l] - NN(n[i..], u)[l]);
  }
  assert Abs(NN(n[i..], v)[l] - NN(n[i..], u)[l]) <= r * Distance(v, u);
}

lemma NeuralNetDefinition(n: NeuralNetwork, v: Vector)
  requires |v| == |n[0][0]|
  requires |n| > 1
  ensures NN(n, v) == NN(n[1..], Layer(n[0], v))
{
  reveal NN();
  if |n| == 2 {
    calc {
      NN(n, v);
      ==
      Layer(n[|n|-1], NN(n[..|n|-1], v));
      ==
      Layer(n[1], Layer(n[0], v));
      ==
      Layer(n[1..][0], Layer(n[0], v));
      ==
      NN(n[1..], Layer(n[0], v));
    }
  } else {
    calc {
      NN(n, v);
      ==
      Layer(n[|n|-1], NN(n[..|n|-1], v));
      ==
      {
        NeuralNetDefinition(n[..|n|-1], v);
      }
      Layer(n[|n|-1], NN(n[..|n|-1][1..], Layer(n[..|n|-1][0], v)));
      ==
      Layer(n[|n|-1], NN(n[1..|n|-1], Layer(n[0], v)));
      ==
      calc {
        n[1..|n|-1];
        ==
        n[1..][..|n[1..]|-1];
      }
      Layer(n[|n|-1], NN(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      Layer(n[|n|-1], NN(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      Layer(n[1..][|n[1..]|-1], NN(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      NN(n[1..], Layer(n[0], v));
    }
  }
}

/* ================================= Lemmas ================================= */

/**
 * Sometimes Dafny needs a separate lemma to prove the obvious. In short,
 * this lemma proves that if this (rather verbose) property holds for all i
 * in S, and x is in S, then it also holds specifically for x.
 */
lemma ProveRobustHelper(v': Vector, e: real, L: seq<real>, x: int)
  requires |v'| == |L|
  requires x == ArgMax(v')
  requires forall n: NeuralNetwork, v: Vector, u: Vector, i |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) && 0 <= i < |L| &&
    |v| == |u| && IsInput(v, n) && Distance(v, u) <= e ::
    Abs(NN(n, v)[i] - NN(n, u)[i]) <= L[i] * e
  ensures forall n: NeuralNetwork, v: Vector, u: Vector |
    |n[|n|-1]| == |L| && AreLipBounds(n, L) &&
    |v| == |u| && IsInput(v, n) && Distance(v, u) <= e ::
    Abs(NN(n, v)[x] - NN(n, u)[x]) <= L[x] * e
{}

/**
 * The product of the spectral norms of each matrix of a neural network n is
 * a Lipschitz bound on the l2 norm of the output vector of n.
 */
lemma SpecNormProductIsLipBound(n: NeuralNetwork, v: Vector, u: Vector,
    s: seq<real>)
  requires |v| == |u| && |s| == |n|
  requires IsInput(v, n) && IsInput(u, n)
  requires forall i | 0 <= i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures Distance(NN(n, v), NN(n, u)) <= Product(s) * Distance(v, u)
{
  if |n| == 1 {
    SpecNormIsLayerLipBound(n[0], v, u, s[0]);
    reveal Product();
  } else {
    SpecNormIsLayerLipBound(n[0], v, u, s[0]);

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

/** An obvious helper-lemma to SpecNormProductIsLipBound. */
lemma MultiplyBothSides(s: seq<real>, s0: seq<real>, s': real, v: Vector,
    u: Vector)
  requires |v| == |u|
  requires Product(s) == s' * Product(s0)
  ensures Product(s) * Distance(v, u) == s' * Product(s0) * Distance(v, u)
{}

/** An obvious helper-lemma to SpecNormProductIsLipBound. */
lemma MultiplicationInequality(n: NeuralNetwork, v: Vector, u: Vector,
    v': Vector, u': Vector, s0: seq<real>, s': real)
  requires |v| == |u|
  requires |v'| == |u'|
  requires s' >= 0.0
  requires IsInput(v, n) && IsInput(u, n)
  requires Distance(NN(n, v), NN(n, u)) <= s' * Distance(v', u')
  requires Distance(v', u') <= Product(s0) * Distance(v, u)
  ensures Distance(NN(n, v), NN(n, u)) <= s' * Product(s0) * Distance(v, u)
{}

/**
 * As seen in the method GenLipBound, computing a Lipschitz bound on logit l
 * for a neural network n involves 'trimming' all rows out of the final layer
 * of n except for row l, and computing the spectral norm of this new neural
 * network n'. This lemma relates a Lipschitz bound on the output vector of
 * n' to a Lipschitz bound on the logit l in n.
 */
lemma LogitLipBounds(n: NeuralNetwork, n': NeuralNetwork, v: Vector,
    u: Vector, l: int)
  requires |v| == |u|
  requires |n| == |n'|
  requires IsInput(v, n)
  requires IsInput(u, n)
  requires 0 <= l < |n[|n|-1]|
  requires n' == n[..|n|-1] + [[n[|n|-1][l]]]
  ensures Distance(NN(n', v), NN(n', u)) == Abs(NN(n, v)[l] - NN(n, u)[l])
{
  TrimmedNN(n, n', v, l);
  TrimmedNN(n, n', u, l);
  NormOfOneDimensionIsAbs();
}

/**
 * The distance between two vectors can only be decreased when the ReLu
 * function is applied to each one. This is equivalent to stating that the
 * spectral norm of the ReLu layer is 1.
 * ||R(v) - R(u)|| <= ||v - u|| where R applies the ReLu activation function.
 */
lemma SmallerRelu(v: Vector, u: Vector)
  requires |v| == |u|
  ensures Distance(ApplyRelu(v), ApplyRelu(u)) <= Distance(v, u)
{
  SmallerL2Norm(Minus(ApplyRelu(v), ApplyRelu(u)), Minus(v, u));
}

/**
 * A neural network layer consists of matrix-vector multiplication, followed
 * by an application of the ReLu activation function. A Lipschitz bound of a
 * layer with matrix m is the spectral norm of that matrix.
 * ||R(m.v) - R(m.u)|| <= ||m|| * ||v - u||
 * where R applies the ReLu activation function.
 */
lemma SpecNormIsLayerLipBound(m: Matrix, v: Vector, u: Vector, s: real)
  requires |m[0]| == |v| == |u|
  requires IsSpecNormUpperBound(s, m)
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
  requires IsSpecNormUpperBound(s, m)
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
  requires IsSpecNormUpperBound(s, m)
  ensures L2(MV(m, Minus(v, u))) <= s * Distance(v, u)
{}
}
