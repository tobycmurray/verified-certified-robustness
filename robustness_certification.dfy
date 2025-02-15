include "basic_arithmetic.dfy"
include "linear_algebra.dfy"
include "operator_norms.dfy"
include "neural_networks.dfy"
include "l2_extra.dfy"

module RobustnessCertification {
import opened BasicArithmetic
import opened LinearAlgebra
import opened OperatorNorms
import opened NeuralNetworks
import opened L2Extra

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

/* ============================= Margin Bounds ============================== */

ghost predicate IsMarginLipBound(n: NeuralNetwork, r: real, p: nat, q: nat)
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]|
{
  forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u)
}

lemma MarginLipBoundDef(n: NeuralNetwork, r: real, p: nat, q: nat)
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]|
  requires IsMarginLipBound(n, r, p, q)
  ensures forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u)
{}

ghost predicate AreMarginLipBounds(n: NeuralNetwork, L: Matrix)
  requires |L| == |L[0]| == |n[|n|-1]|
{
  forall p: nat, q: nat | p < |L| && q < |L[0]| :: IsMarginLipBound(n, L[p][q], p, q)
}

/* ================================ Methods ================================= */

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
    ProveRobust(v', e, L, x);
  }
}

method L2UpperBound(v: Vector) returns (r: real)
  ensures r >= L2(v)
{
  reveal L2();
  var v' := ApplyImpl(v, Square);
  r := SumImpl(v');
  r := SqrtUpperBound(r);
}

method GenMarginBound(n: NeuralNetwork, p: nat, q: nat, s: seq<real>) returns (r: real)
  requires P1: |s| == |n|
  requires P2: p < |n[|n|-1]|
  requires P3: q < |n[|n|-1]|
  requires P4: forall i: nat | i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures IsMarginLipBound(n, r, p, q)
  ensures r >= 0.0
{
  reveal P1;
  reveal P2;
  reveal P3;
  var i := |n| - 1;
  var d: Vector := MinusImpl(n[i][q], n[i][p]);
  var m: Matrix := [d];
  r := L2UpperBound(d);
  L2IsSpecNormUpperBound(r, m);
  assert IsSpecNormUpperBound(r, m);
  SpecNormIsMarginLipBound(n[i..], m, p, q, r);
  assert IsMarginLipBound(n[i..], r, p, q);
  while i > 0
    invariant r >= 0.0
    invariant IsMarginLipBound(n[i..], r, p, q)
  {
    i := i - 1;
    var b := s[i];
    assert b >= 0.0 by { reveal P4; }
    assert IsSpecNormUpperBound(b, n[i]) by { reveal P4; }
    ghost var n' := n[i..];
    var r' := b * r;

    assert Q1: r' == b * r;
    assert Q2: r >= 0.0;
    assert Q3: IsMarginLipBound(n'[1..], r, p, q);

    assert IsMarginLipBound(n', r', p, q) by {
      reveal P4;
      assert |n'| > 1;
      assert p < |n'[|n'|-1]|;
      assert q < |n'[|n'|-1]|;
      assert IsSpecNormUpperBound(b, n'[0]);
      reveal Q3;
      assert IsMarginLipBound(n'[1..], r, p, q);
      assert r' == b * r by { reveal Q1; }
      assert r >= 0.0 by { reveal Q2; }
      reveal Q1;
      reveal Q2;
      reveal Q3;
      MarginRecursive(n', b, r, p, q, r');
    }
    assert IsMarginLipBound(n[i..], r', p, q) by {
      X1(n, n', r', p, q, i);
    }
    r := r';
    X2(n, r', r, p, q, i);
  }
}

lemma X2(n: NeuralNetwork, r': real, r: real, p: nat, q: nat, i: nat)
  requires i < |n|
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]|
  requires IsMarginLipBound(n[i..], r', p, q)
  requires r == r'
  ensures IsMarginLipBound(n[i..], r, p, q)
{}

lemma X1(n: NeuralNetwork, n': NeuralNetwork, r: real, p: nat, q: nat, i: nat)
  requires i < |n|
  requires n' == n[i..]
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]| 
  requires IsMarginLipBound(n', r, p, q)
  ensures IsMarginLipBound(n[i..], r, p, q)
{}

lemma SpecNormIsMarginLipBound(n: NeuralNetwork, m: Matrix, p: nat, q: nat, r: real)
  requires P1: |n| == 1
  requires P2: p < |n[0]| && q < |n[0]|
  requires P3: m == [Minus(n[0][q], n[0][p])]
  requires P4: IsSpecNormUpperBound(r, m)
  ensures IsMarginLipBound(n, r, p, q)
{
  reveal P1;
  reveal P2;
  forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n)
    ensures Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u)
  {
    assert r >= 0.0 by { reveal P4; }
    assert L2(MV(m, Minus(v, u))) <= r * L2(Minus(v, u)) by {
      reveal P3;
      reveal P4;
    }
    assert L2(MV(m, Minus(v, u))) <= r * Distance(v, u);
    calc {
      L2(MV(m, Minus(v, u)));
      ==
      {
        NormOfOneDimensionIsAbs();
        reveal P3;
      }
      Abs(MV(m, Minus(v, u))[0]);
      ==
      calc {
        MV(m, Minus(v, u))[0];
        ==
        Dot(m[0], Minus(v, u));
        ==
        {
          DotIsDistributive(m[0], v, u);
        }
        Dot(m[0], v) - Dot(m[0], u);
        ==
        {
          reveal P3;
        }
        Dot(Minus(n[0][q], n[0][p]), v) - Dot(Minus(n[0][q], n[0][p]), u);
        ==
        calc {
          Dot(Minus(n[0][q], n[0][p]), v);
          ==
          {
            DotIsCommutative(Minus(n[0][q], n[0][p]), v);
          }
          Dot(v, Minus(n[0][q], n[0][p]));
          ==
          {
            DotIsDistributive(v, n[0][q], n[0][p]);
          }
          Dot(v, n[0][q]) - Dot(v, n[0][p]);
          ==
          {
            DotIsCommutative(v, n[0][q]);
            DotIsCommutative(v, n[0][p]);
          }
          Dot(n[0][q], v) - Dot(n[0][p], v);
        }
        calc {
          Dot(Minus(n[0][q], n[0][p]), u);
          ==
          {
            DotIsCommutative(Minus(n[0][q], n[0][p]), u);
          }
          Dot(u, Minus(n[0][q], n[0][p]));
          ==
          {
            DotIsDistributive(u, n[0][q], n[0][p]);
          }
          Dot(u, n[0][q]) - Dot(u, n[0][p]);
          ==
          {
            DotIsCommutative(u, n[0][q]);
            DotIsCommutative(u, n[0][p]);
          }
          Dot(n[0][q], u) - Dot(n[0][p], u);
        }
        (Dot(n[0][q], v) - Dot(n[0][p], v)) - (Dot(n[0][q], u) - Dot(n[0][p], u));
        ==
        Dot(n[0][q], v) - Dot(n[0][p], v) - (Dot(n[0][q], u) - Dot(n[0][p], u));
        ==
        NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p]);
      }
      Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p]));
    }
    assert Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u);
  }
  assert forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v, u);
  assert IsMarginLipBound(n, r, p, q);
}

lemma MarginRecursive(n: NeuralNetwork, s: real, r: real, p: nat, q: nat, r': real)
  requires P1: |n| > 1
  requires P2: p < |n[|n|-1]|
  requires P3: q < |n[|n|-1]|
  requires P4: IsSpecNormUpperBound(s, n[0])
  requires P5: IsMarginLipBound(n[1..], r, p, q)
  requires P6: r' == s * r
  requires P7: r >= 0.0
  ensures IsMarginLipBound(n, r', p, q)
{
  reveal P1;
  reveal P2;
  reveal P3;
  forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n)
    ensures Abs(NN(n, v)[q] - NN(n, v)[p] - (NN(n, u)[q] - NN(n, u)[p])) <= r' * Distance(v, u)
  {
    ghost var v': Vector := Layer(n[0], v);
    ghost var u': Vector := Layer(n[0], u);
    assert Q1: Abs(NN(n[1..], v')[q] - NN(n[1..], v')[p] - (NN(n[1..], u')[q] - NN(n[1..], u')[p])) <= r * Distance(v', u') by {
      reveal P5;
      assert IsInput(v', n[1..]);
      assert IsInput(u', n[1..]);
    }
    assert Q2: Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r * Distance(v', u') by {
      reveal Q1;
      NeuralNetDefinition(n, v);
      NeuralNetDefinition(n, u);
    }
    assert Q3: r * Distance(v', u') <= r * s * Distance(v, u) by {
      reveal P4;
      SpecNormIsLayerLipBound(n[0], v, u, s);
      assert Distance(Layer(n[0], v), Layer(n[0], u)) <= s * Distance(v, u);
      assert Distance(v', u') <= s * Distance(v, u);
      assert r * Distance(v', u') <= r * s * Distance(v, u) by {
        reveal P3;
        reveal P7;
        H(v, u, v', u', r, s);
        assert r * Distance(v', u') <= r * s * Distance(v, u);
      }
    }
    assert Q4: Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r * s * Distance(v, u) by {
      calc {
        r * s * Distance(v, u);
        >=
        {
          reveal Q3;
        }
        r * Distance(v', u');
        >=
        {
          reveal Q2;
        }
        Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p]));
      }
    }
    assert Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r' * Distance(v, u) by {
      reveal Q4;
      reveal P6;
      X3(n, v, u, p, q, r', s, r);
    }
  }
  assert forall v: Vector, u: Vector | IsInput(v, n) && IsInput(u, n) ::
    Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r' * Distance(v, u);
  assert IsMarginLipBound(n, r', p, q);
}

lemma ProveRobust(v': Vector, e: real, L: Matrix, x: nat)
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
      ensures Abs(NN(n, v)[x] - NN(n, v)[j] - (NN(n, u)[x] - NN(n, u)[j])) <= L[j][x] * e
    {
      ghost var y := Abs(NN(n, v)[x] - NN(n, v)[j] - (NN(n, u)[x] - NN(n, u)[j]));
      ghost var z := L[j][x];
      assert y <= z * Distance(v, u) by {
        assert Abs(NN(n, v)[x] - NN(n, v)[j] - (NN(n, u)[x] - NN(n, u)[j])) <= z * Distance(v, u) by {
          assert IsMarginLipBound(n, z, j, x);
          MarginLipBoundDef(n, z, j, x);
          assert forall v': Vector, u': Vector | IsInput(v', n) && IsInput(u', n) ::
            Abs(NN(n, v')[x] - NN(n, v')[j] - (NN(n, u')[x] - NN(n, u')[j])) <= z * Distance(v', u') by {
              reveal IsMarginLipBound();
            }
        }
      }
      DistanceInequality(y, z, e, v, u);
      assert Abs(NN(n, v)[x] - NN(n, v)[j] - (NN(n, u)[x] - NN(n, u)[j])) <= L[j][x] * e by {
        calc {
          Abs(NN(n, v)[x] - NN(n, v)[j] - (NN(n, u)[x] - NN(n, u)[j]));
          ==
          y;
          <=
          z * e;
          ==
          L[j][x] * e;
        }
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

lemma NeuralNetDefinition(n: NeuralNetwork, v: Vector)
  requires |v| == |n[0][0]|
  requires |n| > 1
  ensures NN(n, v) == NN(n[1..], Layer(n[0], v))
{
  reveal NN();
  reveal NNBody();
  if |n| == 2 {
    calc {
      NN(n, v);
      ==
      MV(n[|n|-1], NNBody(n[..|n|-1], v));
      ==
      MV(n[|n|-1], Layer(n[0], v));
      ==
      MV(n[1], Layer(n[0], v));
      ==
      NN(n[1..], Layer(n[0], v));
    }
  } else {
    calc {
      NN(n, v);
      ==
      MV(n[|n|-1], NNBody(n[..|n|-1], v));
      ==
      {
        NeuralNetBodyDefinition(n[..|n|-1], v);
      }
      MV(n[|n|-1], NNBody(n[..|n|-1][1..], Layer(n[..|n|-1][0], v)));
      ==
      MV(n[|n|-1], NNBody(n[1..|n|-1], Layer(n[0], v)));
      ==
      calc {
        n[|n|-1];
        ==
        n[1..][|n[1..]|-1];
      }
      calc {
        n[1..|n|-1];
        ==
        n[1..][..|n[1..]|-1];
      }
      MV(n[1..][|n[1..]|-1], NNBody(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      NN(n[1..], Layer(n[0], v));
    }
  }
}

lemma NeuralNetBodyDefinition(n: NeuralNetwork, v: Vector)
  requires |v| == |n[0][0]|
  requires |n| > 1
  ensures NNBody(n, v) == NNBody(n[1..], Layer(n[0], v))
{
  reveal NNBody();
  if |n| == 2 {
    calc {
      NNBody(n, v);
      ==
      Layer(n[|n|-1], NNBody(n[..|n|-1], v));
      ==
      Layer(n[1], Layer(n[0], v));
      ==
      Layer(n[1..][0], Layer(n[0], v));
      ==
      NNBody(n[1..], Layer(n[0], v));
    }
  } else {
    calc {
      NNBody(n, v);
      ==
      Layer(n[|n|-1], NNBody(n[..|n|-1], v));
      ==
      {
        NeuralNetDefinition(n[..|n|-1], v);
      }
      Layer(n[|n|-1], NNBody(n[..|n|-1][1..], Layer(n[..|n|-1][0], v)));
      ==
      Layer(n[|n|-1], NNBody(n[1..|n|-1], Layer(n[0], v)));
      ==
      calc {
        n[1..|n|-1];
        ==
        n[1..][..|n[1..]|-1];
      }
      Layer(n[|n|-1], NNBody(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      Layer(n[|n|-1], NNBody(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      Layer(n[1..][|n[1..]|-1], NNBody(n[1..][..|n[1..]|-1], Layer(n[0], v)));
      ==
      NNBody(n[1..], Layer(n[0], v));
    }
  }
}

method GenMarginBounds(n: NeuralNetwork, s: seq<real>) returns (r: Matrix)
  requires P1: |s| == |n|
  requires P2: forall i: nat | i < |s| :: IsSpecNormUpperBound(s[i], n[i])
  ensures |r| == |r[0]| == |n[|n|-1]|
  ensures forall p: nat, q: nat | p < |r| && q < |r[0]| :: 0.0 <= r[p][q]
  ensures AreMarginLipBounds(n, r)
{
  reveal P1;
  var r': seq<seq<real>> := [];
  var p: nat := 0;
  while p < |n[|n|-1]|
    // proportions
    invariant p <= |n[|n|-1]|
    invariant |r'| == p
    invariant forall i: nat | i < p :: |r'[i]| == |n[|n|-1]|
    // lip bounds
    invariant forall i: nat, j: nat | i < p && j < |n[|n|-1]| :: IsMarginLipBound(n, r'[i][j], i, j) && 0.0 <= r'[i][j]
  {
    var p_bounds: seq<real> := [];
    var q: nat := 0;
    while q < |n[|n|-1]|
      // proportions
      invariant q <= |n[|n|-1]|
      invariant |p_bounds| == q
      // lip bounds
      invariant forall i: nat | i < q :: IsMarginLipBound(n, p_bounds[i], p, i) && 0.0 <= p_bounds[i]
    {
      reveal P2;
      var bound: real := GenMarginBound(n, p, q, s);
      assert 0.0 <= bound;
      assert IsMarginLipBound(n, bound, p, q);
      assert forall i: nat | i < |p_bounds| :: IsMarginLipBound(n, p_bounds[i], p, i) && 0.0 <= p_bounds[i];
      p_bounds := p_bounds + [bound];
      assert forall i: nat | i < |p_bounds| :: IsMarginLipBound(n, p_bounds[i], p, i) && 0.0 <= p_bounds[i] by {
        assert forall i: nat | i < |p_bounds|-1 :: IsMarginLipBound(n, p_bounds[i], p, i) && 0.0 <= p_bounds[i];
        assert IsMarginLipBound(n, p_bounds[|p_bounds|-1], p, |p_bounds|-1);
      }
      q := q + 1;
      assert q <= |n[|n|-1]|;
      assert |p_bounds| == q;
      assert forall i: nat | i < q :: IsMarginLipBound(n, p_bounds[i], p, i) && 0.0 <= p_bounds[i];
    }
    r' := r' + [p_bounds];
    p := p + 1;
  }
  r := r';
}

/* ================================ Lemmas ================================= */

// CERTIFICATION METHOD

lemma DistanceInequality(y: real, z: real, e: real, v: Vector, u: Vector)
  requires |v| == |u|
  requires y <= z * Distance(v, u)
  requires Distance(v, u) <= e
  requires 0.0 <= z
  ensures y <= z * e
{}

lemma ArgMaxDef(q: Vector, x: int)
  requires 0 <= x < |q|
  requires forall i: nat | i < |q| && i != x :: q[x] > q[i]
  ensures ArgMax(q) == x
{}

// MARGIN BOUNDS GENERATION

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

lemma X3(n: NeuralNetwork, v: Vector, u: Vector, p: nat, q: nat, r': real, s: real, r: real)
  requires r' == s * r
  requires IsInput(v, n)
  requires IsInput(u, n)
  requires p < |n[|n|-1]|
  requires q < |n[|n|-1]|
  requires Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r * s * Distance(v, u)
  ensures Abs((NN(n, v)[q] - NN(n, v)[p]) - (NN(n, u)[q] - NN(n, u)[p])) <= r' * Distance(v, u)
{
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