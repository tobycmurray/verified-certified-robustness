include "basic_arithmetic.dfy"

module Parsing {
import opened BasicArithmetic

datatype Maybe<T> = None | Some(val: T)

/* ========================================================================== */
/* =============================== New Parser =============================== */
/* ========================================================================== */

/* ================================ Grammar ================================= */

ghost predicate G_IsDigit(s: char) {
  s == '0' || s == '1' || s == '2' || s == '3' || s == '4' || s == '5' ||
  s == '6' || s == '7' || s == '8' || s == '9'
}

// Digits ::== Digit || Digits Digit
ghost predicate G_IsDigits(s: string) {
  |s| > 0 && forall i: nat | i < |s| :: G_IsDigit(s[i])
}

// PositiveReal ::== Digits "." Digits
ghost predicate G_IsPositiveReal(s: string) {
  exists s1: string, s2: string :: s == s1 + "." + s2 && G_IsDigits(s1) && G_IsDigits(s2)
}

// Real ::== PositiveReal | "-" PositiveReal
ghost predicate G_IsReal(s: string) {
  G_IsPositiveReal(s) || (|s| > 1 && s[0] == "-" && G_IsPositiveReal(s[1..]))
}

// RealList ::== Real | Real "," RealList
ghost predicate G_IsRealList(s: string) {
  G_IsReal(s) || exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsReal(s1) && G_IsRealList(s2)
}

// Vector ::== "[" RealList "]"
ghost predicate G_IsVector(s: string) {
  |s| > 1 && s[0] == '[' && G_IsRealList(s[1..|s|-1]) && s[|s|-1] == ']'
}

// VectorList ::== Vector | Vector "," VectorList
ghost predicate G_IsVectorList(s: string) {
  G_IsVector(s) || exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsVector(s1) && G_IsVectorList(s2)
}

// Matrix ::== "[" VectorList "]"
ghost predicate G_IsMatrix(s: string) {
  |s| > 1 && s[0] == '[' && G_IsVectorList(s[1..|s|-1]) && s[|s|-1] == ']'
}

// NeuralNet ::== Matrix | Matrix "," NeuralNet
ghost predicate G_IsNeuralNet(s: string) {
  G_IsMatrix(s) || exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsMatrix(s1) && G_IsNeuralNet(s2)
}

/* ============================= Parsing Specs ============================== */

ghost function G_ToDigit(c: char): real
  requires G_IsDigit(c)
{
  if x == '0' then 0.0
  else if x == '1' then 1.0
  else if x == '2' then 2.0
  else if x == '3' then 3.0
  else if x == '4' then 4.0
  else if x == '5' then 5.0
  else if x == '6' then 6.0
  else if x == '7' then 7.0
  else if x == '8' then 8.0
  else 9.0
}

ghost function G_ToNat(s: string): real
  requires G_IsDigits(s)
{
  if |s| == 1 then G_ToDigit(s[0]) else G_ToDigit(s[0]) * 10.0 + G_ToNat(s[1..])
}

ghost predicate G_IsPositiveRealOf(x: real, s: string)
  requires G_IsPositiveReal(s)
{
  exists s1: string, s2: string :: s == s1 + "." + s2 && G_IsDigits(s1) && G_IsDigits(s2) &&
    x == G_ToNat(s1) + (G_ToNat(s2) / Pow(10, |s2|))
}

ghost predicate G_IsRealOf(x: real, s: string)
  requires G_IsReal(s)
{
  (G_IsPositiveReal(s) && G_IsPositiveRealOf(x, s)) || 
  (|s| > 1 && s[0] == "-" && G_IsPositiveReal(s[1..]) && exists y: real :: G_IsPositiveRealOf(y, s[1..]) && x == -y)
}

ghost predicate G_IsVectorOfRealList(v: Vector, s: string)
  requires G_IsRealList(s)
{
  (G_IsReal(s)
    && |v| == 1 && G_IsRealOf(v[0], s)) ||
  exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsReal(s1) && G_IsRealList(s2)
    && |v| > 1 && G_IsRealOf(v[0], s1) && G_IsVectorOfRealList(v[1..], s2)
}

ghost predicate G_IsVectorOf(v: Vector, s: string)
  requires G_IsVector(s)
{
  G_IsVectorOfRealList(v, s[1..|s|-1])
}

ghost predicate G_IsMatrixOfVectorList(m: Matrix, s: string)
  requires G_IsVectorList(s)
{
  (G_IsVector(s)
    && |m| == 1 && G_IsVectorOf(m[0], s)) ||
  exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsVector(s1) && G_IsVectorList(s2)
    && |m| > 1 && G_IsVectorOf(m[0], s1) && G_IsMatrixOfVectorList(m[1..], s2)
}

ghost predicate G_IsMatrixOf(m: Matrix, s: string)
  requires G_IsMatrix(s)
{
  G_IsMatrixOfVectorList(m, s[1..|s|-1])
}

ghost predicate G_IsNeuralNetOf(r: NeuralNetwork, s: string)
  requires G_IsNeuralNet(s)
{
  (G_IsMatrix(s)
    && |r| == 1 && G_IsMatrixOf(r[0], s)) ||
  exists s1: string, s2: string :: s == s1 + "," + s2 && G_IsMatrix(s1) && G_IsNeuralNet(s2)
    && |r| > 1 && G_IsMatrixOf(r[0], s1) && G_IsNeuralNetOf(r[1..], s2)
}

/* ============================ Parsing Methods ============================= */

/*
Todo:

Implement a parser:
method ParseNeuralNet(s: string) returns (r: Maybe<NeuralNetwork>)
  ensures if IsNeuralNet(s) then r.Some? && IsNeuralNetOf(r.val, s) else r.None?

We are also yet to add well-formedness constraints. It's unclear where these
should go.

After implementing, we can remove the "G_" prefix. This is there for now just to
avoid naming clashes with the current parser.
*/

/* ========================================================================== */
/* =============================== Old Parser =============================== */
/* ========================================================================== */

method ParseNeuralNet(xs: string) returns (t: Maybe<NeuralNetwork>) {
  var err: string := "";
  var matrices: seq<Matrix> := [];
  var i := 0;
  while i < |xs| {
    // expect matrix
    if i >= |xs| - 1 || xs[i..i+2] != "[[" {
      return None;
    }
    var j := i + 2;
    while xs[j-2..j] != "]]"
      invariant j <= |xs|
      decreases |xs| - j
    {
      if j >= |xs| {
        return None;
      }
      j := j + 1;
    }
    // xs[i..j] == "[[...],...,[...]]"
    var ys := xs[i+1..j-1];
    // ys == "[...],...,[...]"
    var k := 0;
    var vectors: seq<Vector> := [];
    while k < |ys| {
      // Expect vector
      if ys[k] != '[' {
        return None;
      }
      var l := k;
      while ys[l] != ']'
        invariant l < |ys|
        decreases |ys| - l
      {
        if l + 1 >= |ys| {
          return None;
        }
        l := l + 1;
      }
      // ys[k..l] == "[r1,r2,...,rn"
      var zs := ys[k+1..l];
      // zs == "r1,r2,...,rn"
      var realsStr: seq<string> := StringUtils.Split(zs, ',');
      var areReals: bool := AreReals(realsStr);
      if !areReals {
        return None;
      }
      var v: seq<real> := ParseReals(realsStr);
      if |v| == 0 {
        return None;
      }
      var v': Vector := v;
      vectors := vectors + [v'];
      k := l + 2; // skip comma
    }
    var matrixWellFormed := IsMatrixWellFormed(vectors);
    if !matrixWellFormed {
      return None;
    }
    var matrix: Matrix := vectors;
    matrices := matrices + [Transpose(matrix)]; // need to transpose for comptability with python output
    i := j + 1; // xs[j] == ',' or EOF
  }
  var neuralNetWellFormed := IsNeuralNetWellFormed(matrices);
  if !neuralNetWellFormed {
    return None;
  }
  var neuralNet: NeuralNetwork := matrices;
  return Some(neuralNet);
}

method IsNeuralNetWellFormed(n: seq<Matrix>) returns (b: bool)
  ensures b ==>
    |n| > 0 &&
    forall i: int :: 0 <= i < |n| - 1 ==> |n[i]| == |n[i + 1][0]|
{
  if |n| == 0 {
    return false;
  }
  var i := 0;
  while i < |n| - 1
    invariant 0 <= i <= |n| - 1
    invariant forall j | 0 <= j < i :: |n[j]| == |n[j + 1][0]|
  {
    if |n[i]| != |n[i + 1][0]| {
      return false;
    }
    i := i + 1;
  }
  return true;
}

method IsMatrixWellFormed(m: seq<seq<real>>) returns (b: bool)
  ensures b ==>
    |m| > 0 &&
    |m[0]| > 0 &&
    forall i, j: int :: 0 <= i < |m| && 0 <= j < |m| ==> |m[i]| == |m[j]|
{
  if |m| == 0 || |m[0]| == 0 {
    return false;
  }
  var size := |m[0]|;
  var i := 1;
  while i < |m|
    invariant 0 <= i <= |m|
    invariant forall j | 0 <= j < i :: |m[j]| == size
  {
    if |m[i]| != size {
      return false;
    }
    i := i + 1;
  }
  return true;
}


method AreReals(realsStr: seq<string>) returns (b: bool)
  ensures b ==> forall i | 0 <= i < |realsStr| :: StringUtils.IsReal(realsStr[i])
{
  for i := 0 to |realsStr|
    invariant forall j | 0 <= j < i :: StringUtils.IsReal(realsStr[j])
  {
    var isReal := StringUtils.IsReal(realsStr[i]);
    if !isReal {
      print realsStr[i];
      print "\n";
      return false;
    }
  }
  return true;
}

method ParseReals(realsStr: seq<string>) returns (reals: seq<real>)
  requires forall i | 0 <= i < |realsStr| :: StringUtils.IsReal(realsStr[i])
{
  reals := [];
  for i := 0 to |realsStr| {
    var r := StringUtils.ParseReal(realsStr[i]);
    reals := reals + [r];
  }
}


/**
 * Returns the real number represented by s.
 * Note: This method is not verified.
 */
method ParseReal(s: string) returns (r: real)
  requires IsReal(s)
{
  var neg: bool := false;
  var i: int := 0;
  if s[i] == '-' {
    neg := true;
    i := i + 1;
  }
  r := ParseDigit(s[i]) as real;
  i := i + 1;
  var periodIndex: int := 1;
  while i < |s| {
    if IsDigit(s[i]) {
      r := r * 10.0 + (ParseDigit(s[i]) as real);
    } else {
      periodIndex := i;
    }
    i := i + 1;
  }
  i := 0;
  while i < |s| - periodIndex - 1 {
    r := r / 10.0;
    i := i + 1;
  }
  if neg {
    r := r * (-1.0);
  }
}

/**
 * Returns the integer represented by x.
 * For example, ParseDigit('3') == 3.
 * 
 * Note: This method is not verified.
 */
function ParseDigit(x: char): int
  requires IsDigit(x)
{
  if x == '0' then 0
  else if x == '1' then 1
  else if x == '2' then 2
  else if x == '3' then 3
  else if x == '4' then 4
  else if x == '5' then 5
  else if x == '6' then 6
  else if x == '7' then 7
  else if x == '8' then 8
  else 9
}

/**
 * Returns true iff s represents a real number.
 */
predicate IsReal(s: string) {
  |s| >= 3 &&
  (IsDigit(s[0]) || (s[0] == '-' && IsDigit(s[1]))) &&
  IsDigit(s[|s|-1]) &&
  exists i :: 0 <= i < |s| && s[i] == '.' &&
    forall j :: 1 <= j < |s| && j != i ==> IsDigit(s[j])
}

/**
 * Returns true iff x represents a digit in the range 0-9.
 */
predicate IsDigit(x: char) {
  x == '0' || x == '1' || x == '2' || x == '3' || x == '4' || 
  x == '5' || x == '6' || x == '7' || x == '8' || x == '9'
}

/**
 * Splits xs at every occurrence of the delimiter x.
 * The returned substrings maintain their original order in the sequence and
 * do not contain x.
 * The size of the returned sequence is equal to the number of occurrences of
 * x in xs, plus one.
 * Example output:
 * Split("", ',') == [""]
 * Split(",", ',') == ["", ""]
 * Split("abc", ',') == ["abc"]
 * Split("abc,", ',') == ["abc", ""]
 * Split(",abc", ',') == ["", "abc"]
 * Split("abc,def", ',') == ["abc", "def"]
 */
method Split(xs: string, x: char) returns (r: seq<string>)
  ensures |r| == |Indices(xs, x)| + 1
  // This behaviour is chosen for consistency but can be easily changed.
  ensures Indices(xs, x) == [] ==> r == [xs]
  ensures Indices(xs, x) != [] ==>
    // First segment: From index 0 to the first occurrence of x.
    r[0] == xs[..Indices(xs, x)[0]] &&
    // Last segment: From the last occurrence of x to index |xs|.
    r[|r|-1] == xs[Indices(xs, x)[|Indices(xs, x)|-1]..][1..] &&
    // Middle segments: Between every occurrence of x.
    forall i :: 1 <= i < |Indices(xs, x)| ==>
      r[i] == xs[Indices(xs, x)[i-1]+1..Indices(xs, x)[i]]
{
  var splits := Indices(xs, x);
  if splits == [] {
    return [xs];
  }
  r := [xs[..splits[0]]];
  var i := 1;
  while i < |splits|
    invariant 1 <= i <= |splits|
    invariant |r| == i
    invariant r[0] == xs[..splits[0]]
    invariant forall j: int :: 1 <= j < i ==> 
      r[j] == xs[splits[j-1]+1..splits[j]]
  {
    r := r + [xs[splits[i-1]+1..splits[i]]];
    i := i + 1;
  }
  r := r + [xs[splits[|splits|-1]..][1..]];
}

/**
 * Returns a sequence containing all the indices of x in xs.
 * The returned sequence is in ascending order.
 */
function Indices(xs: string, x: char): (r: seq<int>)
  // Every index in r represents an x.
  ensures forall i: int :: 0 <= i < |r| ==> 0 <= r[i] < |xs| && xs[r[i]] == x
  // There is no x in xs whose index is not in r.
  ensures forall i: int :: 0 <= i < |xs| && xs[i] == x ==> i in r
  // All indices are unique.
  ensures forall i, j: int :: 0 <= i < |r| && 0 <= j < |r| && i != j ==>
    r[i] != r[j]
  // Indices are in ascending order.
  ensures forall i, j: int :: 0 <= i < j < |r| ==> r[i] < r[j]
{
  if |xs| == 0 then []
  else if xs[|xs|-1] == x then Indices(xs[..|xs|-1], x) + [|xs|-1]
  else Indices(xs[..|xs|-1], x)
}

}
