module StringUtils {

  /**
   * Returns the real (non-negative) number represented by xs.
   * For example, ParseReal('4.01') == 4.01.
   * 
   * Note: This method is not verified.
   */
  method ParseReal(xs: string) returns (r: real)
    requires IsReal(xs)
  {
    r := ParseDigit(xs[0]) as real;
    var i := 1;
    var periodIndex: int := 1;
    while i < |xs| {
      if IsDigit(xs[i]) {
        r := r * 10.0 + (ParseDigit(xs[i]) as real);
      } else {
        periodIndex := i;
      }
      i := i + 1;
    }
    i := 0;
    while i < |xs| - periodIndex - 1 {
      r := r / 10.0;
      i := i + 1;
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
   * Returns true iff xs represents a (non-negative) real number.
   * Todo: Add support for negative numbers (?).
   */
  predicate IsReal(xs: string) {
    xs != "" &&
    IsDigit(xs[0]) &&
    IsDigit(xs[|xs|-1]) &&
    '.' in xs &&
    !(exists i :: 0 <= i < |xs| && !IsDigit(xs[i]) && xs[i] != '.') &&
    !(exists i, j :: 0 <= i < |xs| && 0 <= j < |xs| && i != j && xs[i] == '.' && xs[j] == '.')
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
   * The size of the returned sequence is equal to the number of occurrences of x
   * in xs, plus one.
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
    ensures forall i, j: int :: 0 <= i < |r| && 0 <= j < |r| && i != j ==> r[i] != r[j]
    // Indices are in ascending order.
    ensures forall i, j: int :: 0 <= i < j < |r| ==> r[i] < r[j]
  {
    if |xs| == 0 then []
    else if xs[|xs|-1] == x then Indices(xs[..|xs|-1], x) + [|xs|-1]
    else Indices(xs[..|xs|-1], x)
  }
}
