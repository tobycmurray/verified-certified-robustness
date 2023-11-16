include "IO/FileIO.dfy"
include "string_utils.dfy"

module MainModule {
  import FileIO
  import StringUtils

  method Main(args: seq<string>)
    decreases *
  {
    while true
      decreases *
    {
      /* ===================== Parse input from stdin ====================== */

      // Read from stdin.
      print "> ";
      var inputStr: string := ReadFromFile("/dev/stdin");
      print '\n';
      // Extract output vector and error margin, which are separated by spaces.
      var inputSeq: seq<string> := StringUtils.Split(inputStr, ' ');
      if |inputSeq| != 2 {
        print "Error: Expected 1 space in input. Got ", |inputSeq| - 1, ".\n";
        continue;
      }
      // Parse output vector into reals.
      if inputSeq[0] == "" {
        print "Error: The given output vector was found to be empty.\n";
        continue;
      }
      var realsStr: seq<string> := StringUtils.Split(inputSeq[0], ',');
      var areReals: bool := AreReals(realsStr);
      if !areReals {
        print "Error: The given output vector contained non-real values.\n";
        continue;
      }
      var outputVector := ParseReals(realsStr);
      // Parse error margin.
      if inputSeq[1] == "" {
        print "Error: The given error margin was found to be empty.\n";
        continue;
      }
      var isReal: bool := StringUtils.IsReal(inputSeq[1]);
      if !isReal {
        print "Error: The given error margin is not of type real.\n";
        continue;
      }
      var errorMargin := StringUtils.ParseReal(inputSeq[1]);
      print "Received output vector:\n", outputVector, '\n';
      print "Received error margin:\n", errorMargin, '\n';

      /* ======================= Certify Robustness ======================== */

      // todo
    }

  }

  method AreReals(realsStr: seq<string>) returns (b: bool)
    ensures b ==> forall i | 0 <= i < |realsStr| :: StringUtils.IsReal(realsStr[i])
  {
    for i := 0 to |realsStr|
      invariant forall j | 0 <= j < i :: StringUtils.IsReal(realsStr[j])
    {
      var isReal := StringUtils.IsReal(realsStr[i]);
      if !isReal {
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

  method ReadFromFile(filename: string) returns (str: string) {
    var readResult := FileIO.ReadBytesFromFile(filename);
    expect readResult.Success?, "Unexpected failure reading from " +
      filename + ": " + readResult.error;
    str := seq(|readResult.value|, 
      i requires 0 <= i < |readResult.value| => readResult.value[i] as char);
  }
}
