include "IO/FileIO.dfy"
include "string_utils.dfy"
include "lipschitz.dfy"

module MainModule {
  import FileIO
  import StringUtils
  import opened Lipschitz

  method Main(args: seq<string>)
    decreases *
  {
    /* ===================== Generate Lipschitz bounds ===================== */
    print "Generating Lipschitz bounds...\n";
    // Parse neural network from file (unverified).
    var neuralNetStr: string := ReadFromFile("Input/neural_network.txt");
    var maybeNeuralNet: (bool, NeuralNetwork) := ParseNeuralNet(neuralNetStr);
    expect maybeNeuralNet.0, "Failed to parse neural network.";
    var neuralNet: NeuralNetwork := maybeNeuralNet.1;
    // Generate spectral norms for the matrices comprising the neural net.
    // We currently assume an external implementation for generating these.
    var specNorms: seq<real> := GenerateSpecNorms(neuralNet);
    // Generate the Lipschitz bounds for each logit in the output vector.
    var lipBounds: seq<real> := GenLipBounds(neuralNet, specNorms);
    print "Bounds generated: ", lipBounds, "\n\n";

    /* ================= Repeatedly certify output vectors ================= */

    while true
      // This tells Dafny that we don't intend for this loop to terminate.
      decreases *
    {
      /* ===================== Parse input from stdin ====================== */

      // Read from stdin. Currently, input must be terminated with an EOF char.
      print "> ";
      var inputStr: string := ReadFromFile("/dev/stdin");
      print '\n';
      // Extract output vector and error margin, which are space-separated.
      var inputSeq: seq<string> := StringUtils.Split(inputStr, ' ');
      if |inputSeq| != 2 {
        print "Error: Expected 1 space in input. Got ", |inputSeq| - 1, ".\n";
        continue;
      }
      
      // Parse output vector.
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

      // Print parse results.
      print '\n';
      print "Received output vector:\n", outputVector, "\n\n";
      print "Received error margin:\n", errorMargin, "\n\n";

      /* ======================= Certify Robustness ======================== */

      // The given output vector must be compatible with the neural network.
      if |outputVector| != |lipBounds| {
        print "Error: Expected a vector of size ", |lipBounds|,
          ", but got ", |outputVector|, ".\n";
        continue;
      }
      // Use the generated Lipschitz bounds to certify robustness.
      var robust: bool := Certify(outputVector, errorMargin, lipBounds);
      /* Verification guarantees that 'true' is only printed when for all input
      vectors v where applying the neural network to v results in the given
      output vector, this input-output pair of vectors is robust with respect
      to the given error margin. */
      assert robust ==> forall v: Vector |
        CompatibleInput(v, neuralNet) && NN(neuralNet, v) == outputVector ::
        Robust(v, outputVector, errorMargin, neuralNet);
      print "Certification:\n", robust, "\n\n";
    }
  }

  method ParseNeuralNet(xs: string) returns (t: (bool, NeuralNetwork))
    // Todo: Verify
  {
    var matrices: seq<Matrix> := [];
    var i := 0;
    while i < |xs| {
      // Expect matrix
      if i >= |xs| - 1 || xs[i..i+2] != "[[" {
        return (false, [[[0.0]]]);
      }
      var j := i + 2;
      while xs[j-2..j] != "]]"
        invariant j <= |xs|
        decreases |xs| - j
      {
        if j >= |xs| {
          return (false, [[[0.0]]]);
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
          return (false, [[[0.0]]]);
        }
        var l := k;
        while ys[l] != ']'
          invariant l < |ys|
          decreases |ys| - l
        {
          if l + 1 >= |ys| {
            return (false, [[[0.0]]]);
          }
          l := l + 1;
        }
        // ys[k..l] == "[r1,r2,...,rn"
        var zs := ys[k+1..l];
        // zs == "r1,r2,...,rn"
        var realsStr: seq<string> := StringUtils.Split(zs, ',');
        var areReals: bool := AreReals(realsStr);
        if !areReals {
          return (false, [[[0.0]]]);
        }
        var v: seq<real> := ParseReals(realsStr);
        if |v| == 0 {
          return (false, [[[0.0]]]);
        }
        var v': Vector := v;
        vectors := vectors + [v'];
        k := l + 2; // skip comma
      }
      var matrixWellFormed := IsMatrixWellFormed(vectors);
      if !matrixWellFormed {
        return (false, [[[0.0]]]);
      }
      var matrix: Matrix := vectors;
      matrices := matrices + [matrix];
      i := j + 1; // xs[j] == ',' or EOF
    }
    var neuralNetWellFormed := IsNeuralNetWellFormed(matrices);
    if !neuralNetWellFormed {
      return (false, [[[0.0]]]);
    }
    var neuralNet: NeuralNetwork := matrices;
    return (true, neuralNet);
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
