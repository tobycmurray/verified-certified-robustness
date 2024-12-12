include "IO/FileIO.dfy"
include "string_utils.dfy"
include "lipschitz.dfy"
include "basic_arithmetic.dfy"

module MainModule {
  import FileIO
  import StringUtils
  import opened Lipschitz
  import BasicArithmetic

  method Main(args: seq<string>)
    decreases *
  {
    /* ===================== Generate Lipschitz bounds ===================== */
    // Parse neural network from file (unverified).
    print "Parsing...\n";
    var neuralNetStr: string := ReadFromFile("Input/neural_network_3.txt");
    var maybeNeuralNet: Maybe<NeuralNetwork> := ParseNeuralNet(neuralNetStr);
    expect maybeNeuralNet.Some?, "Failed to parse neural network.";
    var neuralNet: NeuralNetwork := maybeNeuralNet.val;
    // Generate spectral norms for the matrices comprising the neural net.
    // We currently assume an external implementation for generating these.
    print "Generating spectral norms...\n";
    var specNorms: seq<real> := GenerateSpecNorms(neuralNet);
    // Generate the Lipschitz bounds for each logit in the output vector.
    print "Generating Lipschitz bounds...\n";
    var lipBounds: seq<real> := GenLipBounds(neuralNet, specNorms);
    print "Bounds generated:\n";
    for i: nat := 0 to |lipBounds| {
      // BasicArithmetic.PrintReal(lipBounds[i], 20);
      print lipBounds[i];
      print "\n\n";
    }

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

  method ReadFromFile(filename: string) returns (str: string) {
    var readResult := FileIO.ReadBytesFromFile(filename);
    expect readResult.Success?, "Unexpected failure reading from " +
      filename + ": " + readResult.error;
    str := seq(|readResult.value|, 
      i requires 0 <= i < |readResult.value| => readResult.value[i] as char);
  }
}
