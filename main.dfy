include "IO/FileIO.dfy"
include "parsing.dfy"
include "linear_algebra.dfy"
include "neural_networks.dfy"
include "operator_norms.dfy"
include "robustness_certification.dfy"

module MainModule {
  import FileIO
  import opened Parsing
  import opened LinearAlgebra
  import opened NeuralNetworks
  import opened OperatorNorms
  import opened RobustnessCertification

  method Main(args: seq<string>)
    decreases *
  {
    /* =================== Parse command line arguments ==================== */
    // parse command line arguments
    if |args| != 3 {
      print "Usage: main <neural_network_input.txt> <GRAM_ITERATIONS>\n";
      return;
    }
    var input_file: string := args[1];

    if !IsInt(args[2]) {
      print "<GRAM_ITERATIONS> should be a positive integer";
      return;
    }
    var GRAM_ITERATIONS: int := ParseInt(args[2]);
    if GRAM_ITERATIONS <= 0 {
      print "<GRAM_ITERATIONS> should be positive";
      return;
    }
    
    print "[\n";
    
    /* ===================== Generate Lipschitz bounds ===================== */
    // Parse neural network from file (unverified).
    var neuralNetStr: string := ReadFromFile(input_file);
    var maybeNeuralNet: Maybe<NeuralNetwork> := ParseNeuralNet(neuralNetStr);
    expect maybeNeuralNet.Some?, "Failed to parse neural network.";
    var neuralNet: NeuralNetwork := maybeNeuralNet.val;
    // Generate spectral norms for the matrices comprising the neural net.
    var specNorms: seq<real> := GenerateSpecNorms(neuralNet, GRAM_ITERATIONS);
    // Generate the margin Lipschitz bounds for each pair of logits in the output vector.
    var lipBounds: Matrix := GenMarginBounds(neuralNet, specNorms);

    print "{\n";
    print "  \"lipschitz_bounds\": ", lipBounds, ",\n";
    print "  \"GRAM_ITERATIONS\": ", GRAM_ITERATIONS, "\n";
    print "}\n"; 
    

    /* ================= Read input vectors from stdin ===================== */
    var inputStr: string := ReadFromFile("/dev/stdin");

    var lines: seq<string> := Split(inputStr, '\n');    
    if |lines| <= 0 {
      return;
    }

    var l := 0;
    while l < |lines|
      decreases |lines| - l
    {
      var line := lines[l];
      l := l + 1;
      var inputSeq: seq<string> := Split(line, ' ');    

      if |inputSeq| != 2 || inputSeq[0] == "" {
        // as soon as we see bad input, stop silently so that the end of the input won't cause junk to be printed
        print "]\n";
        return;
      }
      
      var realsStr: seq<string> := Split(inputSeq[0], ',');
      var areReals: bool := AreReals(realsStr);
      if !areReals {
        print "Error: The given output vector contained non-real values.\n";
        return;
      }
      var outputVector := ParseReals(realsStr);
      
      // Parse error margin.
      if inputSeq[1] == "" {
        print "Error: The given error margin was found to be empty.\n";
        return;
      }
      var isReal: bool := IsReal(inputSeq[1]);
      if !isReal {
        print "Error: The given error margin is not of type real.\n";
        return;
      }
      var errorMargin := ParseReal(inputSeq[1]);

      /* ======================= Certify Robustness ======================== */

      // The given output vector must be compatible with the neural network.
      if |outputVector| != |lipBounds| {
        print "Error: Expected a vector of size ", |lipBounds|,
          ", but got ", |outputVector|, ".\n";
        return;
      }
      // Use the generated Lipschitz bounds to certify robustness.
      var robust: bool := CertifyMargin(outputVector, errorMargin, lipBounds);
      /* Verification guarantees that 'true' is only printed when for all input
      vectors v where applying the neural network to v results in the given
      output vector, this input-output pair of vectors is robust with respect
      to the given error margin. */
      assert robust ==> forall v: Vector |
        IsInput(v, neuralNet) && NN(neuralNet, v) == outputVector ::
        Robust(v, outputVector, errorMargin, neuralNet);

      print ",\n";
      print "{\n";
      print "\"output\": ";
      print outputVector, ",\n";
      print "\"radius\": ";
      print errorMargin, ",\n";
      print "\"certified\": ";
      print robust, "\n";
      print "}\n";
    }
    print "]\n";
  }

  method ReadFromFile(filename: string) returns (str: string) {
    var readResult := FileIO.ReadBytesFromFile(filename);
    expect readResult.Success?, "Unexpected failure reading from " +
      filename + ": " + readResult.error;
    str := seq(|readResult.value|, 
      i requires 0 <= i < |readResult.value| => readResult.value[i] as char);
  }
}
