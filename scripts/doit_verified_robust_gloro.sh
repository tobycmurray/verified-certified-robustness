# train a gloro net

if ! ([ $# -eq 7 ] || [ $# -eq 8 ]); then
    echo "Usage $0 gloro_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]"
    exit 1
fi

EPSILON=$1
INTERNAL_LAYER_SIZES=$2
EVAL_EPSILON=$3
CERTIFIER=$4
GRAM_ITERATIONS=$5
EPOCHS=$6
BATCH_SIZE=$7

INPUT_SIZE=28
if [ $# -eq 8 ]; then
    INPUT_SIZE=$8
fi

if ! [[ $GRAM_ITERATIONS =~ ^[0-9]+$ ]] || [ "$GRAM_ITERATIONS" -le 0 ]; then
    echo "GRAM_ITERATIONS should be positive"
    exit 1
fi

if ! [[ $EPOCHS =~ ^[0-9]+$ ]] || [ "$EPOCHS" -le 0 ]; then
    echo "EPOCHS should be positive"
    exit 1
fi

if ! [[ $BATCH_SIZE =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -le 0 ]; then
    echo "BATCH_SIZE should be positive"
    exit 1
fi

if ! [[ $INPUT_SIZE =~ ^[0-9]+$ ]] || [ "$INPUT_SIZE" -le 0 ]; then
    echo "MODEL_INPUT_SIZE should be positive"
    exit 1
fi

if [ ! -x ${CERTIFIER} ]; then
    echo "${CERTIFIER} doesn't exist or is not executable"
    exit 1
fi

PYTHON=python3

# check for dependencies
if ! which ${PYTHON} > /dev/null 2>&1; then
    echo "Python binary ${PYTHON} not found. Consider editing the PYTHON variable in this script."
    exit 1
fi

if ! which sed > /dev/null 2>&1; then
    echo "sed not found."
    exit 1
fi

if ! which tee > /dev/null 2>&1; then
    echo "tee not found."
    exit 1
fi

if ! which ts > /dev/null 2>&1; then
    echo "ts not found."
    exit 1
fi

if ! which jq > /dev/null 2>&1; then
    echo "jq not found."
    exit 1
fi

# check we won't interfere with any concurrent execution of this same script
if [ -d model_weights_csv ]; then
    echo "Directory model_weights_csv/ still exists."
    exit 1
fi

# clean out any old temporary model weights etc.
rm -rf model_weights_csv

DT=$(date +"%Y-%m-%d_%H:%M:%S")

if [ -d "${DT}" ]; then
    echo "Directory ${DT}/ already exists!"
    exit 1
fi

mkdir "${DT}"

if [ ! -d "${DT}" ]; then
    echo "Error creating directory ${DT}"
    exit 1
fi

echo ""
echo ""
echo "Artifacts and results will live in: ${DT}/"
echo ""

PARAMS_FILE=${DT}/params.txt
echo "    (Global) Model input size: ${INPUT_SIZE}x${INPUT_SIZE}" > "${PARAMS_FILE}"
echo "    (Training) Gloro epsilon: ${EPSILON}" >> "${PARAMS_FILE}"
echo "    (Training) INTERNAL_LAYER_SIZES: ${INTERNAL_LAYER_SIZES}" >> "${PARAMS_FILE}"
echo "    (Training) Epochs: ${EPOCHS}" >> "${PARAMS_FILE}"
echo "    (Training) Batch size: ${BATCH_SIZE}" >> "${PARAMS_FILE}"
echo "    (Certifier) Eval epsilon: ${EVAL_EPSILON}" >> "${PARAMS_FILE}"
echo "    (Certifier) GRAM_ITERATIONS: ${GRAM_ITERATIONS}" >> "${PARAMS_FILE}"


echo "Running with these parameters (saved in ${PARAMS_FILE}):"
cat "$PARAMS_FILE"
echo ""
echo ""

MODEL_WEIGHTS_DIR="${DT}/model_weights_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}"
MODEL_OUTPUTS="${DT}/all_mnist_outputs_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}.txt"
NEURAL_NET_FILE="${DT}/neural_net_mnist_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}.txt"
MODEL_OUTPUTS_EVAL="${DT}/all_mnist_outputs_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}_eval_${EVAL_EPSILON}.txt"
RESULTS_JSON="${DT}/results_epsilon_${EPSILON}_${INTERNAL_LAYER_SIZES}_${EPOCHS}_eval_${EVAL_EPSILON}_gram_${GRAM_ITERATIONS}.json"



# train the gloro model
${PYTHON} train_gloro.py $EPSILON "$INTERNAL_LAYER_SIZES" $EPOCHS $INPUT_SIZE

if [ ! -d model_weights_csv ]; then
    echo "Training gloro model failed or results not successfully saved to model_weights_csv/ dir"
    exit 1
fi

# save the weights
mv model_weights_csv "$MODEL_WEIGHTS_DIR"
# make the outputs from the zero-bias model
${PYTHON} zero_bias_saved_model.py "$INTERNAL_LAYER_SIZES" "$MODEL_WEIGHTS_DIR" "$MODEL_OUTPUTS" $INPUT_SIZE
# make the neural net in a form the certifier can understand
${PYTHON} make_certifier_format.py "$INTERNAL_LAYER_SIZES" "$MODEL_WEIGHTS_DIR" > "$NEURAL_NET_FILE"
# add the epsilon to each model output for the certifier to certify against
sed "s/$/ ${EVAL_EPSILON}/" "$MODEL_OUTPUTS" > "$MODEL_OUTPUTS_EVAL"


echo "Running the certifier. This may take a while..."
cat "$MODEL_OUTPUTS_EVAL" | ${CERTIFIER} "$NEURAL_NET_FILE" "$GRAM_ITERATIONS" | tee "$RESULTS_JSON" | ts "%Y-%m-%d %H:%M:%S" | tee "${RESULTS_JSON}.timestamps"

if ! jq empty "$RESULTS_JSON"; then
    echo "Certifier produced invalid JSON!"
    exit 1
fi


${PYTHON} test_verified_certified_robust_accuracy.py "$INTERNAL_LAYER_SIZES" "$RESULTS_JSON" "$MODEL_WEIGHTS_DIR" $INPUT_SIZE

echo ""
echo "Unverified model statistics (to compare to the above verified ones):"
cat "${MODEL_WEIGHTS_DIR}/gloro_model_stats.json"

# get timestamps
CERTIFIER_FINISH_TIME=$(grep '\]$' "${RESULTS_JSON}.timestamps" | cut -d' ' -f-2)
CERTIFIER_START_TIME=$(grep '\[$' "${RESULTS_JSON}.timestamps" | cut -d' ' -f-2)
LIPSCHITZ_BOUNDS_TIME=$(grep 'lipschitz\_bounds' "${RESULTS_JSON}.timestamps" | cut -d' ' -f-2)

echo ""
echo "Certifier started at:         ${CERTIFIER_START_TIME}"
echo "Certifier produced bounds at: ${LIPSCHITZ_BOUNDS_TIME}"
echo "(The difference between these quantities is therefore the time taken to compute those bounds.)"
echo "Certifier finished at: ${CERTIFIER_FINISH_TIME}"


echo ""
echo "All done."
echo "Artifacts and results all saved in: ${DT}/"
echo "Parameters saved in: ${PARAMS_FILE}, whose contents follows:"
cat "$PARAMS_FILE"
echo ""
echo "Model weights and (unverified) gloro lipschitz constants saved in: ${MODEL_WEIGHTS_DIR}"
echo "(Unverified) gloro model statistics saved in: ${MODEL_WEIGHTS_DIR}/gloro_model_stats.json"
echo "Model outputs saved in: ${MODEL_OUTPUTS}"
echo "Neural network (for certifier) saved in: ${NEURAL_NET_FILE}"
echo "Model outputs for evaluation saved in: ${MODEL_OUTPUTS_EVAL}"
echo "Certified robustness results saved in: ${RESULTS_JSON}"
echo "Timestamped certifier output saved in: ${RESULTS_JSON}.timestamps"

