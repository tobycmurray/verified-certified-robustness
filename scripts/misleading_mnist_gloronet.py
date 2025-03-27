import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Layer

from art.attacks.evasion import ProjectedGradientDescent, AutoAttack, FastGradientMethod, MomentumIterativeMethod
from art.estimators.classification import KerasClassifier

import numpy as np

import json
import os

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images (28x28 -> 784)
#x_train = x_train.reshape(-1, 784)
#x_test = x_test.reshape(-1, 784)

# Define exponential decay learning rate
initial_learning_rate = 0.001
final_learning_rate = 0.00001
epochs = 20
decay_steps = (len(x_train) // 128) * epochs  # Total number of steps

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=final_learning_rate / initial_learning_rate,
    staircase=False  # Smooth decay rather than step-wise
)

inputs=Input((28, 28, 1))
z=Flatten()(inputs)
# FIXME: remove duplication here by buliding the network from INPUT_LAYER_SIZES
INPUT_LAYER_SIZES=[512,374,256,128,64,32]
z = Dense(512, use_bias=False, activation='relu')(z)
z = Dense(374, use_bias=False, activation='relu')(z)
z = Dense(256, use_bias=False, activation='relu')(z)
z = Dense(128, use_bias=False, activation='relu')(z)
z = Dense(64, use_bias=False, activation='relu')(z)
z = Dense(32, use_bias=False, activation='relu')(z)
outputs = Dense(10, use_bias=False)(z)

model = Model(inputs, outputs)

# Compile the model with the learning rate schedule
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Test accuracy: {test_acc:.4f}")

scaling_factor=2.0
second_final_layer = model.layers[-2]
second_final_weights = np.array(second_final_layer.get_weights()[0])
second_final_weights_saved=second_final_weights.copy()
final_layer = model.layers[-1]
final_weights = np.array(final_layer.get_weights()[0])
final_weights_saved=final_weights.copy()
i=0

from gloro.models import GloroNet
from gloro.training.metrics import rejection_rate, vra, clean_acc

epsilon=1.58
g_orig = GloroNet(model=model, epsilon=epsilon)

g_orig.freeze_lc()

lipschitz_constants_orig = g_orig.lipschitz_constant()
#print(f"Lipschitz constants: {lipschitz_constants_orig}")

lipschitz_constants = g_orig.lipschitz_constant()
mean_orig=np.mean(lipschitz_constants_orig[lipschitz_constants_orig != -1])
mean=np.mean(lipschitz_constants[lipschitz_constants != -1])
print(f"Lipschitz mean: {mean}")

# we stop when the mean lipschitz bound doesn't exceed mean_lower
# exit early if the mean gets too big for some reason
mean_lower=0.5
mean_upper=1.2*mean_orig

from tensorflow.keras.utils import to_categorical

y_test_one_hot = to_categorical(y_test, num_classes=10)

gloro_outputs=g_orig.predict(x_test)
rejection_rate_orig=rejection_rate(y_test_one_hot,gloro_outputs)
vra_orig=vra(y_test_one_hot,gloro_outputs)
clean_acc_orig=clean_acc(y_test_one_hot,gloro_outputs)

# continually scale weights in final layer and measure accuracy
while mean > mean_lower and mean <= mean_upper:    
    final_weights_saved=final_weights.copy()
    final_weights *= scaling_factor
    second_final_weights_saved=second_final_weights.copy()
    second_final_weights /= scaling_factor
    i=i+1
    second_final_layer.set_weights([second_final_weights])
    final_layer.set_weights([final_weights])    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("")
    print(f"After doing {i+1} scalings: ")
    print(f"Test Loss (zero bias model): {loss:.4f}")
    print(f"Test Accuracy (zero bias model): {accuracy:.4f}")
    g = GloroNet(model=model, epsilon=epsilon)
    g.freeze_lc()
    lipschitz_constants = g.lipschitz_constant()
    mean=np.mean(lipschitz_constants[lipschitz_constants != -1])    
    #print(f"Lipschitz constants: {lipschitz_constants}")
    print(f"Lipschitz mean: {mean}")
    #gloro_outputs=g.predict(x_test)
    #rr=rejection_rate(gloro_outputs,gloro_outputs)
    #print(f"Rejection rate: {rr}")
    
#second_final_layer.set_weights([second_final_weights_saved])
#final_layer.set_weights([final_weights_saved])
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)



g = GloroNet(model=model, epsilon=epsilon)
g.freeze_lc()
lipschitz_constants = g.lipschitz_constant()
print(f"Lipschitz constants: {lipschitz_constants}")
mean=np.mean(lipschitz_constants[lipschitz_constants != -1])    
#print(f"Lipschitz constants: {lipschitz_constants}")
print(f"Lipschitz mean: {mean}")

gloro_outputs=g.predict(x_test)
rejection_rate_new=rejection_rate(y_test_one_hot,gloro_outputs)
vra_new=vra(y_test_one_hot,gloro_outputs)
clean_acc_new=clean_acc(y_test_one_hot,gloro_outputs)
robustness_new=1.0-rejection_rate_new

# Create a directory to save the files if it does not exist
save_dir = 'model_weights_csv_misleading_mnist_gloro_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def to_serializable(value):
    # If `value` is a tf.Tensor, convert it to a Python list
    if isinstance(value, tf.Tensor):
        return value.numpy().tolist()
    # If it's already a NumPy array, just call .tolist()
    elif hasattr(value, "tolist"):
        return value.tolist()
    else:
        # It's something already JSON-serializable
        return value

# save the statistics calculated by gloro
data = {
    "comment": "these statistics are unverified and calculated by the gloro implementation",
    "eval_epsilon": epsilon,
    "accuracy": to_serializable(clean_acc_new),
    "rejection_rate": to_serializable(rejection_rate_new),
    "robustness": to_serializable(robustness_new),
    "vra": to_serializable(vra_new)
}
file_path = os.path.join(save_dir, "gloro_model_results.json")


    
y_pred=gloro_outputs
answers=[data]
bot_index = tf.cast(tf.shape(y_pred)[1] - 1, 'int64')
bots=0
preds = tf.argmax(y_pred, axis=1)
i=0
for p in preds:
    answer = {}
    if p == bot_index:
        bots=bots+1
        answer["certified"]=False
    else:
        answer["certified"]=True
    # also save information about the input/output point itself
    answer["input_point"]=to_serializable(x_test[i])
    answer["output_point"]=to_serializable(y_pred[i])
    answer["output_class"]=to_serializable(p)
    answer["ground_truth_output"]=to_serializable(y_test[i])

    answers.append(answer)
    i=i+1
    
with open(file_path, "w") as json_file:
    json.dump(answers, json_file, indent=4)


print(f"Original rejection rate: {rejection_rate_orig}. Final rejection rate: {rejection_rate_new}")
print(f"Original vra: {vra_orig}. Final vra: {vra_new}")
print(f"Original clean acc: {clean_acc_orig}. Final clean acc: {clean_acc_new}")
print(f"Final robustness according to gloro: {robustness_new}")


# save the gloro model using their custom format
g.save("misleading_mnist_gloro_model.gloronet")

layer_weights = [layer.get_weights()[0] for layer in g.layers if len(layer.get_weights()) > 0]

for i, weights in enumerate(layer_weights):
    np.savez(f"layer_{i}_weights.npz", weights=weights)


# Loop through each layer, extract weights and biases, and save them
for i, weights in enumerate(layer_weights):
    np.savetxt(os.path.join(save_dir, f'layer_{i}_weights.csv'), weights, delimiter=',', fmt='%.150f')

# Loop through each layer, extract weights and biases, and save them
for i, c in enumerate(lipschitz_constants):
    np.savetxt(os.path.join(save_dir, f'logit_{i}_gloro_lipschitz.csv'), [c], delimiter=',', fmt='%.150f')
        
print('model weights saved in: ', save_dir)

print("model epsilon (e.g. to run a PGD attack etc.): ", epsilon)

print("INPUT_LAYER_SIZES (e.g. to run a PGD attack etc.): ", INPUT_LAYER_SIZES)
