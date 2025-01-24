import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

def build_mnist_model(Input, Flatten, Dense, input_size=28, internal_layer_sizes=[]):
    """set input_size to something smaller if the model is downsampled"""
    inputs = Input((input_size, input_size))
    z = Flatten()(inputs)
    for size in internal_layer_sizes:
        z = Dense(size, use_bias=False, activation='relu')(z)
    outputs = Dense(10, use_bias=False)(z)
    return (inputs, outputs)

def load_and_set_weights(csv_loc, internal_layer_sizes, model):
    """model should already be built. This will compile it too"""
    dense_weights = []
    i=0
    # always one extra iteration than internal_layer_sizes length
    while i<=len(internal_layer_sizes):
        dense_weights.append(np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=","))
        model.layers[i+2].set_weights([dense_weights[i]])
        i=i+1
        
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])


    
def load_mnist_gloro_data(batch_size=256, augmentation='none', input_size=28):
    """set input_size to resize the dataset. Returns a pair (train, test)"""
    train, test, metadata = utils.get_data('mnist', batch_size, augmentation)

    
    if input_size != 28:
        def resize(image, label):
            image = tf.image.resize(image, [input_size, input_size])  
            return image, label
        train = train.map(resize)
        test = test.map(resize)
        
    return (train, test)
    
def load_mnist_test_data(input_size=28):
    """set input_size to resize the test dataset. Returns a pair (x_test, y_test)"""
    # turn off SSL cert checking :(
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_test = x_test.astype('float32') / 255.0
    if input_size != 28:
        resized_tensor = tf.image.resize(x_test[..., tf.newaxis], [input_size, input_size])
        if tf.executing_eagerly():
            x_test=resized_tensor.numpy()
        else:
            # Convert the tensor to NumPy using a session
            with tf.compat.v1.Session() as sess:
                x_test = sess.run(resized_tensor)
                x_test = np.squeeze(x_test, axis=-1)  # Remove the last dimension
    
    # Convert labels to one-hot encoded format
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_test, y_test)
