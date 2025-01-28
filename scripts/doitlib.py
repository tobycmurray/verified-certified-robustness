import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

def build_model(Input, Flatten, Dense, input_size=28, dataset='mnist', internal_layer_sizes=[]):
    """set input_size to something smaller if the model is downsampled"""
    if dataset=="mnist":
        channels=1
    elif dataset=="cifar10":
        channels=3
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")
    
    inputs = Input((input_size, input_size, channels))
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


    
def load_gloro_data(batch_size=256, augmentation='none', input_size=28, dataset='mnist'):
    """set input_size to resize the dataset. Returns a pair (train, test)"""
    train, test, metadata = utils.get_data(dataset, batch_size, augmentation)

    """set input_size to something smaller if the model is downsampled"""
    if dataset=="mnist":
        default_input_size=28
    elif dataset=="cifar10":
        default_input_size=32
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")
    
    if input_size != default_input_size:
        def resize(image, label):
            image = tf.image.resize(image, [input_size, input_size])  
            return image, label
        train = train.map(resize)
        test = test.map(resize)
        
    return (train, test)

def load_test_data(dataset='mnist', input_size=None):
    """Load and preprocess test data for the specified dataset ('mnist' or 'cifar10').                                                              
    Set input_size to resize the test dataset. Returns a pair (x_test, y_test).                                                                     
    """
    # Turn off SSL certificate checking :(                                                                                                          
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import tensorflow as tf
    import numpy as np

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test[..., tf.newaxis]  # Add channel dimension for grayscale images                                                              
        num_classes = 10
        default_size = 28
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
        default_size = 32
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")

    # Normalize pixel values to [0, 1]                                                                                                             
    x_test = x_test.astype('float32') / 255.0

    # Set default input_size if not provided                                                                                                       
    if input_size is None:
        input_size = default_size

    # Resize the test dataset if input_size differs from the default size                                                                          
    if input_size != default_size:
        resized_tensor = tf.image.resize(x_test, [input_size, input_size])
        if tf.executing_eagerly():
            x_test = resized_tensor.numpy()
        else:
            # Convert the tensor to NumPy using a session                                                                                          
            with tf.compat.v1.Session() as sess:
                x_test = sess.run(resized_tensor)

    # Convert labels to one-hot encoded format                                                                                                     
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    return (x_test, y_test)

