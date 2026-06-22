import os
import utils
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.framework.ops import EagerTensor


def my_load_mnist_data(keras_loader):
    (x_train, y_train), (x_test, y_test) = keras_loader()
    x_test = x_test[..., tf.newaxis]  # Add channel dimension for grayscale images
    return (x_train, y_train), (x_test, y_test)

datasets={
    "mnist" : {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 10,
        "keras_load_data_method": lambda: my_load_mnist_data(tf.keras.datasets.mnist.load_data)
    },
    "fashion_mnist" : {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 10,        
        "keras_load_data_method": lambda: my_load_mnist_data(tf.keras.datasets.fashion_mnist.load_data)
    },
    "cifar10" : {
        "channels": 3,
        "default_input_size": 32,
        "num_classes": 10,
        "keras_load_data_method": tf.keras.datasets.cifar10.load_data
    },
    # HIGGS is a tabular (non-image) binary classification benchmark: 28 continuous
    # features, 11M examples. Loaded via tfds (see load_higgs_data) rather than a
    # keras loader, and consumed as flat feature vectors rather than HxWxC images.
    "higgs" : {
        "tabular": True,
        "num_features": 28,
        "num_classes": 2,
        "default_input_size": None,
    },
    # EMNIST: handwritten characters in the same 28x28x1 format as MNIST, but with
    # many more classes and far more data. The dict key is the tfds config name so
    # the existing get_data() tfds path (load_gloro_data) works unchanged; the test
    # split is pulled from tfds in load_test_data (it is not in tf.keras.datasets).
    "emnist/balanced" : {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 47,
    },
    "emnist/byclass" : {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 62,
    }
}


# Canonical HIGGS feature order (Baldi et al. 2014, "Searching for Exotic Particles
# in High-energy Physics with Deep Learning"): 21 low-level kinematic features
# followed by 7 high-level derived features. tfds stores each feature under its own
# dict key; we stack them in this fixed order for reproducibility.
HIGGS_FEATURE_ORDER = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
    "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
    "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag",
    "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]

_HIGGS_CACHE = {}


def load_higgs_data():
    """Load a deterministic, disjoint subset of HIGGS as standardized float32 feature
    vectors. Returns (x_train, y_train, x_test, y_test) with integer labels in {0,1}.

    Uses the CANONICAL HIGGS split (Baldi et al. 2014 / UCI / tfds): train = the
    FIRST n_train rows, test = the LAST n_test rows of the full 11M stream. The
    defaults give the standard split exactly (train = first 10.5M, test = last
    500k). The full stream is read so the test block is always the canonical tail
    (and identical across runs regardless of n_train).

    Per-feature mean/std are computed from the TRAIN block only (no test leakage), so
    L2 perturbations are measured in standardized-sigma units; the stats are saved to
    higgs_standardization.npz so the certifier can apply the identical transform."""
    if "data" in _HIGGS_CACHE:
        return _HIGGS_CACHE["data"]

    HIGGS_TOTAL = 11_000_000
    n_test = int(os.environ.get("HIGGS_N_TEST", "500000"))
    n_train = int(os.environ.get("HIGGS_N_TRAIN", str(HIGGS_TOTAL - 500000)))
    if n_test + n_train > HIGGS_TOTAL:
        raise ValueError(
            f"HIGGS_N_TEST + HIGGS_N_TRAIN ({n_test}+{n_train}) exceeds "
            f"the {HIGGS_TOTAL} available rows")
    tfds_dir = os.environ.get("TFDS_DIR", None)

    print(f"Loading HIGGS via tfds (canonical split: train=first {n_train}, "
          f"test=last {n_test} of {HIGGS_TOTAL}); reading full stream "
          f"(first run downloads/prepares ~2.6GB)...")
    ds = tfds.load("higgs", split="train", data_dir=tfds_dir, shuffle_files=False)

    def to_xy(ex):
        label = tf.cast(ex["class_label"], tf.int32)
        feats = tf.stack(
            [tf.cast(ex[k], tf.float32) for k in HIGGS_FEATURE_ORDER], axis=-1)
        return feats, label

    ds = ds.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE).batch(100000)
    xs, ys = [], []
    for fb, lb in tfds.as_numpy(ds):
        xs.append(fb)
        ys.append(lb)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)

    # Canonical Baldi split: train = first n_train rows, test = last n_test rows.
    x_train, y_train = X[:n_train], Y[:n_train]
    x_test, y_test = X[HIGGS_TOTAL - n_test:], Y[HIGGS_TOTAL - n_test:]

    # Accumulate the standardization moments in float64: at 10.5M rows a float32
    # accumulator loses enough precision that the standardized data is not actually
    # unit-variance.
    mean = x_train.mean(axis=0, dtype=np.float64)
    std = x_train.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "higgs_standardization.npz")
    np.savez(stats_path, mean=mean, std=std,
             feature_order=np.array(HIGGS_FEATURE_ORDER))
    print(f"HIGGS loaded: train {x_train.shape}, test {x_test.shape}; "
          f"train positive-class fraction={y_train.mean():.4f}; "
          f"standardization saved to {stats_path}")

    _HIGGS_CACHE["data"] = (x_train, y_train, x_test, y_test)
    return _HIGGS_CACHE["data"]


def build_model(Input, Flatten, Dense, input_size=28, dataset='mnist', internal_layer_sizes=[]):
    """set input_size to something smaller if the model is downsampled"""
    if dataset not in datasets.keys():
        raise ValueError(f"Unsupported dataset. Choose one of: {datasets.keys()}")

    if datasets[dataset].get("tabular"):
        # Tabular input: a flat feature vector. We still apply Flatten (a no-op on a
        # rank-1 input) so the layer indexing matches the image models, keeping
        # load_and_set_weights and the downstream eval scripts unchanged.
        inputs = Input((datasets[dataset]["num_features"],))
    else:
        channels = datasets[dataset]["channels"]
        inputs = Input((input_size, input_size, channels))
    z = Flatten()(inputs)
    for size in internal_layer_sizes:
        z = Dense(size, use_bias=False, activation='relu')(z)
    outputs = Dense(datasets[dataset]["num_classes"], use_bias=False)(z)
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
    if dataset == "higgs":
        # Tabular: build (feature_vector, integer_label) datasets directly; no image
        # augmentation/resize. Labels stay integer for sparse_crossentropy training.
        x_train, y_train, x_test, y_test = load_higgs_data()
        train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(min(len(x_train), 100000))
                 .batch(batch_size))
        test = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .batch(batch_size))
        return (train, test)

    train, test, metadata = utils.get_data(dataset, batch_size, augmentation)

    if dataset not in datasets.keys():
        raise ValueError(f"Unsupported dataset. Choose one of: {dataset.keys()}")

    default_input_size = datasets[dataset]["default_input_size"]
        
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

    if dataset not in datasets.keys():
        raise ValueError(f"Unsupported dataset. Choose one of: {dataset.keys()}")

    if dataset == "higgs":
        # Tabular test set: standardized feature vectors with one-hot labels (to
        # match the gloro metric helpers used at the end of train_gloro.py).
        _, _, x_test, y_test = load_higgs_data()
        y_test = tf.keras.utils.to_categorical(
            y_test, num_classes=datasets[dataset]["num_classes"])
        return (x_test, y_test)

    if dataset.startswith("emnist"):
        # EMNIST has no tf.keras.datasets loader; pull the test split from tfds.
        # Orientation matches the (tfds-native) train split, so no transpose is
        # needed for train/test consistency.
        tfds_dir = os.environ.get("TFDS_DIR", None)
        x_test, y_test = tfds.as_numpy(tfds.load(
            dataset, split="test", data_dir=tfds_dir,
            as_supervised=True, batch_size=-1))
        x_test = x_test.astype("float32") / 255.0
        y_test = tf.keras.utils.to_categorical(
            y_test, num_classes=datasets[dataset]["num_classes"])
        return (x_test, y_test)

    load_data = datasets[dataset]["keras_load_data_method"]
    (x_train, y_train), (x_test, y_test) = load_data()

    num_classes = datasets[dataset]["num_classes"]
    default_size = datasets[dataset]["default_input_size"]

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

