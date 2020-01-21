import numpy as np
import tensorflow as tf


def to_float(X):
    """
    Scale all images to have pixels in the range [0, 1]

    :param X: Input images. Shape (n_images, height, width, channels)
    :type X: np.ndarray
    :return: Scaled images. Same shape as X
    :rtype: np.ndarray
    """
    old_shape = X.shape
    x = X.reshape((old_shape[0], -1))
    x = x.astype(np.float32)
    x -= np.min(x, axis=1, keepdims=True)
    x = x / np.max(x, axis=1, keepdims=True)
    return x.reshape(old_shape)


def mnist():
    """
    Load the MNIST dataset

    :return: MNIST images and labels
    :rtype: tuple of np.ndaray
    """
    (X, y), _ = tf.keras.datasets.mnist.load_data()
    X = to_float(X)[..., None]
    return X, y


def fmnist():
    """
    Load the Fashion-MNIST dataset

    :return: Fashion-MNIST images and labels
    :rtype: tuple of np.ndaray
    """
    (X, y), _ = tf.keras.datasets.fashion_mnist.load_data()
    X = to_float(X)[..., None]
    return X, y


# Dictionary with available data-loading functions. Format: <dataset_name>: <dataset loader>. Used in 'main.py' to
# load the dataset specified by the '--dataset' argument.
LOADERS = {
    "mnist": mnist,
    "fmnist": fmnist,
}
