import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers

from kernels import get_tensor_kernel, get_vector_kernel
from loss import get_loss_funcs


MODELS_DIR = "models"
DEFAULT_CNN_LAYERS = (
    # Block 1
    ("conv", 5, 5, 32, "relu"),
    ("conv", 5, 5, 32, None),
    ("bn",),
    ("relu",),
    ("pool", 2, 2),
    ("K",),
    # Block 2
    ("conv", 3, 3, 32, "relu"),
    ("conv", 3, 3, 32, None),
    ("bn",),
    ("relu",),
    ("pool", 2, 2),
    ("K",),
)


class DTKC:
    def __init__(self, input_shape, n_clusters, sigma=0.15, lam=0.01, n_hidden=100, hidden_activation="relu", batch_norm=True,
                 cnn_layers=DEFAULT_CNN_LAYERS, clipnorm=10, learning_rate=1E-3, use_companion_losses=True):
        """
        Class representing the DTKC model.

        :param input_shape: Shape of the input images (height, width, channels).
        :type input_shape: tuple
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :param sigma: Scaling factor for the sigma hyperparameter
        :type sigma: float
        :param lam: Lambda hyperparameter
        :type lam: float
        :param n_hidden: Number of units in the first fully-connected (hidden) layer
        :type n_hidden: int
        :param hidden_activation: Activation function in the hidden layer.
        :type hidden_activation: str
        :param batch_norm: Use batch normalization after the hidden layer?
        :type batch_norm: bool
        :param cnn_layers: List or tuple where each element is a tuple with parameters for a layer or operation in the
                           CNN. See 'DEFAULT_CNN_LAYERS' for an example.
        :type cnn_layers: list or tuple
        :param clipnorm: Gradient norm for gradient clipping.
        :type clipnorm: float
        :param learning_rate: Learning rate for Adam optimizer.
        :type learning_rate: float
        :param use_companion_losses: Use companion objectives? When False, DTKC becomes DDC.
        :type use_companion_losses: bool
        """
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.lam = lam
        self.n_hidden = n_hidden
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.cnn_layers = cnn_layers
        self.clipnorm = clipnorm
        self.learning_rate = learning_rate
        self.use_companion_losses = use_companion_losses

        self.input_layer = tf.keras.Input(shape=self.input_shape)

        # Build the different model components
        self._build_cnn()
        self._build_clustering_module()
        self._build_kernels()
        # Get the loss functions
        self.loss_funcs, self.total_loss_func = get_loss_funcs(self.kernels, self.n_clusters, self.lam)
        # Create a Keras-model
        self.model = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)
        self.model.summary()

    def _build_cnn(self):
        """
        Create the CNN
        """
        conv_padding = "valid"
        pool_padding = "valid"
        conv_initializer = "he_uniform"
        self.block_outputs = []

        layer = self.input_layer

        for layer_type, *layer_params in self.cnn_layers:
            if layer_type == "conv":
                kx, ky, n_filters, activation = layer_params
                layer = layers.Conv2D(filters=n_filters, kernel_size=(kx, ky), padding=conv_padding,
                                      activation=activation, kernel_initializer=conv_initializer)(layer)

            elif layer_type == "pool":
                layer = layers.MaxPool2D(pool_size=[*layer_params], padding=pool_padding)(layer)

            elif layer_type == "bn":
                layer = layers.BatchNormalization(axis=-1)(layer)

            elif layer_type == "relu":
                layer = tf.keras.activations.relu(layer)

            elif layer_type == "K":
                self.block_outputs.append(layer)

            else:
                raise ValueError("Received unknown layer type " + layer_type)

            self.cnn_output = layer

    def _build_clustering_module(self):
        """
        Create the clustering module
        """
        layer = layers.Flatten()(self.cnn_output)
        layer = layers.Dense(units=self.n_hidden, activation=self.hidden_activation,
                             kernel_initializer="he_uniform")(layer)
        # Batch Norm
        if self.batch_norm:
            layer = layers.BatchNormalization()(layer)
        # Hidden layer
        self.hidden_layer = layer
        # Output-layer
        self.logits = layers.Dense(units=self.n_clusters, activation=None)(layer)
        self.output_layer = tf.keras.activations.softmax(self.logits)

    def _build_kernels(self):
        """
        Extract the kernels for companion objective computations
        """
        self.kernels = []

        if self.use_companion_losses:
            # Kernels from block-outputs
            for bo in self.block_outputs:
                self.kernels.append(get_tensor_kernel(bo, self.sigma))

        # Kernel from hidden layer
        self.kernels.append(get_vector_kernel(self.hidden_layer, self.sigma))

    def compile(self):
        """
        Compile the Keras model
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm),
            loss=self.total_loss_func,
            metrics=self.loss_funcs,
        )

    def _make_save_dir(self, save_name):
        """
        Create the directory for model-saving if it does not exist.
        :param save_name: Name of directory
        :type save_name: str
        """
        save_dir = os.path.join(MODELS_DIR, save_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    def save(self, save_name, res=None):
        """
        Save the model.

        :param save_name: Name of save-directory
        :type save_name: str
        :param res: Optional result-object from model.fit
        :type res:
        """
        save_dir = os.path.join(MODELS_DIR, save_name)
        print("Saving model to:", save_dir)

        self.model.save(os.path.join(save_dir, "dtkc.h5"))
        if res is not None:
            with open(os.path.join(save_dir, "history.pkl"), "wb") as history_file:
                pickle.dump(res.history, history_file)

    def fit(self, X, y, save_name=None, **kwargs):
        """
        Train the DTKC model.

        :param X: Training images.
        :type X: np.ndarray
        :param y:
        :type y:
        :param save_name: Name of save-directory. If None, the model will not be saved.
        :type save_name: str or None
        :param kwargs: Keyword arguments passed to model.fit
        :type kwargs: dict
        :return: Result-object from model.fit
        :rtype:
        """
        if save_name is not None:
            self._make_save_dir(save_name)
        res = self.model.fit(X, y, **kwargs)
        if save_name is not None:
            self.save(save_name, res=res)
        return res

    def predict(self, *args, **kwargs):
        """
        Predict cluster membership vectors for the DTKC model.

        :param args: Arguments passed to model.predict
        :type args: list
        :param kwargs: Keyword arguments passed to model.predict
        :type kwargs: dict
        :return: Predicted cluster membership vectors
        :rtype: np.ndarray
        """
        return self.model.predict(*args, **kwargs)

    def load(self, dataset_name, run, tag=None, checkpoint=None):
        """
        Load parameters from a saved model. The model will be loaded from '<MODELS_DIR>/<dataset_name>_<tag>/run-<run>.

        :param dataset_name: Name of the dataset the model was trained on
        :type dataset_name: str
        :param run: Which run to load
        :type run: int
        :param tag: Tag which specifies which model to load. If None, the latest model will be loaded.
        :type tag: str or None
        :param checkpoint: Restore parameters from a specific checkpoint? If None, the model saved after training will
                           be loaded.
        :type checkpoint: int or None
        """
        if tag is None:
            dirnames = sorted([d for d in os.listdir(MODELS_DIR) if d.startswith(dataset_name + "_")])
            tag = dirnames[-1].split("_")[-1]
        model_path = lambda fname: os.path.join(MODELS_DIR, dataset_name + "_" + tag, "run-" + str(run), "{}.h5"
                                                .format(fname))
        fname = model_path("dtkc") if checkpoint is None else model_path("checkpoint_{:03d}".format(checkpoint))
        self.model.load_weights(fname, by_name=False)
