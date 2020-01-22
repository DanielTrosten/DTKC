import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.callbacks import EarlyStopping
import argparse

from data import LOADERS
from dtkc import DTKC


MODELS_DIR = "models"
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dataset", dest="dataset_name", type=str, default="mnist", help="Name of the dataset. Can be "
                                                                                      "either 'mnist' or 'fmnist'")
PARSER.add_argument("--n_clusters", dest="n_clusters", type=int, default=10, help="Number of clusters")
PARSER.add_argument("--n_runs", dest="n_runs", type=int, default=20, help="Number of runs")
PARSER.add_argument("--n_epochs", dest="n_epochs", type=int, default=100, help="Number of epochs")
PARSER.add_argument("--batch_size", dest="batch_size", type=int, default=120, help="Batch size")
PARSER.add_argument("--sigma", dest="sigma", default=0.15, type=float, help="Scaling factor for the sigma "
                                                                            "hyperparameter")
PARSER.add_argument("--lam", dest="lam", default=0.01, type=float, help="Lambda hyperparameter")
PARSER.add_argument("--n_hidden", dest="n_hidden", default=100, type=int, help="Number of units in the hidden layer")
PARSER.add_argument("--hidden_activation", dest="hidden_activation", default="relu", type=str, help="Activation "
                                                                                                    "function for the "
                                                                                                    "hidden layer")
PARSER.add_argument("--batch_norm", dest="batch_norm", default=True, type=bool, help="Use batch normalization after the"
                                                                                     " hidden layer")
PARSER.add_argument("--clipnorm", dest="clipnorm", default=10, type=float, help="Gradient norm for gradient clipping")
PARSER.add_argument("--learning_rate", dest="learning_rate", default=1E-3, type=float, help="Learning rate for the "
                                                                                            "Adam optimizer")
PARSER.add_argument("--use_companion_losses", dest="use_companion_losses", default=True, type=bool, help="Enable "
                                                                                                         "companion "
                                                                                                         "objectives?")
ARGS = PARSER.parse_args()


def to_onehot(y, k=None):
    if k is None:
        k = len(np.unique(y))
    return np.eye(k)[y]


def run_multiple():
    """
    Run multiple DTKC-training runs. The training parameters are specified as command line arguments.
    """
    X, y = LOADERS[ARGS.dataset_name]()
    if y is not None:
        y_oh = to_onehot(y, ARGS.n_clusters)
    else:
        y_oh = None

    dtkc_params = dict(
        input_shape=X.shape[1:],
        n_clusters=ARGS.n_clusters,
        sigma=ARGS.sigma,
        lam=ARGS.lam,
        n_hidden=ARGS.n_hidden,
        hidden_activation=ARGS.hidden_activation,
        batch_norm=ARGS.batch_norm,
        clipnorm=ARGS.clipnorm,
        learning_rate=ARGS.learning_rate,
        use_companion_losses=ARGS.use_companion_losses,
    )

    fit_params = dict(
        X=X, y=y_oh,
        epochs=ARGS.n_epochs,
        batch_size=ARGS.batch_size,
        verbose=1,
    )

    tag = "_" + str(int(time.time()))
    save_name = lambda r: os.path.join(ARGS.dataset_name + tag, "run-{}").format(r)

    for r in range(ARGS.n_runs):
        fit_params["save_name"] = save_name(r)
        fit_params["callbacks"] = [EarlyStopping(monitor="loss", patience=30, restore_best_weights=True)]
        run_experiment(dtkc_params, fit_params)


def run_experiment(dtkc_params, fit_params):
    """
    Perform one DTKC training run.

    :param dtkc_params: Initialization parameters for the DTKC model
    :type dtkc_params: dict
    :param fit_params: Training parameters for the DTKC model
    :type fit_params: dict
    """
    tf.keras.backend.clear_session()
    dtkc = DTKC(**dtkc_params)
    dtkc.compile()
    dtkc.fit(**fit_params)


if __name__ == '__main__':
    disable_eager_execution()
    run_multiple()
