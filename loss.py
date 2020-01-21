"""
Loss function computations
"""

import tensorflow as tf
from kernels import cdist

EPSILON = 1E-6


def triu(X, name=None):
    # Sum of strictly upper triangular part
    return tf.reduce_sum(tf.linalg.band_part(X, 0, -1) - tf.linalg.band_part(X, 0, 0), name=name)


def d_cs(A, K, n_clusters):
    nom = tf.transpose(A) @ K @ A
    dnom = tf.sqrt(tf.expand_dims(tf.linalg.diag_part(nom), -1) @ tf.expand_dims(tf.linalg.diag_part(nom), 0) + EPSILON)
    d = 2 / (n_clusters * (n_clusters - 1) ) * triu(nom / dnom)
    return d


def get_loss_funcs(block_kernels, n_clusters, lam):
    # Sum of companion objectives
    def companion_loss(_, A):
        losses = []
        for K in block_kernels[:-1]:
            losses.append(lam * d_cs(A, K, n_clusters))
        return sum(losses)

    # L_1
    def loss_1(_, A):
        return d_cs(A, block_kernels[-1], n_clusters)

    # L_2
    def loss_2(_, A):
        return triu(A @ tf.transpose(A))

    # L_3
    def loss_3(_, A):
        return d_cs(tf.exp(-cdist(A, tf.eye(n_clusters))), block_kernels[-1], n_clusters)

    losses = [companion_loss, loss_1, loss_2, loss_3]

    # Total loss
    def total_loss(*args):
        return companion_loss(*args) + loss_1(*args) + loss_2(*args) + loss_3(*args)

    return losses, total_loss
