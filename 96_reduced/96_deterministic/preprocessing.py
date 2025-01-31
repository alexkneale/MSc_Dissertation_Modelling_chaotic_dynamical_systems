"""
Data preprocessing tools.
"""

import tensorflow as tf
tf.keras.backend.set_floatx("float64")



class Scaler:
    """
    Class for scaling data by standardisation. Includes methods for inverting
    the scaling of data and related probability densities, means and
    covariances.
    """

    def __init__(self, X):
        assert len(X.shape) > 1, "X must have dimension greater than 1."
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def standardise(self, X):
        return (X - self.mean) / self.std

    def invert_standardisation(self, X):
        return (X * self.std) + self.mean
