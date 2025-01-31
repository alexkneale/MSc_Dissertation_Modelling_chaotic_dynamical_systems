import tensorflow as tf
tf.keras.backend.set_floatx("float64")

class Scaler:
    def __init__(self, X):
        assert len(X.shape) > 1, "X must have dimension greater than 1."
        self.mean = tf.constant(tf.reduce_mean(X, axis=0), dtype=tf.float64)
        self.std = tf.constant(tf.math.reduce_std(X, axis=0), dtype=tf.float64)

    def standardise(self, X):
        return (X - self.mean) / self.std

    def invert_standardisation(self, X):
        return (X * self.std) + self.mean

    def invert_standardisation_prob(self, prob):
        return prob / tf.reduce_prod(self.std)

    def invert_standardisation_log_prob(self, prob):
        return prob - tf.math.log(tf.reduce_prod(self.std))

    def invert_standardisation_loc(self, loc):
        return self.invert_standardisation(loc)

    def invert_standardisation_cov(self, cov):
        return cov * (self.std[:, None] @ self.std[None, :])