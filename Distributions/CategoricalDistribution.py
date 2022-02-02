import tensorflow as tf


class Categorical:
    def __init__(self, logits: tf.Tensor):
        self.probs = tf.math.softmax(logits)
        self.logits = tf.math.log(self.probs + tf.keras.backend.epsilon())

    def log_prob(self, s):
        return self.logits[s]

    def prob(self, s):
        return self.probs[s]

    def probs_parameter(self):
        return self.probs

    def sample(self):
        return tf.squeeze(tf.random.categorical(tf.expand_dims(self.logits, axis=0), num_samples=1, ))


def categorical_smoothing_function(dist: Categorical, factor: float) -> Categorical:
    probs = dist.probs_parameter()

    factor = tf.clip_by_value(factor, 0.0, 1.0)

    new_probs = probs * (1 - factor)
    new_probs += (factor / probs.shape[0])
    new_probs /= tf.reduce_sum(probs)

    return Categorical(tf.math.log(new_probs))
