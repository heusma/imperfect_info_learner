import tensorflow as tf
import tensorflow_probability as tfp


def categorical_smoothing_function(dist: tfp.distributions.Categorical, factor: float) -> tfp.distributions.Categorical:
    probs = dist.probs_parameter()

    factor = tf.clip_by_value(factor, 0.0, 1.0)

    new_probs = probs * (1 - factor)
    new_probs += (factor / probs.shape[0])
    new_probs /= tf.reduce_sum(probs)

    return tfp.distributions.Categorical(tf.math.log(new_probs))
