import tensorflow as tf
import tensorflow_probability as tfp


def forward_min_max_mapping(sample: tf.Tensor, min: float, max: float):
    s = (max - min) * sample + min
    return s


def inverse_min_max_mapping(sample: tf.Tensor, min: float, max: float):
    s = (sample - min) / (max - min)
    return s


class BoxDistribution:
    def __init__(self, min: float, max: float, c0, c1):
        assert max > min

        self.min = min
        self.max = max
        self.distance = max - min

        self.distribution = tfp.distributions.Beta(
            concentration0=1 + tf.math.exp(c0), concentration1=1 + tf.math.exp(c1),
            allow_nan_stats=False,
            force_probs_to_zero_outside_support=False,
        )

    def sample(self) -> tf.Tensor:
        s = self.distribution.sample()
        # now map this sample between min and max
        s = forward_min_max_mapping(s, self.min, self.max)
        return s

    def prob(self, sample: tf.Tensor):
        s = inverse_min_max_mapping(sample, self.min, self.max)
        return tf.clip_by_value(self.distribution.prob(s), 1e-12, 1e12)

    def log_prob(self, sample: tf.Tensor):
        s = inverse_min_max_mapping(sample, self.min, self.max)
        return tf.clip_by_value(self.distribution.log_prob(s), 1e-12, 1e12)


class SmoothedBoxDistribution:
    def __init__(self, box_dist: BoxDistribution, factor: float):
        self.box = box_dist
        self.uni = tfp.distributions.Uniform(low=0, high=1)

        self.factor = factor

        self.helper_dist = tfp.distributions.Categorical(probs=[factor, 1 - factor])

    def sample(self) -> tf.Tensor:
        chosen_dist_id = self.helper_dist.sample()
        if chosen_dist_id == 0:
            s = self.uni.sample()
            return forward_min_max_mapping(s, self.box.min, self.box.max)
        else:
            return self.box.sample()

    def prob(self, sample: tf.Tensor):
        box_prob = self.box.prob(sample)

        s = inverse_min_max_mapping(sample, self.box.min, self.box.max)
        uni_prob = self.uni.prob(s)

        return tf.clip_by_value(uni_prob * self.factor + (1 - self.factor) * box_prob, 1e-12, 1e12)

    def log_prob(self, sample: tf.Tensor):
        l_p = tf.math.log(self.prob(sample))
        return tf.clip_by_value(l_p, 1e-12, 1e12)


def box_smoothing_function(dist: BoxDistribution, factor: float) -> SmoothedBoxDistribution:
    return SmoothedBoxDistribution(dist, factor)
