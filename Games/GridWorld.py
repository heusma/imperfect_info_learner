import copy
import os
import random
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from Distributions.CategoricalDistribution import categorical_smoothing_function
from EvaluationTool import Estimator, VTraceTarget, VTraceGradients
from MDP import Game, State, InfoSet, Leaf, ActionSchema

size = 20
goal_x = 5
goal_y = 5


class GridWorldState(State):
    def __init__(self, x, y):
        self.position_x = x
        self.position_y = y


class GridWorldActionSchema(ActionSchema):
    def __init__(self, categorical_distribution: tfp.distributions.Categorical):
        self.dist = categorical_distribution

    def sample(self) -> Tuple[float, tf.Tensor]:
        s = self.dist.sample()
        p = self.dist.prob(s)
        return p, s

    def prob(self, action: tf.Tensor) -> tf.Tensor:
        p = self.dist.prob(action)
        return p

    def log_prob(self, action: tf.Tensor) -> tf.Tensor:
        l_p = self.dist.log_prob(action)
        return l_p


class GridWorldInfoSet(InfoSet):
    def __init__(self, state: GridWorldState):
        super().__init__(0)

        self.position_x = state.position_x
        self.position_y = state.position_y

    def get_action_schema(self) -> ActionSchema:
        return GridWorldActionSchema(tfp.distributions.Categorical(logits=tf.zeros(shape=(4,))))


class GridWorld(Game):
    @staticmethod
    def act(state: GridWorldState, info_set: InfoSet, action: tf.Tensor) -> Tuple[
        State or Leaf, InfoSet or Leaf, float]:
        assert tf.size(action) == 1

        direction = action

        new_state = copy.deepcopy(state)

        if not (new_state.position_x == goal_x and new_state.position_y == goal_y):
            if direction == 0:
                new_state.position_y += 1
            if direction == 1:
                new_state.position_x += 1
            if direction == 2:
                new_state.position_y -= 1
            if direction == 3:
                new_state.position_x -= 1

        new_state.position_x = min(max(0, new_state.position_x), size)
        new_state.position_y = min(max(0, new_state.position_y), size)

        if new_state.position_x == goal_x and new_state.position_y == goal_y:
            dr = tf.ones(shape=(1,))
            return Leaf(), Leaf(), dr
        else:
            dr = tf.zeros(shape=(1,))
            return new_state, GridWorldInfoSet(new_state), dr

    @staticmethod
    def get_root() -> Tuple[State, InfoSet]:
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        root_state = GridWorldState(start_x, start_y)
        return root_state, GridWorldInfoSet(root_state)

    @staticmethod
    def show_tile_values(estimator: Estimator):
        board = tf.zeros(shape=(size, size)).numpy()
        for i in range(size):
            for j in range(size):
                info_set = GridWorldInfoSet(GridWorldState(i, j))
                values, _ = zip(*estimator.evaluate([info_set]))
                board[i][j] = values[0]
        plt.imshow(board, interpolation='none')
        plt.show()

    @staticmethod
    def test_performance(estimator: any):
        GridWorld.show_tile_values(estimator)
        info_set = GridWorldInfoSet(GridWorldState(2, 3))
        values, action_schemas = zip(*estimator.evaluate([info_set]))
        tf.print("2, 3 estimate:")
        tf.print(values[0])
        tf.print(action_schemas[0].dist.probs_parameter())


def grid_world_exploration_function(action_schema: ActionSchema) -> ActionSchema:
    assert isinstance(action_schema, GridWorldActionSchema)

    smoothed_dist = categorical_smoothing_function(action_schema.dist, factor=0.4)

    return GridWorldActionSchema(smoothed_dist)


class GridWorldNetwork(tf.keras.Model):
    def __init__(self, num_layers: int, dff: int, outputs: int):
        super().__init__()

        self.internal_layers = []
        self.internal_layer_norms = []
        for _ in range(num_layers):
            self.internal_layers.append(
                tf.keras.layers.Dense(
                    dff,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0, mode='fan_in', distribution='truncated_normal'))
            )
            self.internal_layer_norms.append(
                tf.keras.layers.LayerNormalization()
            )
        self.output_layer = tf.keras.layers.Dense(
            outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, input, *args, **kwargs):
        activation = input
        for i in range(len(self.internal_layers)):
            l = self.internal_layers[i]
            ln = self.internal_layer_norms[i]
            activation = l(ln(activation))
        return self.output_layer(activation)


class GridWorldEstimator(Estimator):
    def __init__(self):
        self.weight_decay = 1e-4
        self.internal_network_policy = GridWorldNetwork(num_layers=3, dff=40, outputs=4)
        self.internal_network_value = GridWorldNetwork(num_layers=3, dff=40, outputs=1)
        self.optimizer_policy = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
        self.optimizer_value = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)

    def info_set_to_vector(self, info_set: InfoSet):
        assert isinstance(info_set, GridWorldInfoSet)
        return tf.cast([info_set.position_x / size, info_set.position_y / size], dtype=tf.float32)

    def vector_to_action_schema(self, logits):
        c_dist = tfp.distributions.Categorical(logits)
        return GridWorldActionSchema(c_dist)

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        output_policy = self.internal_network_policy(tf.stack(batch))
        action_schemas: List[ActionSchema] = [
            self.vector_to_action_schema(vector) for vector in tf.unstack(output_policy)
        ]
        output_value = self.internal_network_value(tf.stack(batch))
        values = tf.unstack(output_value, axis=0)
        return list(zip(values, action_schemas))

    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        with tf.GradientTape(persistent=True) as tape:
            info_sets, reach_weights, value_targets, q_value_targets = zip(*targets)
            value_estimates, on_policy_action_schemas = zip(*self.evaluate(info_sets))

            value_losses = reach_weights * tf.keras.losses.huber(
                y_pred=tf.stack(value_estimates), y_true=tf.stack(value_targets)
            )
            value_loss = tf.reduce_mean(value_losses)

            policy_losses = []
            for i in range(len(targets)):
                local_policy_losses = []
                action_schema = on_policy_action_schemas[i]
                assert isinstance(action_schema, GridWorldActionSchema)
                value_estimate = value_estimates[i]
                for action, importance, q_value in q_value_targets[i]:
                    advantage = q_value - value_estimate
                    on_policy_log_prob = action_schema.log_prob(action)
                    policy_loss = importance * on_policy_log_prob * advantage
                    local_policy_losses.append(policy_loss)
                policy_losses.append(-tf.reduce_mean(tf.stack(local_policy_losses)))

            policy_loss = tf.reduce_mean(reach_weights * tf.stack(policy_losses))

            for weights in self.internal_network_value.get_weights():
                value_loss += self.weight_decay * tf.nn.l2_loss(weights)

            for weights in self.internal_network_policy.get_weights():
                policy_loss += self.weight_decay * tf.nn.l2_loss(weights)

        tv = self.internal_network_value.trainable_variables
        value_grads = tape.gradient(value_loss, tv)

        tp = self.internal_network_policy.trainable_variables
        policy_grads = tape.gradient(policy_loss, tp)

        return value_grads, policy_grads

    def apply_gradients(self, grads: VTraceGradients):
        value_grads, policy_grads = grads

        tv = self.internal_network_value.trainable_variables
        value_grads, _ = tf.clip_by_global_norm(value_grads, 5.0)
        self.optimizer_value.apply_gradients(zip(value_grads, tv))

        tp = self.internal_network_policy.trainable_variables
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 5.0)
        self.optimizer_policy.apply_gradients(zip(policy_grads, tp))

    def save(self, checkpoint_location: str) -> None:
        pass

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        pass
