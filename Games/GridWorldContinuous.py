import copy
import os
import random
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np

import matplotlib.pyplot as plt

from Distributions.BoxDistribution import BoxDistribution, box_smoothing_function
from EvaluationTool import Estimator, VTraceTarget, VTraceGradients, take_action, create_trajectory, \
    identity_exploration_function
from Helpers.hessianFreeOptimizer import get_natural_gradient, \
    get_natural_gradient_hessian_free_back_over_back, get_natural_gradient_hessian_free_forward_over_back
from Helpers.lbfgsOptimizer import get_lbfg_gradient
from MDP import Game, State, InfoSet, Leaf, ActionSchema

size = 20
goal_x = 5
goal_y = 5
goal_proximity = 0.1


class GridWorldState(State):
    def __init__(self, x, y):
        self.position_x = x
        self.position_y = y


class GridWorldContinuousActionSchema(ActionSchema):
    def __init__(self, dist_x: BoxDistribution, dist_y: BoxDistribution):
        self.dist_x = dist_x
        self.dist_y = dist_y

    def sample(self) -> Tuple[float, tf.Tensor]:
        s_x = self.dist_x.sample()
        p_x = self.dist_x.prob(s_x)

        s_y = self.dist_y.sample()
        p_y = self.dist_y.prob(s_y)

        return p_x * p_y, tf.concat([s_x, s_y], axis=0)

    def prob(self, action: tf.Tensor) -> float:
        p_x = self.dist_x.prob(action[0])
        p_y = self.dist_y.prob(action[1])
        p = p_x * p_y
        return p

    def log_prob(self, action: tf.Tensor) -> tf.Tensor:
        p_x = self.dist_x.prob(action[0])
        p_y = self.dist_y.prob(action[1])
        l_p = tf.math.log(p_x * p_y)
        return l_p


class GridWorldContinuousInfoSet(InfoSet):
    def __init__(self, state: GridWorldState):
        super().__init__(0)

        self.position_x = state.position_x
        self.position_y = state.position_y

    def get_action_schema(self) -> ActionSchema:
        return GridWorldContinuousActionSchema(
            BoxDistribution(min=-1, max=1, c0=0, c1=0),
            BoxDistribution(min=-1, max=1, c0=0, c1=0),
        )


class GridWorldContinuous(Game):
    @staticmethod
    def act(state: GridWorldState, info_set: InfoSet, action: tf.Tensor) -> Tuple[
        State or Leaf, InfoSet or Leaf, float]:
        assert tf.size(action) == 2

        new_state = copy.deepcopy(state)

        new_state.position_x += tf.clip_by_value(action[0], -1, 1)
        new_state.position_y += tf.clip_by_value(action[1], -1, 1)

        new_state.position_x = tf.cast(min(max(0.0, new_state.position_x), size), dtype=tf.float32)
        new_state.position_y = tf.cast(min(max(0.0, new_state.position_y), size), dtype=tf.float32)

        if tf.abs(new_state.position_x - goal_x) <= size * 0.1 and tf.abs(new_state.position_y - goal_y) <= size * 0.1:
            dr = tf.ones(shape=(1,))
            return Leaf(), Leaf(), dr
        else:
            dr = tf.zeros(shape=(1,))
            return new_state, GridWorldContinuousInfoSet(new_state), dr

    @staticmethod
    def get_root() -> Tuple[State, InfoSet]:
        start_x = random.random() * size
        start_y = random.random() * size
        root_state = GridWorldState(start_x, start_y)
        return root_state, GridWorldContinuousInfoSet(root_state)

    @staticmethod
    def show_tile_values(estimator: Estimator):
        board = tf.zeros(shape=(size, size)).numpy()
        for i in range(size):
            for j in range(size):
                info_set = GridWorldContinuousInfoSet(GridWorldState(i, j))
                values, _ = zip(*estimator.evaluate([info_set]))
                board[i][j] = values[0]
        plt.imshow(board, interpolation='none')
        plt.show()

    @staticmethod
    def test_policy(estimator: Estimator):
        root, root_info_set = GridWorldContinuous.get_root()
        traj = create_trajectory(GridWorldContinuous, root, root_info_set, estimator, identity_exploration_function,
                                 max_steps=size * size)

        tf.print("trajectory summary:")
        for te in traj:
            if isinstance(te, Leaf):
                tf.print("Leaf!")
                break
            _, info_set, value_estimate, _, _, _, _, action, direct_reward = te

    @staticmethod
    def test_performance(estimator: any):
        GridWorldContinuous.test_policy(estimator)
        GridWorldContinuous.show_tile_values(estimator)
        info_set = GridWorldContinuousInfoSet(GridWorldState(0, 0))
        values, action_schemas = zip(*estimator.evaluate([info_set]))
        tf.print("0, 0 estimate:")
        tf.print(values[0])
        tf.print(
            tf.concat([action_schemas[0].dist_x.distribution.concentration0,
                       action_schemas[0].dist_y.distribution.concentration0], axis=0)
        )
        tf.print(
            tf.concat([action_schemas[0].dist_x.distribution.concentration1,
                       action_schemas[0].dist_y.distribution.concentration1], axis=0)
        )
        sample_x = action_schemas[0].dist_x.sample()
        sample_y = action_schemas[0].dist_y.sample()
        tf.print(
            tf.concat([sample_x, sample_y], axis=0)
        )
        tf.print(
            tf.concat([action_schemas[0].dist_x.prob(sample_x), action_schemas[0].dist_y.prob(sample_y)], axis=0)
        )


def grid_world_continuous_exploration_function(action_schema: ActionSchema) -> ActionSchema:
    assert isinstance(action_schema, GridWorldContinuousActionSchema)
    return GridWorldContinuousActionSchema(
        box_smoothing_function(action_schema.dist_x, factor=0.2),
        box_smoothing_function(action_schema.dist_y, factor=0.2),
    )


class GridWorldContinuousNetwork(tf.keras.Model):
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
            activation = ln(l(activation))
        return self.output_layer(activation)


class GridWorldContinuousEstimator(Estimator):
    def __init__(self):
        self.weight_decay = 1e-4
        self.internal_network_policy = GridWorldContinuousNetwork(num_layers=3, dff=69, outputs=4)
        self.internal_network_value = GridWorldContinuousNetwork(num_layers=3, dff=69, outputs=1)

        self.optimizer_policy = tfa.optimizers.SGDW(
            weight_decay=self.weight_decay,
            learning_rate=0.005,
            momentum=0.9,
            nesterov=True,
        )
        self.optimizer_value = tfa.optimizers.SGDW(
            weight_decay=self.weight_decay,
            learning_rate=0.005,
            momentum=0.9,
            nesterov=True,
        )

        self.version = 0

    def get_variables(self) -> List[tf.Variable]:
        result = []
        result += self.internal_network_value.trainable_variables
        result += self.internal_network_policy.trainable_variables
        return result

    def info_set_to_vector(self, info_set: InfoSet):
        assert isinstance(info_set, GridWorldContinuousInfoSet)
        return tf.cast([info_set.position_x / size, info_set.position_y / size], dtype=tf.float32)

    def vector_to_action_schema(self, c0_x, c1_x, c0_y, c1_y):
        return GridWorldContinuousActionSchema(
            BoxDistribution(min=-1, max=1, c0=c0_x, c1=c1_x),
            BoxDistribution(min=-1, max=1, c0=c0_y, c1=c1_y),
        )

    def policy_network_to_action_schemas(self, output_policy):
        action_schemas: List[ActionSchema] = [
            self.vector_to_action_schema(loc_x, scale_x, loc_y, scale_y)
            for loc_x, scale_x, loc_y, scale_y in tf.unstack(output_policy)
        ]
        return action_schemas

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        output_policy = self.internal_network_policy(tf.stack(batch))
        action_schemas = self.policy_network_to_action_schemas(output_policy)
        output_value = self.internal_network_value(tf.stack(batch))
        values = tf.unstack(output_value, axis=0)
        return list(zip(values, action_schemas))

    def policy_loss(self, model, batch, targets_and_value_estimates_and_weight_decay):
        output_policy = self.internal_network_policy(tf.stack(batch))
        on_policy_action_schemas = self.policy_network_to_action_schemas(output_policy)
        targets, value_estimates, weight_decay = targets_and_value_estimates_and_weight_decay
        info_sets, reach_weights, value_targets, q_value_targets = zip(*targets)

        policy_losses = []
        for i in range(len(targets)):
            local_policy_losses = []
            action_schema = on_policy_action_schemas[i]
            assert isinstance(action_schema, GridWorldContinuousActionSchema)
            value_estimate = value_estimates[i]
            for action, importance, q_value in q_value_targets[i]:
                advantage = q_value - value_estimate
                on_policy_log_prob = action_schema.log_prob(action)
                policy_loss = importance * on_policy_log_prob * advantage
                local_policy_losses.append(policy_loss)
            policy_losses.append(-tf.reduce_mean(tf.stack(local_policy_losses)))

        policy_loss = tf.reduce_mean(reach_weights * tf.stack(policy_losses))

        return policy_loss

    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        info_sets, reach_weights, value_targets, q_value_targets = zip(*targets)
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        with tf.GradientTape() as tape:
            output_value = self.internal_network_value(tf.stack(batch))
            value_estimates = tf.unstack(output_value, axis=0)

            value_losses = reach_weights * tf.keras.losses.huber(
                y_pred=tf.stack(value_estimates), y_true=tf.stack(value_targets)
            )
            value_loss = tf.reduce_mean(value_losses)

        tv = self.internal_network_value.trainable_variables
        value_grads = tape.gradient(value_loss, tv)

        with tf.GradientTape() as inner_tape:
            loss = self.policy_loss(self.internal_network_policy, batch, (
                targets, value_estimates, self.weight_decay,
            ))
        policy_grads = inner_tape.gradient(loss, self.internal_network_policy.trainable_variables)

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
        value_net_location = os.path.dirname(checkpoint_location) + "/data/value"
        policy_net_location = os.path.dirname(checkpoint_location) + "/data/policy"
        checkpoint_object = jsonpickle.encode({
            "version": random.randint(1, 10000000),
            "value_net_location": value_net_location,
            "policy_net_location": policy_net_location
        })
        success = False
        while success is False:
            try:
                os.makedirs(os.path.dirname(value_net_location), exist_ok=True)
                self.internal_network_value.save_weights(value_net_location)
                self.internal_network_policy.save_weights(policy_net_location)
                with open(checkpoint_location, 'w') as f:
                    f.write(checkpoint_object)
                success = True
                tf.print("saved")
            except:
                tf.print("save error")
                sleep(1)

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        self.internal_network_value(tf.zeros(shape=(1, 2)))
        self.internal_network_policy(tf.zeros(shape=(1, 2)))
        success = False
        while success is False:
            try:
                with open(checkpoint_location, 'r') as f:
                    json_object = f.read()
                    checkpoint = jsonpickle.decode(json_object)
                    version = checkpoint["version"]
                    if self.version != version:
                        tf.print("loaded new version")
                        self.internal_network_value.load_weights(checkpoint["value_net_location"])
                        self.internal_network_policy.load_weights(checkpoint["policy_net_location"])
                        self.version = version
                    success = True
            except:
                tf.print("load error")
                sleep(10)
                if blocking is False:
                    success = True