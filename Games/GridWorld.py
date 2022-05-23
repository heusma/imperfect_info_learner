import copy
import os
import random
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt

from Distributions.CategoricalDistribution import categorical_smoothing_function, Categorical
from EvaluationTool import Estimator, VTraceTarget, VTraceGradients, CompiledModel
from MDP import Game, State, InfoSet, Leaf, ActionSchema

size = 5
goal_x = 2
goal_y = 2


class GridWorldState(State):
    def __init__(self, x, y):
        self.position_x = x
        self.position_y = y


class GridWorldActionSchema(ActionSchema):
    def __init__(self, categorical_distribution: Categorical):
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
        return GridWorldActionSchema(Categorical(logits=tf.zeros(shape=(4,))))


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

    smoothed_dist = categorical_smoothing_function(action_schema.dist, factor=0.2)

    return GridWorldActionSchema(smoothed_dist)


class GridWorldNetwork(CompiledModel):
    def __init__(self, optimizer, num_layers: int, dff: int, outputs: int):
        super().__init__(optimizer)

        self.internal_layers = []
        self.internal_layer_norms = [tf.keras.layers.LayerNormalization()]
        for _ in range(num_layers):
            self.internal_layers.append(
                tf.keras.layers.Dense(
                    dff,
                    activation=tf.keras.layers.LeakyReLU(),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0, mode='fan_in', distribution='truncated_normal'))
            )
            self.internal_layer_norms.append(
                tf.keras.layers.LayerNormalization()
            )
        self.input_layer = tf.keras.layers.Dense(
            dff,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )
        self.output_layer = tf.keras.layers.Dense(
            outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, input, *args, **kwargs):
        activation = self.internal_layers[-1](self.input_layer(input))
        for i in range(len(self.internal_layers)):
            l_in = activation
            l = self.internal_layers[i]
            ln = self.internal_layer_norms[i]
            activation = ln(l(activation))
            activation = (activation + l_in) / 2
        return self.output_layer(activation)


class GridWorldEstimator(Estimator):
    def __init__(self):
        weight_decay = 1e-4
        optimizer_policy = tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=0.001,
        )
        optimizer_value = tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=0.001,
        )

        value_network = GridWorldNetwork(num_layers=5, dff=100, outputs=1, optimizer=optimizer_value)
        policy_network = GridWorldNetwork(num_layers=5, dff=100, outputs=4, optimizer=optimizer_policy)

        super().__init__(value_network, policy_network)

        self.version = 0

    def get_variables(self) -> List[tf.Variable]:
        result = []
        result += self.value_network.trainable_variables
        result += self.policy_network.trainable_variables
        return result

    def info_set_to_vector(self, info_set: InfoSet):
        assert isinstance(info_set, GridWorldInfoSet)
        return tf.cast([info_set.position_x / size, info_set.position_y / size], dtype=tf.float32)

    def vector_to_action_schema(self, logits):
        c_dist = Categorical(logits)
        return GridWorldActionSchema(c_dist)

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        output_policy = self.policy_network(tf.stack(batch))
        action_schemas: List[ActionSchema] = [
            self.vector_to_action_schema(vector) for vector in tf.unstack(output_policy)
        ]
        output_value = self.value_network(tf.stack(batch))
        values = tf.unstack(output_value, axis=0)
        return list(zip(values, action_schemas))

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
                self.value_network.save_weights(value_net_location)
                self.policy_network.save_weights(policy_net_location)
                with open(checkpoint_location, 'w') as f:
                    f.write(checkpoint_object)
                success = True
                tf.print("saved")
            except:
                tf.print("save error")
                sleep(1)

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        self.policy_network(tf.zeros(shape=(1, 2)))
        self.value_network(tf.zeros(shape=(1, 2)))
        success = False
        while success is False:
            try:
                with open(checkpoint_location, 'r') as f:
                    json_object = f.read()
                    checkpoint = jsonpickle.decode(json_object)
                    version = checkpoint["version"]
                    if self.version != version:
                        tf.print("loaded new version")
                        self.value_network.load_weights(checkpoint["value_net_location"])
                        self.policy_network.load_weights(checkpoint["policy_net_location"])
                        self.version = version
                    success = True
            except:
                tf.print("load error")
                sleep(10)
                if blocking is False:
                    success = True