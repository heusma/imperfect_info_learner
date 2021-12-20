import copy
import os
import random
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import matplotlib.pyplot as plt

from EvaluationTool import PolicyEstimator, apply_estimator, perform_action
from Game import Game, StateNode, State, DiscreteDistribution, ChanceNode, InfoSet, ActionSchema, Node, Leaf

size = 10
goal_x = 2
goal_y = 2


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
            activation = ln(l(activation))
        return self.output_layer(activation)


class GridWorldState(State):
    def __init__(self, current_player: 0, position_x: int, position_y: int):
        super().__init__(current_player)

        self.position_x = position_x
        self.position_y = position_y

    def to_info_set(self) -> InfoSet:
        return GridWorldInfoSet(
            self.position_x,
            self.position_y,
        )


class GridWorldActionSchema(ActionSchema):
    def __init__(self, chance_nodes: List[ChanceNode], root_node: int = 0):
        super().__init__(chance_nodes, root_node)

    def sample_internal(self, ghost_mode: bool = False):
        p, v, nn = self.root_node().sample(ghost_mode)
        return [p], [v], self.root_node(), nn


class GridWorldInfoSet(InfoSet):
    def __init__(self, position_x: int, position_y: int):
        self.position_x = position_x
        self.position_y = position_y

    def get_action_schema(self) -> ActionSchema:
        cn = ChanceNode(DiscreteDistribution(4))
        cn.payoff_estimate = tf.zeros(shape=(1,))
        return GridWorldActionSchema(
            [cn],
        )


class GridWorldPolicyEstimator(PolicyEstimator):
    def __init__(self):
        self.weight_decay = 1e-4
        self.internal_network_policy = GridWorldNetwork(num_layers=3, dff=100, outputs=4)
        self.internal_network_value = GridWorldNetwork(num_layers=3, dff=100, outputs=1)
        self.optimizer_policy = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
        self.optimizer_value = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)

        self.version = 0

    def info_set_to_vector(self, info_set: GridWorldInfoSet):
        return tf.constant([info_set.position_x / size, info_set.position_y / size], dtype=tf.float32)

    def vector_to_action_schema(self, vector: tf.Tensor):
        probs = vector[:4]
        payoff = vector[4:]
        distribution = DiscreteDistribution(4)
        distribution.assign(probs)
        cn = ChanceNode(distribution)
        cn.payoff_estimate = payoff
        return GridWorldActionSchema(
            [cn]
        )

    def action_schema_to_vector(self, action_schema: ActionSchema):
        dist = action_schema.root_node().on_policy_distribution
        assert isinstance(dist, DiscreteDistribution)
        return tf.concat([dist.probs, action_schema.root_node().computed_payoff_estimate], axis=0)

    def __call__(self, info_sets: List[GridWorldInfoSet], *args, **kwargs) -> List[ActionSchema]:
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        output_policy = self.internal_network_policy(tf.stack(batch))
        output_value = self.internal_network_value(tf.stack(batch))
        output = tf.concat([tf.math.softmax(output_policy, axis=-1), output_value], axis=-1)
        result = [self.vector_to_action_schema(vector) for vector in tf.unstack(output)]
        return result

    def train(self, info_sets: List[GridWorldInfoSet], action_schema_targets: List[ActionSchema]):
        batch = tf.stack([self.info_set_to_vector(info_set) for info_set in info_sets])
        test = [self.action_schema_to_vector(action_schema_target) for action_schema_target in action_schema_targets]
        target_batch = tf.stack(test)

        with tf.GradientTape() as tape:
            output_value = self.internal_network_value(tf.stack(batch))

            value_loss = tf.keras.losses.huber(y_pred=output_value, y_true=target_batch[:, 4:])

            loss = tf.reduce_mean(value_loss)

            for weights in self.internal_network_value.get_weights():
                loss += self.weight_decay * tf.nn.l2_loss(weights)

            tv = self.internal_network_value.trainable_variables
            grads = tape.gradient(loss, tv)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            self.optimizer_value.apply_gradients(zip(grads, tv))

        with tf.GradientTape() as tape:
            output_policy = tf.math.softmax(self.internal_network_policy(tf.stack(batch)), axis=-1)

            policy_loss = tf.keras.losses.categorical_crossentropy(y_pred=output_policy, y_true=target_batch[:, :4])

            loss = tf.reduce_mean(policy_loss)

            for weights in self.internal_network_policy.get_weights():
                loss += self.weight_decay * tf.nn.l2_loss(weights)

            tv = self.internal_network_policy.trainable_variables
            grads = tape.gradient(loss, tv)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            self.optimizer_policy.apply_gradients(zip(grads, tv))

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
        if self.version == 0:
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


class GridWorldStateNode(StateNode):
    def __init__(self, state: GridWorldState):
        super().__init__(state)

    """
    The action is an int describing the direction of the next move.
        0
        |
    3 -   - 1
        |
        2
    """

    def act(self, action: List[int or float]) -> Tuple[tf.Tensor, StateNode or Leaf]:
        assert len(action) == 1
        direction = action[0]
        new_state = copy.deepcopy(self.state)
        assert isinstance(new_state, GridWorldState)

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
            return tf.ones(shape=(1,)), Leaf()
        else:
            return tf.zeros(shape=(1,)), GridWorldStateNode(new_state)


class GridWorld(Game):
    @staticmethod
    def start(options: dict) -> StateNode:
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        root_state = GridWorldState(0, start_x, start_y)
        return GridWorldStateNode(root_state)

    @staticmethod
    def show_tile_values(policy_estimator: PolicyEstimator):
        board = tf.zeros(shape=(size, size)).numpy()
        for i in range(size):
            for j in range(size):
                state_node = GridWorldStateNode(GridWorldState(0, i, j))
                apply_estimator([state_node], policy_estimator)
                action_schema = state_node.action_schema
                assert len(action_schema.chance_nodes) == 1
                board[i][j] = action_schema.chance_nodes[0].payoff_estimate[0]
        plt.imshow(board, interpolation='none')
        plt.show()

    @staticmethod
    def test_policy(policy_estimator: PolicyEstimator):
        root = GridWorld.start(dict())
        node = root
        i = 0
        while not isinstance(node, Leaf) and i < size * size:
            i += 1
            apply_estimator([node], policy_estimator)
            tf.print([node.state.position_x, node.state.position_y])
            tf.print(node.action_schema.root_node().on_policy_distribution.probs)
            _, _, _, node = perform_action(node)
            if isinstance(node, Leaf):
                tf.print("Leaf")
        tf.print("---")
