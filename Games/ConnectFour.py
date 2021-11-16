import copy
import random
from typing import Tuple, List

import tensorflow as tf
import matplotlib.pyplot as plt

from EvaluationTool import PolicyEstimator, apply_estimator, perform_action
from Game import Game, StateNode, State, DiscreteDistribution, ChanceNode, InfoSet, ActionSchema, Node, Leaf


class ConnectFourNetwork(tf.keras.Model):
    def __init__(self, num_layers: int, dff: int, outputs: int):
        super().__init__()

        self.internal_layers = []
        for _ in range(num_layers):
            self.internal_layers.append(
                tf.keras.layers.Dense(dff, activation='relu')
            )
        self.internal_layers.append(
            tf.keras.layers.Dense(outputs, activation=None)
        )

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, input, *args, **kwargs):
        activation = input
        for l in self.internal_layers:
            activation = l(activation)
        return activation


class ConnectFourState(State):
    def __init__(self, current_player: 0, position_x: int, position_y: int):
        super().__init__(current_player)

        self.position_x = position_x
        self.position_y = position_y

    def to_info_set(self) -> InfoSet:
        return ConnectFourInfoSet(
            self.position_x,
            self.position_y,
        )


class ConnectFourActionSchema(ActionSchema):
    def __init__(self, chance_nodes: List[ChanceNode], root_node: int = 0):
        super().__init__(chance_nodes, root_node)

    def sample_internal(self) -> Tuple[List[float], List[float], List[float], ChanceNode, Node]:
        p, i, v, nn = self.root_node().sample()
        return [p], [i], [v], self.root_node(), nn


class ConnectFourInfoSet(InfoSet):
    def __init__(self, position_x: int, position_y: int):
        self.position_x = position_x
        self.position_y = position_y

    def get_action_schema(self) -> ActionSchema:
        cn = ChanceNode(DiscreteDistribution(4))
        cn.payoff_estimate = tf.zeros(shape=(1,))
        return ConnectFourActionSchema(
            [cn],
        )


class ConnectFourPolicyEstimator(PolicyEstimator):
    def __init__(self):
        self.internal_network_policy = ConnectFourNetwork(num_layers=4, dff=80, outputs=4)
        self.internal_network_value = ConnectFourNetwork(num_layers=4, dff=80, outputs=1)
        self.optimizer_policy = tf.keras.optimizers.SGD(0.0001)
        self.optimizer_value = tf.keras.optimizers.SGD(0.0001)

    def info_set_to_vector(self, info_set: ConnectFourInfoSet):
        return tf.constant([info_set.position_x / size, info_set.position_y / size], dtype=tf.float32)

    def vector_to_action_schema(self, vector: tf.Tensor):
        probs = vector[:4]
        payoff = vector[4:]
        distribution = DiscreteDistribution(4)
        distribution.assign(probs)
        cn = ChanceNode(distribution)
        cn.payoff_estimate = payoff
        return ConnectFourActionSchema(
            [cn]
        )

    def action_schema_to_vector(self, action_schema: ActionSchema):
        dist = action_schema.root_node().on_policy_distribution
        assert isinstance(dist, DiscreteDistribution)
        return tf.concat([dist.probs, action_schema.root_node().computed_payoff_estimate], axis=0)

    def __call__(self, info_sets: List[ConnectFourInfoSet], *args, **kwargs) -> List[ActionSchema]:
        batch = [self.info_set_to_vector(info_set) for info_set in info_sets]
        output_policy = self.internal_network_policy(tf.stack(batch))
        output_value = self.internal_network_value(tf.stack(batch))
        output = tf.concat([tf.math.softmax(output_policy, axis=-1), output_value], axis=-1)
        result = [self.vector_to_action_schema(vector) for vector in tf.unstack(output)]
        return result

    def train(self, info_sets: List[ConnectFourInfoSet], action_schema_targets: List[ActionSchema]):
        batch = tf.stack([self.info_set_to_vector(info_set) for info_set in info_sets])
        test = [self.action_schema_to_vector(action_schema_target) for action_schema_target in action_schema_targets]
        target_batch = tf.stack(test)

        with tf.GradientTape() as tape:
            output_value = self.internal_network_value(tf.stack(batch))

            value_loss = tf.keras.losses.mean_absolute_error(output_value, target_batch[:, 4:])
            tf.print(output_value[0])
            #tf.print(target_batch[:, 4:][0])
            tf.print("value_loss:")
            tf.print(tf.reduce_mean(value_loss))

            loss = tf.reduce_mean(value_loss)

            tv = self.internal_network_value.trainable_variables
            grads = tape.gradient(loss, tv)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer_value.apply_gradients(zip(grads, tv))

        with tf.GradientTape() as tape:
            output_policy = tf.math.softmax(self.internal_network_policy(tf.stack(batch)), axis=-1)

            policy_loss = tf.keras.losses.categorical_crossentropy(y_pred=output_policy, y_true=target_batch[:, :4])
            tf.print(output_policy[0])
            tf.print("policy_loss:")
            tf.print(tf.reduce_mean(policy_loss))

            loss = tf.reduce_mean(policy_loss)

            tv = self.internal_network_policy.trainable_variables
            grads = tape.gradient(loss, tv)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer_policy.apply_gradients(zip(grads, tv))


class ConnectFourStateNode(StateNode):
    def __init__(self, state: ConnectFourState):
        super().__init__(state)

    """
    The action is an int describing the direction of the next move.
        0
        |
    3 -   - 1
        |
        2
    """

    def act(self, action: List[int or float]) -> Tuple[List[float], StateNode or Leaf]:
        assert len(action) == 1
        direction = action[0]
        new_state = copy.deepcopy(self.state)
        assert isinstance(new_state, ConnectFourState)

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
            return tf.zeros(shape=(1,)), ConnectFourStateNode(new_state)


class ConnectFour(Game):
    @staticmethod
    def start(options: dict) -> StateNode:
        start_x = random.randint(0, size)
        start_y = random.randint(0, size)
        root_state = ConnectFourState(0, start_x, start_y)
        return ConnectFourStateNode(root_state)

    @staticmethod
    def show_tile_values(policy_estimator: PolicyEstimator):
        board = tf.zeros(shape=(size, size)).numpy()
        for i in range(size):
            for j in range(size):
                state_node = ConnectFourStateNode(ConnectFourState(0, i, j))
                apply_estimator([state_node], policy_estimator)
                action_schema = state_node.action_schema
                assert len(action_schema.chance_nodes) == 1
                board[i][j] = action_schema.chance_nodes[0].payoff_estimate[0]
        plt.imshow(board, interpolation='none')
        plt.show()

    @staticmethod
    def test_policy(policy_estimator: PolicyEstimator):
        root = ConnectFour.start(dict())
        node = root
        i = 0
        while not isinstance(node, Leaf) and i < size * size:
            i += 1
            apply_estimator([node], policy_estimator)
            tf.print([node.state.position_x, node.state.position_y])
            tf.print(node.action_schema.root_node().on_policy_distribution.probs)
            _, _, _, _, node = perform_action(node)
            if isinstance(node, Leaf):
                tf.print("Leaf")
        tf.print("---")