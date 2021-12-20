import copy
import os
import random
import sys
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from EvaluationTool import PolicyEstimator, apply_estimator, perform_action, rollout
from Game import Game, StateNode, State, DiscreteDistribution, ChanceNode, InfoSet, ActionSchema, Node, Leaf

rows = 6
columns = 7

non_player_value = -1


class ConnectFourState(State):
    def __init__(self, current_player: 0, board: tf.Tensor):
        super().__init__(current_player)

        self.board = board

    def to_info_set(self) -> InfoSet:
        return ConnectFourInfoSet(
            self.current_player,
            self.board,
        )


class ConnectFourActionSchema(ActionSchema):
    def __init__(self, chance_nodes: List[ChanceNode], root_node: int = 0):
        super().__init__(chance_nodes, root_node)

    def sample_internal(self, ghost_mode: bool = False):
        p, v, nn = self.root_node().sample(ghost_mode)
        return [p], [v], self.root_node(), nn


class ConnectFourInfoSet(InfoSet):
    def __init__(self, current_player: int, board: tf.Tensor):
        self.current_player = current_player
        self.board = board

    def get_action_schema(self) -> ActionSchema:
        cn = ChanceNode(DiscreteDistribution(columns))
        cn.payoff_estimate = tf.zeros(shape=(2,))
        return ConnectFourActionSchema(
            [cn],
        )


def check_lists_for_winner(lists: List[List[int]]):
    for li in lists:
        c = 1
        s = li[0]
        for i in range(1, len(li)):
            if li[i] == s:
                c += 1
            else:
                s = li[i]
                c = 1
            if c == 4 and s != non_player_value:
                return True
    return False


def check_board_for_winner_vertical(board: List[List[int]]) -> bool:
    return check_lists_for_winner(board)


def check_board_for_winner_horizontal(board: List[List[int]]) -> bool:
    return check_lists_for_winner(np.array(board).T.tolist())


def check_board_for_winner_diagonal_left(board: List[List[int]]) -> bool:
    start = [0, rows - 1]
    candidates = []
    while start[0] < columns:
        pointer = copy.deepcopy(start)
        diag = [
            board[pointer[0]][pointer[1]]
        ]
        while pointer[0] < columns - 1 and pointer[1] < rows - 1:
            pointer = [pointer[0] + 1, pointer[1] + 1]
            diag.append(board[pointer[0]][pointer[1]])

        candidates.append(diag)

        if start[1] > 0:
            start[1] -= 1
        else:
            start[0] += 1

    return check_lists_for_winner(candidates)


def check_board_for_winner_diagonal_right(board: List[List[int]]) -> bool:
    start = [columns - 1, rows - 1]
    candidates = []
    while start[0] > -1:
        pointer = copy.deepcopy(start)
        diag = [
            board[pointer[0]][pointer[1]]
        ]
        while pointer[0] > 0 and pointer[1] < rows - 1:
            pointer = [pointer[0] - 1, pointer[1] + 1]
            diag.append(board[pointer[0]][pointer[1]])

        candidates.append(diag)

        if start[1] > 0:
            start[1] -= 1
        else:
            start[0] -= 1

    return check_lists_for_winner(candidates)


def check_board_for_winner(board: List[List[int]]) -> bool:
    return check_board_for_winner_vertical(board) \
           or check_board_for_winner_horizontal(board) \
           or check_board_for_winner_diagonal_left(board) \
           or check_board_for_winner_diagonal_right(board)


def check_board_for_board_full(board: List[List[float]]) -> bool:
    is_full = True
    for column in board:
        if column[-1] == non_player_value:
            is_full = False
            break
    return is_full


class ConnectFourStateNode(StateNode):
    def __init__(self, state: ConnectFourState):
        super().__init__(state)

    """
    The action gives the column number where the next chip should be inserted.

    The board is a 2D List in (columns, rows) order.
    """

    def act(self, action: List[int or float]) -> Tuple[tf.Tensor, StateNode or Leaf]:
        assert len(action) == 1
        action = action[0]
        assert 0 <= action < columns
        new_state = copy.deepcopy(self.state)
        assert isinstance(new_state, ConnectFourState)

        board = new_state.board
        assert isinstance(board, List)

        # find_valid_columns
        valid_column_ids = []
        for column_id in range(columns):
            if board[column_id][-1] == non_player_value:
                valid_column_ids.append(column_id)

        # if the given action tries to make an illegal move choose a random id from the list instead.
        if action not in valid_column_ids:
            action = random.choice(valid_column_ids)

        # Insert a coin in the given column
        column = board[action]
        for i in range(rows):
            if column[i] == non_player_value:
                column[i] = new_state.current_player
                break

        # check if the game is over (there is a winner or no more valid columns)
        winner = check_board_for_winner(board)
        if winner:
            payout = [0.0, 0.0]
            payout[new_state.current_player] = 1.0
            return tf.cast(payout, dtype=tf.float32), Leaf()
        board_full = check_board_for_board_full(board)
        if board_full:
            return tf.cast([0.0, 0.0], dtype=tf.float32), Leaf()

        # if the game continues, create a new state from the edited board and switch players
        new_state.current_player = 1 - new_state.current_player
        return tf.cast([0.0, 0.0], dtype=tf.float32), ConnectFourStateNode(new_state)


class ConnectFour(Game):
    @staticmethod
    def start(options: dict) -> StateNode:
        inital_board = np.full([columns, rows], non_player_value, np.int32).tolist()
        root_state = ConnectFourState(0, inital_board)
        return ConnectFourStateNode(root_state)

    @staticmethod
    def test_policy_selfplay(policy_estimator: PolicyEstimator, samples, batch_size, discount):
        root = ConnectFour.start(dict())
        node = root
        i = 0
        while not isinstance(node, Leaf):
            i += 1
            apply_estimator([node], policy_estimator)
            tf.print(i)
            state = node.state
            assert isinstance(state, ConnectFourState)
            plt.imshow(np.rot90(state.board), interpolation='none')
            plt.show()

            # improve estimate
            rollout(node, samples, batch_size, policy_estimator)
            node.compute_payoff(discount)
            node.action_schema.optimize(node.state.current_player, discount)

            # act greedy
            tf.print(node.action_schema.root_node().on_policy_distribution.probs, summarize=columns)
            tf.print(
                "We estimate the payoff per player to be: {}".format(node.action_schema.root_node().computed_payoff_estimate)
            )
            node.action_schema = node.action_schema.gready()
            for cn in node.action_schema.chance_nodes:
                cn.children = dict()

            _, _, _, node = perform_action(node)
            if isinstance(node, Leaf):
                tf.print("Leaf")
                plt.imshow(np.rot90(state.board), interpolation='none')
                plt.show()
        tf.print("---")

    @staticmethod
    def test_policy_human(policy_estimator: PolicyEstimator, ai_player: int, samples, batch_size, discount):
        root = ConnectFour.start(dict())
        node = root
        i = 0
        while not isinstance(node, Leaf):
            i += 1
            apply_estimator([node], policy_estimator)
            tf.print(i)
            state = node.state
            assert isinstance(state, ConnectFourState)

            if node.state.current_player == ai_player:
                payoff_estimate = node.action_schema.root_node().payoff_estimate

                # improve estimate
                rollout(node, samples, batch_size, policy_estimator)
                node.compute_payoff(discount)
                node.action_schema.optimize(node.state.current_player, discount)
                payoff_estimate = node.action_schema.root_node().computed_payoff_estimate

                # act greedy
                tf.print(node.action_schema.root_node().on_policy_distribution.probs, summarize=columns)
                tf.print(
                    "We estimate the payoff per player to be: {}".format(
                        payoff_estimate)
                )
                node.action_schema = node.action_schema.gready()
                for cn in node.action_schema.chance_nodes:
                    cn.children = dict()

                _, _, _, node = perform_action(node)
            else:
                plt.imshow(np.rot90(state.board), interpolation='none')
                plt.show()

                move_id = input()
                move_id = int(move_id)
                _, node = node.act([move_id])

            if isinstance(node, Leaf):
                tf.print("Leaf")
                plt.imshow(np.rot90(state.board), interpolation='none')
                plt.show()
        tf.print("---")

    @staticmethod
    def test_policy_baseline(policy_estimator: PolicyEstimator):
        global baseline_payoff
        global baseline_payoff_lr

        ai_player = 0
        root = ConnectFour.start(dict())
        node = root
        i = 0
        while not isinstance(node, Leaf):
            i += 1
            if node.state.current_player == ai_player:
                apply_estimator([node], policy_estimator)
            else:
                node.action_schema = node.info_set.get_action_schema()
            state = node.state
            assert isinstance(state, ConnectFourState)
            tf.print(node.action_schema.root_node().on_policy_distribution.probs)
            _, _, _, node = perform_action(node)
            if isinstance(node, Leaf):
                tf.print("Leaf")
                ai_player_payoff = root.compute_payoff(discount=1.0)[ai_player]
                baseline_payoff += baseline_payoff_lr * (ai_player_payoff - baseline_payoff)
                tf.print(ai_player_payoff)
                tf.print(baseline_payoff)
        tf.print("---")


baseline_payoff = 0.5
baseline_payoff_lr = 0.01

"""
This is a strict copy of the network for GridWorld.
"""


class ConnectFourNetwork(tf.keras.Model):
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


class ConnectFourPolicyEstimator(PolicyEstimator):
    def __init__(self):
        self.weight_decay = 1e-4
        self.internal_network_policy = ConnectFourNetwork(num_layers=8, dff=400, outputs=columns)
        self.internal_network_value = ConnectFourNetwork(num_layers=8, dff=400, outputs=2)
        self.optimizer_policy = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)
        self.optimizer_value = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)

        self.version = 0

    def info_set_to_vector(self, info_set: ConnectFourInfoSet):
        board_tensor = tf.cast(info_set.board, dtype=tf.float32) + 1
        board_tensor_normed = board_tensor / tf.cast(tf.abs(non_player_value - 1), dtype=tf.float32) + 0.1
        board_vector = tf.reshape(board_tensor_normed, shape=(-1))
        return board_vector

    def vector_to_action_schema(self, vector: tf.Tensor):
        probs = vector[:columns]
        payoff = vector[columns:]
        distribution = DiscreteDistribution(columns)
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

            value_loss = tf.keras.losses.huber(y_pred=output_value, y_true=target_batch[:, columns:])

            loss = tf.reduce_mean(value_loss)

            for weights in self.internal_network_value.get_weights():
                loss += self.weight_decay * tf.nn.l2_loss(weights)

            tv = self.internal_network_value.trainable_variables
            grads = tape.gradient(loss, tv)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            self.optimizer_value.apply_gradients(zip(grads, tv))

        with tf.GradientTape() as tape:
            output_policy = tf.math.softmax(self.internal_network_policy(tf.stack(batch)), axis=-1)

            policy_loss = tf.keras.losses.categorical_crossentropy(y_pred=output_policy,
                                                                   y_true=target_batch[:, :columns])

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
                sleep(2)

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        if self.version == 0:
            self.internal_network_value(tf.zeros(shape=(1, columns * rows)))
            self.internal_network_policy(tf.zeros(shape=(1, columns * rows)))
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
                sleep(2)
                if blocking is False:
                    success = True

