import copy
import random
from abc import abstractmethod
from threading import Thread
from time import sleep
from typing import List, Callable, Tuple

import tensorflow as tf
from mpi4py import MPI

from Game import StateNode, Leaf, ActionSchema, InfoSet, Game, ChanceNode


class PolicyEstimator:
    @abstractmethod
    def __call__(self, info_sets: List[InfoSet], *args, **kwargs) -> List[ActionSchema]:
        pass

    @abstractmethod
    def train(self, info_sets: List[InfoSet], action_schema_targets: List[ActionSchema]):
        pass


def to_key(info_set: InfoSet):
    return str(list(info_set.__dict__.values()))


class TabularPolicyEstimator(PolicyEstimator):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.table = dict()

    def __call__(self, info_sets: List[InfoSet], *args, **kwargs) -> List[ActionSchema]:
        results: List[None or ActionSchema] = [None] * len(info_sets)
        for id in range(len(info_sets)):
            info_set = info_sets[id]
            key = to_key(info_set)
            if key in self.table:
                results[id] = copy.deepcopy(self.table[key])
            else:
                results[id] = info_set.get_action_schema()
                self.table[key] = copy.deepcopy(results[id])
        return results

    def train(self, info_sets: List[InfoSet], action_schema_targets: List[ActionSchema]):
        for info_set, policy_target in zip(info_sets, action_schema_targets):
            key = to_key(info_set)
            if key in self.table:
                p = self.table[key]
                assert isinstance(p, ActionSchema)
                for t_cn, s_cn in zip(p.chance_nodes, policy_target.chance_nodes):
                    s_cn.payoff_estimate = s_cn.computed_payoff_estimate
                    if t_cn.payoff_estimate is None:
                        t_cn.payoff_estimate = s_cn.payoff_estimate
                    else:
                        distance = s_cn.payoff_estimate - t_cn.payoff_estimate
                        t_cn.payoff_estimate += self.learning_rate * distance
                    t_cn_params = t_cn.on_policy_distribution.get_parameters()
                    s_cn_params = s_cn.on_policy_distribution.get_parameters()
                    t_cn_params += self.learning_rate * (s_cn_params - t_cn_params)
                    t_cn.on_policy_distribution.set_parameters(t_cn_params)
            else:
                policy_target_copy = copy.deepcopy(policy_target)
                for cn in policy_target_copy.chance_nodes:
                    cn.children = dict()
                self.table[key] = policy_target_copy


def default_exploration_policy_function(action_schema: ActionSchema):
    for cn in action_schema.chance_nodes:
        if cn.off_policy_distribution is None:
            dist = copy.deepcopy(cn.on_policy_distribution)
            factor = 0.1
            dist.probs *= (1 - factor)
            dist.probs += (factor / dist.probs.shape[0])
            dist.probs /= tf.reduce_sum(dist.probs)
            cn.off_policy_distribution = dist


def perform_action(node: StateNode):
    p, i, v, cn, nn = node.action_schema.sample()
    if nn is None:
        direkt_reward, nn = node.act(v)
        cn.add(p[-1], i[-1], v[-1], direkt_reward, nn)
    return p, i, v, cn, nn


def rollout_internal(root: StateNode, exploration_policy_function: Callable) -> StateNode or Leaf:
    node = root
    while isinstance(node, Leaf) is False and node.action_schema is not None:
        p, i, v, cn, nn = perform_action(node)
        node = nn
    assert isinstance(node, StateNode) or isinstance(node, Leaf)
    return node


def rollout(state_node: StateNode, iterations: int, batch_size: int, policy_estimator: PolicyEstimator,
            exploration_policy_function: Callable):
    for _ in range(iterations):
        batch = []
        for _ in range(batch_size):
            rollout_leaf = rollout_internal(state_node, exploration_policy_function)
            if isinstance(rollout_leaf, StateNode):
                batch.append(rollout_leaf)
        apply_estimator(batch, policy_estimator)
    return state_node


def apply_estimator(batch: List[StateNode], policy_estimator: PolicyEstimator):
    if len(batch) == 0:
        return
    info_set_batch = [batch_item.info_set for batch_item in batch]
    estimated_policies = policy_estimator(info_set_batch)
    for id in range(len(batch)):
        batch_item = batch[id]
        batch_item.action_schema = estimated_policies[id]


class ReplayBuffer:
    def __init__(self, capacity: int, rank: int):
        self.capacity = capacity
        self.buffer = []

        self.rank = rank
        if self.rank == 0:
            self.start_message_watcher()

    def add(self, info_sets: List[InfoSet], action_schema_targets: List[ActionSchema]):
        if self.rank == 0:
            tf.print("recived")
            for info_set, action_schema_target in zip(info_sets, action_schema_targets):
                if len(self.buffer) > self.capacity:
                    self.buffer.pop(0)
                self.buffer.append(
                    (info_set, action_schema_target)
                )
        else:
            tf.print("send")
            MPI.COMM_WORLD.send((info_sets, action_schema_targets), dest=0, tag=0)

    def fetch(self, batch_size: int) -> Tuple[List[InfoSet] or None, List[ActionSchema] or None]:
        if len(self.buffer) < batch_size:
            return None, None
        samples = random.sample(self.buffer, batch_size)
        info_sets, action_schema_targets = zip(*samples)
        return info_sets, action_schema_targets

    def watch_messages(self):
        while True:
            info_sets, action_schema_targets = MPI.COMM_WORLD.recv(tag=0)
            self.add(info_sets, action_schema_targets)

    def start_message_watcher(self):
        thread = Thread(target=self.watch_messages)
        thread.start()
        return thread


def gather_episodes(game: Game, discount: float, replay_buffer: ReplayBuffer, policy_estimator: PolicyEstimator,
                    exploration_policy_function: Callable, samples: int, batch_size: int,
                    max_depth: int, game_config: dict):
    while True:
        root = game.start(game_config)
        trajectory = [root]
        while True:
            current_node = trajectory[-1]
            assert isinstance(current_node, StateNode)

            # Now estimate the current_node
            if current_node.action_schema is None:
                apply_estimator([current_node], policy_estimator)

            # expand the tree under the current node if max_depth not reached
            if len(trajectory) > max_depth:
                break
            # Introduce a secondary policy for exploration
            exploration_policy_function(current_node.action_schema)
            rollout(current_node, samples, batch_size, policy_estimator, exploration_policy_function)
            # REMOVE THE SECONDARY DISTRIBUTIONS!
            for cn in current_node.action_schema.chance_nodes:
                assert isinstance(cn, ChanceNode)
                cn.off_policy_distribution = None
            # pick the next action from unchanged distribution
            _, _, _, _, next_node = perform_action(current_node)
            # append it to the trajectory if not a Leaf
            if isinstance(next_node, Leaf):
                break
            trajectory.append(next_node)

        # now we compute the value for each state from the bottom
        for node in reversed(trajectory):
            node.compute_payoff(discount)

        # now optimize the policy in each node of the trajectory:
        for node in reversed(trajectory):
            node.action_schema.optimize(node.state.current_player, discount)

        replay_buffer.add([node.info_set for node in trajectory], [node.action_schema for node in trajectory])


def train_estimator(replay_buffer: ReplayBuffer, policy_estimator: PolicyEstimator, batch_size: int):
    i = 1
    while True:
        info_sets, action_schema_targets = replay_buffer.fetch(batch_size)
        if info_sets is None:
            i = 1
            sleep(10)
        policy_estimator.train(info_sets, action_schema_targets)

        if i % 1 == 0:

