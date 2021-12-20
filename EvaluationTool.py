import copy
import os
import random
from abc import abstractmethod
from threading import Thread
from time import sleep
from typing import List, Callable, Tuple, Type

import jsonpickle
import tensorflow as tf
import numpy as np

from mpi4py import MPI

from Game import StateNode, Leaf, ActionSchema, InfoSet, Game, ChanceNode, DiscreteDistribution


class PolicyEstimator:
    @abstractmethod
    def __call__(self, info_sets: List[InfoSet], *args, **kwargs) -> List[ActionSchema]:
        pass

    @abstractmethod
    def train(self, info_sets: List[InfoSet], action_schema_targets: List[ActionSchema]):
        pass

    @abstractmethod
    def save(self, checkpoint_location: str) -> None:
        pass

    @abstractmethod
    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        pass


class DefaultPolicyEstimator(PolicyEstimator):
    def __call__(self, info_sets: List[InfoSet], *args, **kwargs) -> List[ActionSchema]:
        return [info_set.get_action_schema() for info_set in info_sets]

    def train(self, info_sets: List[InfoSet], action_schema_targets: List[ActionSchema]):
        pass

    def save(self, checkpoint_location: str) -> None:
        pass

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        pass


def to_key(info_set: InfoSet):
    return str(list(info_set.__dict__.values()))


class TabularPolicyEstimator(PolicyEstimator):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.table = dict()

        self.version: int = 0

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

    def save(self, checkpoint_location: str) -> None:
        json_object = jsonpickle.encode(self)
        data_location = os.path.dirname(checkpoint_location) + "/data/checkpoint.json"
        checkpoint_object = jsonpickle.encode({"version": random.randint(1, 10000000), "data_location": data_location})
        success = False
        while success is False:
            try:
                os.makedirs(os.path.dirname(data_location), exist_ok=True)
                with open(data_location, 'w') as f:
                    f.write(json_object)
                with open(checkpoint_location, 'w') as f:
                    f.write(checkpoint_object)
                success = True
                tf.print("saved")
            except IOError:
                tf.print("save error")
                sleep(2)

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        success = False
        while success is False:
            try:
                with open(checkpoint_location, 'r') as f:
                    json_object = f.read()
                    checkpoint = jsonpickle.decode(json_object)
                    version = checkpoint["version"]
                    if self.version != version:
                        with open(checkpoint["data_location"], 'r') as f2:
                            tf.print("loaded new version")
                            json_object = f2.read()
                            loaded_pe_instance = jsonpickle.decode(json_object)
                            self.learning_rate = loaded_pe_instance.learning_rate
                            self.table = loaded_pe_instance.table
                            self.version = version
                    success = True
            except IOError:
                tf.print("load error")
                sleep(2)
                if blocking is False:
                    success = True


def add_exploration_noise(node: StateNode):
    root_dirichlet_alpha = 0.3
    frac = 0.25

    for cn in node.action_schema.chance_nodes:
        assert isinstance(cn, ChanceNode)
        dist = cn.on_policy_distribution
        if isinstance(dist, DiscreteDistribution):
            noise = np.random.dirichlet([root_dirichlet_alpha] * dist.logits.shape[0])
            new_policy = tf.cast(dist.probs, dtype=tf.float32) * (1 - frac) + noise * frac
            dist.assign(new_policy)
        else:
            pass


def perform_action(node: StateNode, ghost_mode: bool = False):
    p, v, cn, nn = node.action_schema.sample(ghost_mode)
    if nn is None:
        direkt_reward, nn = node.act(v)
        if ghost_mode is False:
            cn.add(p[-1], v[-1], direkt_reward, nn)
    return p, v, cn, nn


def rollout_internal(root: StateNode) -> StateNode or Leaf:
    node = root
    while isinstance(node, Leaf) is False and node.action_schema is not None:
        p, v, cn, nn = perform_action(node)
        node = nn
    assert isinstance(node, StateNode) or isinstance(node, Leaf)
    return node


def rollout(state_node: StateNode, iterations: int, batch_size: int, policy_estimator: PolicyEstimator):
    for _ in range(iterations):
        batch = []
        for _ in range(batch_size):
            rollout_leaf = rollout_internal(state_node)
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
            for info_set, action_schema_target in zip(info_sets, action_schema_targets):
                if len(self.buffer) > self.capacity:
                    self.buffer.pop(0)
                self.buffer.append(
                    (info_set, action_schema_target)
                )
        else:
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


def gather_episodes(game: Type[Game], discount: float, replay_buffer: ReplayBuffer, policy_estimator: PolicyEstimator,
                    checkpoint_location: str,
                    samples: int, batch_size: int,
                    max_depth: int, game_config: dict):

    estimator: PolicyEstimator = DefaultPolicyEstimator()
    while True:

        if os.path.isfile(checkpoint_location):
            policy_estimator.load(checkpoint_location)
            estimator = policy_estimator

        root = game.start(game_config)
        current_node = root
        trajectory = []
        while True:
            assert isinstance(current_node, StateNode)

            # Now estimate the current_node
            if current_node.action_schema is None:
                apply_estimator([current_node], estimator)

            if len(trajectory) >= max_depth:
                break

            trajectory.append(current_node)

            # add noise to the estimated policy
            add_exploration_noise(current_node)
            # remove children from the current_node
            for cn in current_node.action_schema.chance_nodes:
                cn.children = dict()

            rollout(current_node, samples, batch_size, estimator)

            current_node.compute_payoff(discount)
            current_node.action_schema.optimize(current_node.state.current_player, discount)

            # compute payoff with optimized policy
            for cn in current_node.action_schema.chance_nodes:
                policy_sum = 0.0
                payoff_estiamte = 0.0
                for value in cn.children.keys():
                    probability, visits, direkt_reward, node = cn.children[value]
                    value_prob = cn.on_policy_distribution.get_probability(value)
                    payoff_estiamte += value_prob * (direkt_reward + discount * node.compute_payoff(discount))
                    policy_sum += value_prob
                payoff_estiamte /= policy_sum
                cn.computed_payoff_estimate = payoff_estiamte

            # remove children from the current_node
            for cn in current_node.action_schema.chance_nodes:
                cn.children = dict()

            # Sample from the optimized action schema but do not reset computed_values
            _, _, _, next_node = perform_action(current_node, ghost_mode=True)

            # append it to the trajectory if not a Leaf or max_depth reached
            if isinstance(next_node, Leaf):
                break
            current_node = next_node

        info_set_result = [node.info_set for node in trajectory]
        action_schema_result = [node.action_schema for node in trajectory]

        for action_schema in action_schema_result:
            for cn in action_schema.chance_nodes:
                if len(cn.children) > 0:
                    raise AssertionError("Noooo")
                cn.children = dict()

        replay_buffer.add(info_set_result, action_schema_result)


def train_estimator(replay_buffer: ReplayBuffer, policy_estimator: PolicyEstimator, batch_size: int,
                    min_buffer_training: int,
                    checkpoint_interval: int, checkpoint_location: str, game: Type[Game]):

    policy_estimator.load(checkpoint_location, blocking=False)

    i = 1
    # this is a grid world specific placeholder to test the estimaters quality
    tf.print(i)
    game.test_policy_baseline(policy_estimator)

    while True:
        i += 1

        info_sets, action_schema_targets = replay_buffer.fetch(batch_size)
        if info_sets is None or len(replay_buffer.buffer) < min_buffer_training:
            i = 1
            tf.print("sleep")
            tf.print(len(replay_buffer.buffer))
            sleep(10)
            continue
        policy_estimator.train(info_sets, action_schema_targets)

        if i % checkpoint_interval == 0:
            policy_estimator.save(checkpoint_location)
            # this is a grid world specific placeholder to test the estimaters quality
            tf.print(i)
            game.test_policy_baseline(policy_estimator)
            tf.print(len(replay_buffer.buffer))
