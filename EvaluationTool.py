import copy
from abc import abstractmethod
from typing import List

import tensorflow as tf

from Game import StateNode, Leaf, ActionSchema, InfoSet, Game
from Games.KuhnPoker import Cards


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
                    if t_cn.payoff_estimate is None:
                        t_cn.payoff_estimate = s_cn.payoff_estimate
                    else:
                        distance = s_cn.payoff_estimate - t_cn.payoff_estimate
                        t_cn.payoff_estimate += self.learning_rate * distance
                    t_cn_params = t_cn.distribution.get_parameters()
                    s_cn_params = s_cn.distribution.get_parameters()
                    t_cn_params += self.learning_rate * (s_cn_params - t_cn_params)
                    t_cn.distribution.set_parameters(t_cn_params)
            else:
                policy_target_copy = copy.deepcopy(policy_target)
                for cn in policy_target_copy.chance_nodes:
                    cn.children = dict()
                self.table[key] = policy_target_copy


def perform_action(node: StateNode):
    p, v, cn, nn = node.action_schema.sample()
    if nn is None:
        direkt_reward, nn = node.act(v)
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


def train(game: Game, discount: float, policy_estimator: PolicyEstimator, samples: int, batch_size: int,
          max_depth: int, iterations: int, game_config: dict):
    for i in range(iterations):
        root = game.start(game_config)
        trajectory = [root]
        while True:
            current_node = trajectory[-1]
            assert isinstance(current_node, StateNode)
            if current_node.action_schema is None:
                apply_estimator([current_node], policy_estimator)
            # expand the tree under the current node if max_depth not reached
            if len(trajectory) > max_depth:
                break
            rollout(current_node, samples, batch_size, policy_estimator)
            # pick the next action from unchanged distribution
            _, _, _, next_node = perform_action(current_node)
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

        # this is a placeholder where we train the estimator
        policy_estimator.train([node.info_set for node in trajectory], [node.action_schema for node in trajectory])

        # this is a grid world specific placeholder to test the estimaters quality
        if i % 1000 == 0:
            game.test_policy(policy_estimator)
            game.show_tile_values(policy_estimator)
