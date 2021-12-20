import random
from abc import abstractmethod
from typing import List, Tuple
import tensorflow as tf


class Distribution:
    @abstractmethod
    def sample(self) -> Tuple[float, float or int]:
        pass

    @abstractmethod
    def get_probability(self, value: float or int) -> float:
        pass

    @abstractmethod
    def optimize(self, values: List[int or float], probabilities: List[float]):
        pass

    @abstractmethod
    def get_parameters(self) -> tf.Tensor:
        pass

    @abstractmethod
    def set_parameters(self, paramters: tf.Tensor):
        pass

    # Returns a gready distribution.
    @abstractmethod
    def gready(self) -> any:
        pass


class DiscreteDistribution(Distribution):
    def __init__(self, size: int):
        self.assign(tf.constant(1 / size, shape=(size,)))

    def assign(self, probs: List[float]):
        self.probs = probs
        self.logits = tf.math.log(self.probs)

    def sample(self) -> Tuple[float, float or int]:
        assert self.logits is not None
        sample_id = tf.squeeze(tf.random.categorical(tf.expand_dims(self.logits, axis=0), num_samples=1)).numpy()
        probability = self.probs[sample_id]
        return probability, sample_id

    def get_probability(self, value: float or int) -> float:
        return self.probs[value]

    def get_parameters(self) -> List[float]:
        return self.probs

    def set_parameters(self, paramters: List[float]):
        self.assign(paramters)

    def gready(self) -> Distribution:
        self_size = self.logits.shape[0]
        max_id = tf.argmax(self.probs)
        gready_probs = tf.one_hot(max_id, depth=self_size)
        res = DiscreteDistribution(self_size)
        res.assign(gready_probs)
        return res


    def optimize(self, values: List[int or float], probabilities: List[float]):
        probs = [0.0] * self.probs.shape[0]
        for v, p in zip(values, probabilities):
            probs[v] = p
        probs = tf.stack(probs)
        self.assign(probs)


class ContinuesDistribution(Distribution):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

        self.loc = (max - min) / 2
        self.scale = 1.0

    def assign(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def sample(self) -> Tuple[float, float or int]:
        sample_value = tf.random.normal(shape=(), mean=self.loc, stddev=self.scale)
        sample_value = tf.clip_by_value(sample_value, self.min, self.max)
        return 0.0, sample_value


class Node:
    """
    Returns an upstream payoff which will be passed to the parent node
    and a full list of payoffs found in this structure.
    """

    @abstractmethod
    def compute_payoff(self, discount) -> List[float]:
        pass


class Leaf(Node):
    def compute_payoff(self, discount):
        return 0.0


class ChanceNode(Node):
    def __init__(self, distribution: Distribution):
        self.on_policy_distribution = distribution
        self.children = dict()

        self.payoff_estimate: List[float] or None = None
        self.computed_payoff_estimate: List[float] or None = None

        self.intermediate_action_schema_node: bool = False

    def sample(self, ghost_mode: bool = False):
        if ghost_mode is False:
            self.computed_payoff_estimate = None

        probability, value = self.on_policy_distribution.sample()

        next_node = None
        # Is this value known?
        if value in self.children:
            look_up_probability, _, _, next_node = self.children[value]
            assert probability == look_up_probability
            # Increase visits by one.
            if ghost_mode is False:
                self.children[value][1] += 1
        return probability, value, next_node

    def add(self, probability: float, value: int or float, direkt_reward: float, node: Node):
        self.children[value] = [probability, 1, direkt_reward, node]

    def gready(self) -> any:  # -> ChanceNode
        return ChanceNode(distribution=self.on_policy_distribution.gready())

    def compute_payoff(self, discount):
        if self.computed_payoff_estimate is not None:
            return self.computed_payoff_estimate
        if len(self.children) == 0:
            self.computed_payoff_estimate = self.payoff_estimate
            return self.payoff_estimate
        if self.intermediate_action_schema_node is True:
            discount = 1.0
        value = 0
        sum_visits = 0
        for probability, visits, direkt_reward, node in self.children.values():
            upstream_payoff = node.compute_payoff(discount)
            action_value = direkt_reward + discount * upstream_payoff
            value += visits * action_value
            sum_visits += visits
        if sum_visits == 0:
            payoff = tf.zeros(shape=(1,))
        else:
            payoff = value / sum_visits
        self.computed_payoff_estimate = payoff
        return payoff

    def optimize(self, current_player: int, discount: float):
        assert self.computed_payoff_estimate is not None
        if self.intermediate_action_schema_node is True:
            discount = 1.0

        node_value = self.computed_payoff_estimate
        current_player_node_value = node_value[current_player]

        advantages = [0.0] * len(self.children)
        for value, index in zip(self.children.keys(), range(len(self.children))):
            _, visits, dr, nn = self.children[value]
            assert isinstance(nn, Node)
            # remember intermediate action schema chance nodes will set their discount to 1.0 so make sure
            # that all action schema leafs were allready computed with the right discount.
            upstream_payoff = nn.compute_payoff(discount)
            action_value_computed = dr + discount * upstream_payoff

            current_player_action_value = action_value_computed[current_player]
            advantage = current_player_action_value - current_player_node_value

            if isinstance(nn, StateNode):
                estimated_upstream_payoff = nn.action_schema.root_node().payoff_estimate
                current_player_action_value_estimate = estimated_upstream_payoff[current_player]
                novelty = tf.abs(nn.compute_payoff(discount)[current_player] - current_player_action_value_estimate)
            else:
                novelty = 0.0

            advantages[index] = advantage + novelty

        positive_advantages = tf.maximum(advantages, 0.0)

        if positive_advantages.shape[0] > 0:
            positive_advantages_sum = tf.reduce_sum(positive_advantages)
            if positive_advantages_sum == 0:
                target_policy = tf.constant(1 / positive_advantages.shape[0], shape=positive_advantages.shape)
            else:
                target_policy = positive_advantages / positive_advantages_sum
            target_policy = tf.cast(target_policy, dtype=tf.float32)

            self.on_policy_distribution.optimize(list(self.children.keys()), target_policy)


class ActionSchema(Node):
    def __init__(self, chance_nodes: List[ChanceNode], root_node_index: int = 0):
        self.root_node_index = root_node_index
        self.chance_nodes = chance_nodes
        self.leafs = []

        self.find_leaf_nodes()

    def find_leaf_nodes(self):
        for i, cn in enumerate(self.chance_nodes):
            is_leaf = True
            for _, _, nn in cn.children.values():
                if nn in self.chance_nodes:
                    is_leaf = False
                    break
            if is_leaf:
                self.leafs.append(cn)
            else:
                cn.intermediate_action_schema_node = True

    def root_node(self):
        return self.chance_nodes[self.root_node_index]

    def gready(self) -> any:  # -> ActionSchema
        num_chance_nodes = len(self.chance_nodes)
        cns: List[ChanceNode or None] = [None] * num_chance_nodes
        for i in range(num_chance_nodes):
            cns[i] = self.chance_nodes[i].gready()
        return type(self)(
            cns, self.root_node_index,
        )

    def sample(self, ghost_mode: bool = False):
        return self.sample_internal(ghost_mode)

    """
    @:returns:
    0: List with probabilities for sapling each of the values.
    1: List of sample importance.
    1: List with sampled values
    2: The last evaluated ChanceNode
    3: The looked_up next node if known.
    """

    @abstractmethod
    def sample_internal(self, ghost_mode: bool = False) -> Tuple[List[float], List[float], List[float], ChanceNode, Node]:
        pass

    """
    If the discount is not 1.0 it is important to first compute the payoff for the leaves with the given discount.
    These nodes will save the computed value as an estimate.
    Then we can compute the value of all intermediate action schema nodes which will st there discount to 1.0
    automatically when computing their payoff.
    """

    def compute_payoff(self, discount):
        for cn in self.leafs:
            cn.compute_payoff(discount)

        for cn in self.chance_nodes:
            cn.compute_payoff(1.0)

        return self.root_node().compute_payoff(1.0)

    def optimize(self, current_player: int, discount: float):
        self.compute_payoff(discount)
        for cn in self.chance_nodes:
            cn.optimize(current_player, discount)


class InfoSet:
    @abstractmethod
    def get_action_schema(self) -> ActionSchema:
        pass


class State:
    def __init__(self, current_player: int):
        self.current_player = current_player

    @abstractmethod
    def to_info_set(self) -> InfoSet:
        pass


class StateNode(Node):
    def __init__(self, state: State):
        self.state = state
        self.info_set = state.to_info_set()
        self.action_schema: ActionSchema or None = None

    @abstractmethod
    def act(self, action: List[int or float]) -> Tuple[tf.Tensor, Node]:
        pass

    def compute_payoff(self, discount):
        return self.action_schema.compute_payoff(discount)


class Game:
    @staticmethod
    @abstractmethod
    def start(options: dict) -> StateNode:
        pass
