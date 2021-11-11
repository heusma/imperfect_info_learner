from abc import abstractmethod
from typing import List, Tuple
import tensorflow as tf


class Distribution:
    @abstractmethod
    def apply(self, distribution):
        pass

    @abstractmethod
    def sample(self) -> Tuple[float, float or int]:
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


class DiscreteDistribution(Distribution):
    def __init__(self, size: int):
        self.probs = tf.constant(1 / size, shape=(size,))

    def apply(self, distribution: Distribution):
        assert isinstance(distribution, DiscreteDistribution)
        self.assign(distribution.probs)

    def assign(self, probs: tf.Tensor):
        self.probs = probs

    def sample(self) -> Tuple[float, float or int]:
        sample_id = tf.squeeze(tf.random.categorical(tf.expand_dims(tf.math.log(self.probs), axis=0), num_samples=1)).numpy()
        probability = self.probs[sample_id]
        return probability, sample_id

    def get_parameters(self) -> tf.Tensor:
        return self.probs

    def set_parameters(self, paramters: tf.Tensor):
        self.assign(paramters)

    def optimize(self, values: List[int or float], probabilities: List[tf.Tensor]):
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
        self.distribution = distribution
        self.children = dict()

        self.payoff_estimate: List[float] or None = None

        self.intermediate_action_schema_node: bool = False

    def apply(self, chance_node):
        assert isinstance(chance_node, ChanceNode)
        self.distribution.apply(chance_node.distribution)

    def sample(self):
        self.payoff_estimate = None
        probability, value = self.distribution.sample()
        next_node = None
        # Is this value known?
        if value in self.children:
            look_up_probability, _, next_node = self.children[value]
            assert look_up_probability == probability
        return probability, value, next_node

    def add(self, probability: float, value: int or float, direkt_reward: float, node: Node):
        self.children[value] = [probability, direkt_reward, node]

    def compute_payoff(self, discount):
        if self.payoff_estimate is not None:
            return self.payoff_estimate
        if self.intermediate_action_schema_node is True:
            discount = 1.0
        value = 0
        probability_sum = 0
        for probability, direkt_reward, node in self.children.values():
            upstream_payoff = node.compute_payoff(discount)
            value += probability * (direkt_reward + discount * upstream_payoff)
            probability_sum += probability
        if probability_sum == 0:
            payoff = 0.0
        else:
            payoff = value / probability_sum
        self.payoff_estimate = payoff
        return payoff

    def optimize(self, current_player: int, discount: float):
        assert self.payoff_estimate is not None

        node_value = self.payoff_estimate
        current_player_node_value = node_value[current_player]

        advantages = [0.0] * len(self.children)
        for value, index in zip(self.children.keys(), range(len(self.children))):
            p, v, nn = self.children[value]
            assert isinstance(nn, Node)
            # remember intermediate action schema chance nodes will set their discount to 1.0 so make sure
            # that all action schema leafs were allready computed with the right discount.
            upstream_payoff = nn.compute_payoff(discount)
            action_value = tf.cast(v, dtype=tf.float32) + discount * upstream_payoff
            current_player_action_value = action_value[current_player]
            advantage = current_player_action_value - current_player_node_value
            advantages[index] = advantage

        positive_advantages = tf.maximum(advantages, 0.0)

        if positive_advantages.shape[0] > 0:
            positive_advantages_sum = tf.reduce_sum(positive_advantages)
            if positive_advantages_sum == 0:
                target_policy = tf.constant(1 / positive_advantages.shape[0], shape=positive_advantages.shape)
            else:
                target_policy = positive_advantages / positive_advantages_sum
            target_policy = tf.cast(target_policy, dtype=tf.float32)

            self.distribution.optimize(list(self.children.keys()), target_policy)


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

    def apply(self, action_schema):
        assert isinstance(action_schema, ActionSchema)
        assert len(action_schema.chance_nodes) == len(self.chance_nodes)
        for source_chance_node, target_chance_node in zip(action_schema.chance_nodes, self.chance_nodes):
            source_chance_node.apply(target_chance_node)

    def sample(self):
        for cn in self.chance_nodes:
            cn.payoff_estimate = None
        return self.sample_internal()

    """
    @:returns:
    0: List with probabilities for sapling each of the values.
    1: List with sampled values
    2: The last evaluated ChanceNode
    3: The looked_up next node if known.
    """
    @abstractmethod
    def sample_internal(self) -> Tuple[List[float], List[float], ChanceNode, Node]:
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
    def act(self, action: List[int or float]) -> Tuple[float, Node]:
        pass

    def compute_payoff(self, discount):
        return self.action_schema.compute_payoff(discount)


class Game:
    @staticmethod
    @abstractmethod
    def start(options: dict) -> StateNode:
        pass
