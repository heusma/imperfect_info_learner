from abc import abstractmethod
from typing import Tuple

import tensorflow as tf


class State:
    pass


class Leaf:
    pass


class ActionSchema:
    """
    :returns The probability of the sampled action + The action_description as a tensor.
    """

    @abstractmethod
    def sample(self) -> Tuple[float, tf.Tensor]:
        pass

    @abstractmethod
    def prob(self, action: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def log_prob(self, action: tf.Tensor) -> tf.Tensor:
        pass


class InfoSet:
    def __init__(self, current_player: int):
        self.current_player = current_player

    """
    :returns A sampleable ActionSchema for the current InfoSet which permits only valid actions for this info_set.
    """

    @abstractmethod
    def get_action_schema(self) -> ActionSchema:
        pass


class Game:
    """
    :param state - The current state.
    :param info_set - The current info_set. Combination of all observations until now.
    A masked representation of the current state.
    :param action - An action_description as a tensor. Most likely sampled from an action schema.

    :returns The next state + The next info_set [The old info_set updated by the new observation]
    + The direkt reward for this (action, transition) pair.
    """

    @staticmethod
    @abstractmethod
    def act(state: State, info_set: InfoSet, action: tf.Tensor) -> Tuple[State or Leaf, InfoSet or Leaf, float]:
        pass

    @staticmethod
    @abstractmethod
    def get_root() -> Tuple[State, InfoSet]:
        pass

    @staticmethod
    @abstractmethod
    def test_performance(estimator: any):
        pass
