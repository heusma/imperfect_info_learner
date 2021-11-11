import copy
import random
import tensorflow as tf

from EvaluationTool import rollout, TabularPolicyEstimator, train
from Game import Leaf
from Games.GridWorld import GridWorld, GridWorldPolicyEstimator
from Games.KuhnPoker import KuhnPoker, Cards

tf.random.set_seed(0)
random.seed(0)

root = KuhnPoker.start(dict())
root.value = 0
root.action_schema = root.info_set.get_action_schema()

pe = GridWorldPolicyEstimator()

train(GridWorld, 0.99, pe, samples=20, batch_size=1, max_depth=1, iterations=200000, game_config=dict())

print(root)
