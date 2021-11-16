import copy
import random
import tensorflow as tf
from mpi4py import MPI

from EvaluationTool import default_exploration_policy_function, gather_episodes, ReplayBuffer, train_estimator
from Games.GridWorld import GridWorld, GridWorldPolicyEstimator
from Games.KuhnPoker import KuhnPoker

tf.random.set_seed(0)
random.seed(0)

## config
rollout_max_depth = 1
rollout_sample_passes_per_state = 4
batch_size_rollout = 10

batch_size_training = 40

##

#MPI.Init()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

root = KuhnPoker.start(dict())
root.value = 0
root.action_schema = root.info_set.get_action_schema()

rb = ReplayBuffer(capacity=1000, rank=rank)
pe = GridWorldPolicyEstimator()
if rank == 0:
    train_estimator(rb, pe, batch_size_training)
else:
    gather_episodes(GridWorld, 0.99, rb, pe, default_exploration_policy_function, samples=rollout_sample_passes_per_state, batch_size=batch_size_rollout, max_depth=rollout_max_depth, game_config=dict())

print(root)

MPI.Finalize()
