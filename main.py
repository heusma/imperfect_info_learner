import copy
import random
from time import sleep

import tensorflow as tf
from mpi4py import MPI

from EvaluationTool import gather_episodes, ReplayBuffer, train_estimator, \
    TabularPolicyEstimator
from Games.GridWorld import GridWorld, GridWorldPolicyEstimator
from Games.KuhnPoker import KuhnPoker

## config
discount = 0.98

rollout_max_depth = 1
rollout_sample_passes_per_state = 5
batch_size_rollout = 20

buffer_capacity = 2000
min_buffer_training = int(0.5 * buffer_capacity)
batch_size_training = 80

checkpoint_interval = 2000
checkpoint_location = "./checkpoints/checkpoint_grid_world_10_net.json"
##

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

game_type = GridWorld

rb = ReplayBuffer(capacity=buffer_capacity, rank=rank)
pe = GridWorldPolicyEstimator()
if rank == 0:
    pe.load(checkpoint_location, blocking=False)
    pe.save(checkpoint_location)
    train_estimator(
        rb,
        pe,
        batch_size_training,
        min_buffer_training,
        checkpoint_interval,
        checkpoint_location,
        game_type,
    )
else:
    pe.load(checkpoint_location)
    gather_episodes(
        game_type,
        discount,
        rb,
        pe,
        checkpoint_location,
        samples=rollout_sample_passes_per_state,
        batch_size=batch_size_rollout,
        max_depth=rollout_max_depth,
        game_config=dict()
    )

MPI.Finalize()
