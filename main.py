import copy
import random
from time import sleep
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from mpi4py import MPI

from EvaluationTool import gather_episodes, ReplayBuffer, train_estimator, \
    TabularPolicyEstimator
from Games.ConnectFour import ConnectFour, ConnectFourNetwork, ConnectFourPolicyEstimator
from Games.GridWorld import GridWorld, GridWorldPolicyEstimator
from Games.KuhnPoker import KuhnPoker

work_dir = "."
#work_dir = os.environ["WORK"]

## config
discount = 0.997

rollout_max_depth = 20
rollout_sample_passes_per_state = 10
batch_size_rollout = 10

buffer_capacity = 1000
min_buffer_training = int(1.0 * buffer_capacity)
batch_size_training = 100

checkpoint_interval = 2000
checkpoint_location = work_dir + "/checkpoints/checkpoint_connect_4_net.json"
##

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

game_type = ConnectFour

rb = ReplayBuffer(capacity=buffer_capacity, rank=rank)
pe = ConnectFourPolicyEstimator()
if rank == 0:
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
