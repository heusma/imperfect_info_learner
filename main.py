import os

from Games.GridWorldContinuous import GridWorldContinuous, GridWorldContinuousEstimator, \
      grid_world_continuous_exploration_function
from Games.financial_model.Games.stock_world import StockWorld, stock_world_exploration_function, StockWorldEstimator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mpi4py import MPI

from EvaluationTool import train, identity_exploration_function
from Games.GridWorld import GridWorld, GridWorldEstimator, grid_world_exploration_function

work_dir = "."
#work_dir = os.environ["WORK"]

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## config
game = GridWorld
discount = 0.9
max_steps = 40
horizon = 4
num_trajectory_samples = 1
max_targets_per_trajectory = 20
num_additional_unroll_samples_per_visited_state = 2

estimator = GridWorldEstimator()
batch_size = 40

exploration_function = grid_world_exploration_function
p = 10
c = 1
r = c

test_interval = 100
checkpoint_interval = 100
checkpoint_path = work_dir + "/checkpoints/checkpoint_stock_world.json"
##

estimator.load(checkpoint_path, blocking=False)

train(game, discount, max_steps, horizon, num_trajectory_samples, max_targets_per_trajectory,
      num_additional_unroll_samples_per_visited_state, estimator,
      batch_size, exploration_function, p, c, r, rank, size, test_interval, checkpoint_interval, checkpoint_path)

MPI.Finalize()
