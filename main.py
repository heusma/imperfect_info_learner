# mpiexec -n 3 python main.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from mpi4py import MPI

from EvaluationTool import train, identity_exploration_function

#from Games.GridWorld import GridWorld, grid_world_exploration_function, GridWorldEstimator

from Games.financial_model.Games.StockWorld import StockWorld, StockWorldEstimator, stock_world_exploration_function

work_dir = "."
# work_dir = os.environ["WORK"]

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

## config
game = StockWorld
discount = 0.97
horizon = 12
num_trajectory_samples = 4
max_targets_per_trajectory = 5
max_steps = max(3 * 12, max_targets_per_trajectory + horizon)
num_additional_unroll_samples_per_visited_state = 3

estimator = StockWorldEstimator()
batch_size = 40

exploration_function = stock_world_exploration_function
p = 10
c = 1
r = c

test_interval = 20
checkpoint_interval = 500
checkpoint_path = work_dir + "/checkpoints_stock/checkpoint.json"
##

# sync estimators before training
if rank == 0:
    estimator.load(checkpoint_path, blocking=False)
    estimator.save(checkpoint_path)
else:
    estimator.load(checkpoint_path, blocking=True)

train(game, discount, max_steps, horizon, num_trajectory_samples, max_targets_per_trajectory,
      num_additional_unroll_samples_per_visited_state, estimator,
      batch_size, exploration_function, p, c, r, rank, size, test_interval, checkpoint_interval, checkpoint_path)

MPI.Finalize()
