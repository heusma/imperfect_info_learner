import random
import time
from abc import abstractmethod
from typing import Tuple, Callable, List, Type

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from MDP import State, InfoSet, Leaf, Game, ActionSchema

QValueTarget = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
VTraceTarget = Tuple[InfoSet, float, tf.Tensor, List[QValueTarget]]
VTraceGradients = Tuple[List[tf.Tensor], List[tf.Tensor]]


class Estimator:
    @abstractmethod
    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        pass

    @abstractmethod
    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        pass

    @abstractmethod
    def apply_gradients(self, grads: VTraceGradients):
        pass

    @abstractmethod
    def save(self, checkpoint_location: str) -> None:
        pass

    @abstractmethod
    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        pass


def identity_exploration_function(action_schema: ActionSchema) -> ActionSchema:
    return action_schema


# [state, info_set, value_estimate, action_schema_estimate, off_policy_action_schema,
#                 on_policy_probability, off_policy_probability, action, direct_reward]
TrajectoryElement = Tuple[State, InfoSet, tf.Tensor, ActionSchema, ActionSchema, float, float, tf.Tensor, float]
Trajectory = List[TrajectoryElement or Leaf]


def take_action(game: Type[Game], state: State, info_set: InfoSet, on_policy_action_schema: ActionSchema,
                off_policy_action_schema: ActionSchema):
    off_policy_probability, action = off_policy_action_schema.sample()
    on_policy_probability = on_policy_action_schema.prob(action)
    state, info_set, direct_reward = game.act(state, info_set, action)
    return on_policy_probability, off_policy_probability, action, direct_reward, state, info_set


def create_trajectory(game: Type[Game], state: State, info_set: InfoSet, estimator: Estimator,
                      exploration_function: Callable,
                      max_steps: int):
    trajectory: Trajectory = []
    while len(trajectory) < max_steps:
        value_estimate, action_schema_estimate = estimator.evaluate([info_set])[0]
        off_policy_action_schema = exploration_function(action_schema_estimate)
        on_policy_probability, off_policy_probability, action, direct_reward, new_state, new_info_set = \
            take_action(game, state, info_set, action_schema_estimate, off_policy_action_schema)
        trajectory.append(
            (
                state, info_set, value_estimate, action_schema_estimate, off_policy_action_schema,
                on_policy_probability, off_policy_probability, action, direct_reward
            )
        )
        state, info_set = new_state, new_info_set

        if isinstance(state, Leaf):
            trajectory.append(Leaf())
            break

    return trajectory


state_info_pair_information = Tuple[State, InfoSet, tf.Tensor, ActionSchema, ActionSchema] or None


# SPEED UP THIS PART!!!!
def gather_unrolls(game: Type[Game], start: TrajectoryElement, num_samples: int, batch_size: int, estimator: Estimator,
                   exploration_function: Callable,
                   max_steps: int):
    state, info_set, value_estimate, on_policy_action_schema, off_policy_action_schema, on_policy_probability, off_policy_probability, action, direct_reward = \
        start

    unrolls: List[Trajectory] = []
    current_state_info_pairs: List[state_info_pair_information] = [(state, info_set, value_estimate,
                                                                    on_policy_action_schema,
                                                                    off_policy_action_schema)] * num_samples
    for _ in range(num_samples):
        unrolls.append([])

    k = 0
    while True:
        evaluation_worklist: List[Tuple[int, State, InfoSet]] = []
        # sample one action for each rollout path
        for j in range(num_samples):
            current_state_info_pair = current_state_info_pairs[j]
            if current_state_info_pair is None:
                continue
            state, info_set, value_estimate, on_policy_action_schema, off_policy_action_schema = \
                current_state_info_pairs[j]
            on_policy_probability, off_policy_probability, action, direct_reward, new_state, new_info_set = \
                take_action(game, state, info_set, on_policy_action_schema, off_policy_action_schema)
            unrolls[j].append(
                (
                    state, info_set, value_estimate, on_policy_action_schema, off_policy_action_schema,
                    on_policy_probability, off_policy_probability, action, direct_reward
                )
            )
            # gather new_info_sets in the evaluation_worklist for batch processing.
            if isinstance(new_info_set, Leaf):
                current_state_info_pairs[j] = None
                unrolls[j].append(Leaf())
            else:
                evaluation_worklist.append((j, new_state, new_info_set))

        k += 1
        if k == max_steps:
            break

        # group worklist into batches and compute estimates
        worklist_pointer = 0
        while worklist_pointer < len(evaluation_worklist):
            batch = evaluation_worklist[worklist_pointer:worklist_pointer + batch_size]
            worklist_pointer += batch_size
            id_batch, state_batch, info_set_batch = zip(*batch)
            value_estimates, on_policy_action_schemas = zip(*estimator.evaluate(info_set_batch))
            off_policy_action_schemas = [exploration_function(action_schema) for action_schema in
                                         on_policy_action_schemas]
            for z in range(len(id_batch)):
                id = id_batch[z]
                current_state_info_pairs[id] = (
                    state_batch[z], info_set_batch[z], value_estimates[z], on_policy_action_schemas[z],
                    off_policy_action_schemas[z]
                )

    return unrolls


def trajectory_to_vtrace_targets(trajectory: Trajectory, discount: float, p: float, c: float):
    # first compute v_s for each state
    value_targets: List[tf.Tensor] = [tf.zeros(shape=())] * (len(trajectory) - 1)
    q_value_targets = [tf.zeros(shape=())] * (len(trajectory) - 1)

    te = trajectory[-1]
    if isinstance(te, Leaf):
        v_s_nn = 0.0
        value_estimate_nn = 0.0
    else:
        _, _, v_s_nn, _, _, _, _, _, _ = trajectory[-1]
        value_estimate_nn = v_s_nn

    for i in reversed(range(len(trajectory) - 1)):
        _, _, value_estimate_cn, _, _, on_policy_probability, off_policy_probability, _, direct_reward = trajectory[i]

        importance_weight = on_policy_probability / off_policy_probability
        p_t = tf.minimum(p, importance_weight)
        c_t = tf.minimum(c, importance_weight)

        td = p_t * (direct_reward + discount * value_estimate_nn - value_estimate_cn)
        v_s = value_estimate_cn + td + discount * c_t * (v_s_nn - value_estimate_nn)

        value_targets[i] = v_s

        q_s = direct_reward + discount * v_s_nn
        q_value_targets[i] = q_s

        v_s_nn = v_s
        value_estimate_nn = value_estimate_cn

    return zip(value_targets, q_value_targets)


def valid_q_target(importance: tf.Tensor, q_value: tf.Tensor):
    tmp = tf.concat([importance, q_value], axis=0)
    return not (tf.reduce_any(tf.math.is_nan(tmp)) or tf.reduce_any(tf.math.is_inf(tmp)))


def valid_v_trace_target(reach_importance_weight, mean_value_target, q_value_targets):
    if len(q_value_targets) == 0:
        return False
    tmp = tf.concat([reach_importance_weight, tf.squeeze(mean_value_target)], axis=0)
    return not (tf.reduce_any(tf.math.is_nan(tmp)) or tf.reduce_any(tf.math.is_inf(tmp)))


def analyse(game: Type[Game], trajectory: Trajectory,
            max_targets_per_trajectory: int,
            num_samples: int,
            batch_size: int,
            estimator: Estimator,
            exploration_function: Callable,
            max_steps: int,
            discount: float,
            r: float,
            p: float,
            c: float) -> List[VTraceTarget]:
    on_policy_reach = 1.0
    off_policy_reach = 1.0
    trajectory_v_trace_targets = []

    chosen_elements = range(len(trajectory) - 1)
    if len(chosen_elements) > max_targets_per_trajectory:
        chosen_elements = random.sample(chosen_elements, max_targets_per_trajectory)
    for i in chosen_elements:

        trajectory_element = trajectory[i]
        _, info_set, _, _, _, on_policy_probability, off_policy_probability, _, _ = trajectory_element

        on_policy_reach *= on_policy_probability
        off_policy_reach *= off_policy_probability
        reach_importance_weight = tf.minimum(r, on_policy_reach / off_policy_reach)

        unrolls = gather_unrolls(game, trajectory_element, num_samples, batch_size, estimator, exploration_function,
                                 max_steps)

        if len(trajectory) - i >= max_steps:
            unroll_from_outer_trajectory = []
            for u in range(max_steps):
                unroll_from_outer_trajectory.append(
                    trajectory[i + u]
                )
            if len(trajectory) > i + max_steps and isinstance(trajectory[i + max_steps], Leaf):
                unroll_from_outer_trajectory.append(Leaf())
            unrolls.append(unroll_from_outer_trajectory)

        v_trace_targets = [
            list(trajectory_to_vtrace_targets(unroll, discount, p, c))[0] for unroll in unrolls
        ]

        mean_value_target = tf.zeros(shape=())
        q_value_targets = []
        for j in range(num_samples):
            value_target, q_value_target = v_trace_targets[j]
            mean_value_target += value_target
            _, _, _, _, _, on_policy_probability, off_policy_probability, action, _ = unrolls[j][0]
            imp = tf.minimum(p, on_policy_probability / off_policy_probability)
            if valid_q_target(imp, q_value_target[info_set.current_player]):
                q_value_targets.append(
                    (
                        action,
                        imp,
                        q_value_target[info_set.current_player],
                    )
                )
        mean_value_target /= num_samples

        if valid_v_trace_target(reach_importance_weight, mean_value_target, q_value_targets):
            trajectory_v_trace_targets.append(
                (info_set, reach_importance_weight, mean_value_target, q_value_targets)
            )

    return trajectory_v_trace_targets


def compute_gradient_mean(grads: List[VTraceGradients]) -> VTraceGradients:
    result = [None, None]
    for grad_pair in grads:
        for i in range(2):
            if result[i] is None:
                result[i] = grad_pair[i]
            else:
                result[i] = [
                    result[i][z] + grad_pair[i][z] for z in range(len(result[i]))
                ]

    for i in range(2):
        for j in range(len(result[i])):
            result[i][j] /= len(grads)

    v_grads, p_grads = result
    return v_grads, p_grads


def get_gradient_from_targets(targets: List[VTraceTarget], batch_size: int, estimator: Estimator) -> VTraceGradients:
    worklist_pointer = 0
    gradients = []
    while worklist_pointer < len(targets):
        batch = targets[worklist_pointer:worklist_pointer + batch_size]
        worklist_pointer += batch_size
        local_grads = estimator.compute_gradients(batch)
        gradients.append(local_grads)

    return compute_gradient_mean(gradients)


def flatten_gradients(gradients: VTraceGradients):
    shapes = ([], [])
    grad_values = []
    for i in range(2):
        grads = gradients[i]
        for g in grads:
            shapes[i].append(tf.shape(g))
            grad_values.append(
                tf.reshape(g, shape=(-1,))
            )

    grad_values = tf.concat(grad_values, axis=0)

    return grad_values, shapes


def reshape_gradient_values(gradient_values: tf.Tensor, shapes: Tuple[List[tf.TensorShape], List[tf.TensorShape]]):
    gradients = ([], [])
    pointer = 0
    for i in range(2):
        grads = gradients[i]
        l_shapes = shapes[i]
        for j in range(len(l_shapes)):
            shape = l_shapes[j]
            element_count = tf.reduce_prod(shape)
            grads.append(
                tf.reshape(
                    gradient_values[pointer:pointer + element_count],
                    shape
                )
            )
            pointer += element_count

    return gradients


def split_equal(a: np.array, n: int) -> List[np.array]:
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


def ring_all_reduce(gradient_values: tf.Tensor, rank: int, size: int) -> tf.Tensor:
    gradient_values_numpy = gradient_values.numpy()
    n_d_type = MPI.FLOAT

    destination = (rank + 1) % size
    source = (rank - 1) % size

    partitions = split_equal(gradient_values_numpy, size)

    # first_ring
    i = rank
    for _ in range(size - 1):
        prev_rank = (i - 1) % size

        send_request = MPI.COMM_WORLD.Isend([partitions[i], n_d_type], dest=destination)

        reduced_partition = np.empty(partitions[prev_rank].size, dtype='f')
        MPI.COMM_WORLD.Recv([reduced_partition, MPI.FLOAT], source=source)

        send_request.wait()

        partitions[prev_rank] = partitions[prev_rank] + reduced_partition

        i = prev_rank

    partitions[i] = partitions[i] / size

    # second_ring
    for _ in range(size - 1):
        prev_rank = (i - 1) % size

        send_request = MPI.COMM_WORLD.Isend([partitions[i], n_d_type], dest=destination)

        reduced_partition = np.empty(partitions[prev_rank].size, dtype='f')
        MPI.COMM_WORLD.Recv([reduced_partition, MPI.FLOAT], source=source)

        send_request.wait()

        partitions[prev_rank] = reduced_partition

        i = prev_rank

    result = tf.convert_to_tensor(
        np.concatenate(partitions, axis=0),
        dtype=tf.float32,
    )

    return result


def sync_gradients(rank: int, size: int, gradients: VTraceGradients) -> VTraceGradients:
    if size == 1:
        return gradients

    gradient_values, shapes = flatten_gradients(gradients)

    allreduce_gradient_values = ring_all_reduce(gradient_values, rank, size)

    gradients = reshape_gradient_values(allreduce_gradient_values, shapes)

    return gradients


def filter_invalid_gradients(gradients: VTraceGradients) -> VTraceGradients:
    for i in range(2):
        contains_nan = False
        grads = gradients[i]
        for j in range(len(grads)):
            g = grads[j]
            if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                contains_nan = True
                grads[j] = tf.zeros_like(grads[j])
        if contains_nan:
            tf.print("filtered nan values!")
    return gradients


def train(game: Type[Game],
          discount: float,
          max_steps: int,
          horizon: int,
          num_trajectory_samples: int,
          max_targets_per_trajectory: int,
          num_unroll_samples_per_visited_state: int,
          estimator: Estimator,
          batch_size: int,
          exploration_function: Callable,
          p: float,
          c: float,
          r: float,
          rank: int,
          size: int,
          test_interval: int,
          checkpoint_interval: int,
          checkpoint_path: str,
          ):
    i = 0
    while True:
        start_t = time.time()
        gradients = []
        for _ in range(num_trajectory_samples):
            root_s, root_i = game.get_root()

            trajectory = create_trajectory(game=game, state=root_s, info_set=root_i, estimator=estimator,
                                           exploration_function=exploration_function,
                                           max_steps=max_steps)

            targets = analyse(game, trajectory, max_targets_per_trajectory,
                              num_samples=num_unroll_samples_per_visited_state, batch_size=batch_size,
                              estimator=estimator, exploration_function=exploration_function, max_steps=horizon,
                              discount=discount, p=p, c=c, r=r)

            if len(targets) == 0:
                continue

            local_gradients = get_gradient_from_targets(targets, batch_size=batch_size, estimator=estimator)

            local_gradients = filter_invalid_gradients(local_gradients)

            gradients.append(local_gradients)

        if len(gradients) == 0:
            tf.print("loop again caused by missing gradients.")
            continue

        gradients = compute_gradient_mean(gradients)

        gradients = sync_gradients(rank, size, gradients)

        estimator.apply_gradients(gradients)

        i += 1
        if rank == 0:
            if i % test_interval == 0:
                tf.print(f'{int(i / test_interval)} test in this session.')
                game.test_performance(estimator)
            if i % checkpoint_interval == 0:
                tf.print(f'{int(i / checkpoint_interval)} savepoint in this session.')
                estimator.save(checkpoint_path)

            end_t = time.time()
            tf.print(f'one iteration took: {end_t - start_t}')
