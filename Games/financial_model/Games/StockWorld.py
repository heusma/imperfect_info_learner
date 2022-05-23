import copy
import os

from Games.financial_model.archive.Archive import Archive
from Games.financial_model.archive.Normalizer import Normalizer
from Games.financial_model.archive.structures.Timeline import Timeline, TimelinePointer
from Games.financial_model.kraken.FinModPrep.stocks.datatypes.entities.Dividend import DividendType

import random
from datetime import datetime
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import tensorflow_addons as tfa

from Distributions.CategoricalDistribution import categorical_smoothing_function, Categorical
from EvaluationTool import Estimator, VTraceTarget, VTraceGradients, create_trajectory, identity_exploration_function
from MDP import Game, State, InfoSet, Leaf, ActionSchema

"""Config"""
work_dir = "."
# work_dir = os.environ["WORK"]

max_symbols_per_game = 3
max_trajectory_length = 4 * 12
max_horizon = 2 * 12

allowed_discrete_actions_per_symbol = [
    +1000, +9999999999999,
    -1000, -9999999999999,
]
days_skipped_after_action = 3 * 30

max_budget_at_start = 10000
min_budget_at_start = max(0.0, min([i for i in allowed_discrete_actions_per_symbol if i > 0]))

archive = Archive(work_dir + '/StockWorldData/archive.json')
normed_archive = Archive(work_dir + '/StockWorldData/archive.json')
normalizer = Normalizer(work_dir + '/StockWorldData/norm.json')
normalizer.apply(normed_archive)

# Hier fängt die conversion config an:
additional_data_sources = [
    ('financial_statements', 3 * 4),  # Jahre * Quartale
]
max_global_indicators = 3 * 4  # Jahre * Quartale

"""------"""

num_actions = 1 + len(allowed_discrete_actions_per_symbol)
max_trading_days_per_game = max_trajectory_length * days_skipped_after_action
horizon_in_trading_days = max_horizon * days_skipped_after_action


def timeline_dict_to_pointer_dict(t_dict: dict, timestamp: datetime):
    result = dict()
    for key in t_dict:
        elem = t_dict[key]
        if isinstance(elem, Timeline):
            result[key] = elem.current(timestamp)
        elif isinstance(elem, dict):
            result[key] = timeline_dict_to_pointer_dict(elem, timestamp)
        else:
            result[key] = elem
    return result


def copy_internal_pointers(p_dict):
    result = dict()
    for key in p_dict:
        elem = p_dict[key]
        if isinstance(elem, TimelinePointer):
            result[key] = elem
        elif isinstance(elem, dict):
            result[key] = copy_internal_pointers(elem)
    return result


def forward_pointer_dict(p_dict, timestamp):
    for key in p_dict:
        elem = p_dict[key]
        if isinstance(elem, TimelinePointer):
            while elem.value is not None and elem.value.timestamp < timestamp:
                elem = elem.next()
            p_dict[key] = elem
        elif isinstance(elem, dict):
            forward_pointer_dict(elem, timestamp)


class StockWorldState(State):
    def __init__(self, budget: int, shares: dict, timestamp: datetime, visible_symbols: List[str],
                 internal_pointers: dict = None):
        self.budget = budget
        self.shares = shares

        self.timestamp = timestamp

        self.visible_symbols = visible_symbols
        if internal_pointers is None:
            self.internal_pointers = self.init_pointers()
        else:
            self.internal_pointers = internal_pointers

        self.worth = self.compute_current_worth()

    def init_pointers(self):
        result = dict()
        for symbol in self.visible_symbols:
            result[symbol] = timeline_dict_to_pointer_dict(archive.dict[symbol], self.timestamp)
        result['GLOBAL'] = timeline_dict_to_pointer_dict(archive.dict['GLOBAL'], self.timestamp)
        return result

    def compute_current_worth(self):
        owned_symbols = []
        for symbol in self.visible_symbols:
            if self.shares[symbol] > 0:
                owned_symbols.append((symbol, self.shares[symbol]))

        worth = self.budget
        for symbol, num_shares in owned_symbols:
            latest_price = self.internal_pointers[symbol]['price']['close'].previous().value.description
            worth += latest_price * num_shares

        return worth

    def copy(self):
        return StockWorldState(
            self.budget, self.shares.copy(), self.timestamp, self.visible_symbols,
            copy_internal_pointers(self.internal_pointers),
        )


class StockWorldActionSchema(ActionSchema):
    def __init__(self, symbol_distribution: Categorical, action_distribution: Categorical):
        self.symbol_dist = symbol_distribution
        self.action_dist = action_distribution

    def sample(self) -> Tuple[float, tf.Tensor]:
        s_symbol = self.symbol_dist.sample()
        p_symbol = self.symbol_dist.prob(s_symbol)

        s_action = self.action_dist.sample()
        p_action = self.action_dist.prob(s_symbol)

        return p_symbol * p_action, tf.concat([s_symbol, s_action], axis=0)

    def prob(self, action: tf.Tensor) -> tf.Tensor:
        p_symbol = self.symbol_dist.prob(action[0])
        p_action = self.action_dist.prob(action[1])
        return p_symbol * p_action

    def log_prob(self, action: tf.Tensor) -> tf.Tensor:
        l_p_symbol = self.symbol_dist.log_prob(action[0])
        l_p_action = self.action_dist.log_prob(action[1])
        return l_p_symbol + l_p_action


class StockWorldInfoSet(InfoSet):
    def __init__(self, state: StockWorldState):
        super().__init__(0)

        self.internal_state = state

    def get_action_schema(self) -> ActionSchema:
        return StockWorldActionSchema(
            Categorical(logits=tf.zeros(shape=(max_symbols_per_game,))),
            Categorical(logits=tf.zeros(shape=(num_actions,))),
        )


class StockWorld(Game):

    @staticmethod
    def apply_action(state: StockWorldState, symbol: str, quantity: float):
        # find the worst price for the day
        if quantity > 0:
            worst_daily_price = state.internal_pointers[symbol]['price']['high'].value.description
        else:
            worst_daily_price = state.internal_pointers[symbol]['price']['low'].value.description

        num_shares = int(quantity / worst_daily_price)
        if num_shares < 0:
            num_shares = max(num_shares, -state.shares[symbol])
        elif num_shares > 0:
            num_shares = min(num_shares, int(state.budget / worst_daily_price))
        real_cost = num_shares * worst_daily_price

        state.budget -= real_cost
        state.shares[symbol] += num_shares

    @staticmethod
    def get_next_trading_day_by_forwarding_price_open(state: StockWorldState, steps: int):
        # first fast forward prices only then everything else
        next_trading_date = None
        for symbol in state.visible_symbols:
            for _ in range(steps):
                state.internal_pointers[symbol]['price']['open'] = state.internal_pointers[symbol]['price'][
                    'open'].next()
            if next_trading_date is None:
                next_trading_date = state.internal_pointers[symbol]['price']['open'].value.timestamp
            assert state.internal_pointers[symbol]['price']['open'].value.timestamp == next_trading_date

        return next_trading_date

    @staticmethod
    def apply_dividends(state: StockWorldState, next_trading_day):
        # Gehe durch jedes symbol.
        # Finde heraus, ob der nächste Handelstag ein ExDate für eine Dividende ist.
        # Wenn ja führe sie aus und sete den Pointer eins weiter.
        for symbol in state.visible_symbols:
            if state.shares[symbol] <= 0:
                continue
            if 'dividend' not in state.internal_pointers[symbol]:
                continue
            upcomming_dividend_dict = state.internal_pointers[symbol]['dividend']
            if upcomming_dividend_dict['dividend'].value is None:
                continue
            dividend_pointer = upcomming_dividend_dict['dividend']
            dividend_type_pointer = upcomming_dividend_dict['dividend_type']
            while dividend_pointer.value is not None and dividend_pointer.value.timestamp <= next_trading_day:
                # apply this dividend
                factor = dividend_pointer.value.description
                type = dividend_type_pointer.value.description
                if type == DividendType.cash.name:
                    state.budget += state.shares[symbol] * factor
                else:
                    state.shares[symbol] *= factor

                upcomming_dividend_dict['dividend'] = dividend_pointer.next()
                upcomming_dividend_dict['dividend_type'] = dividend_type_pointer.next()
                dividend_pointer = upcomming_dividend_dict['dividend']
                dividend_type_pointer = upcomming_dividend_dict['dividend_type']

    @staticmethod
    def act(state: StockWorldState, info_set: InfoSet, action: tf.Tensor):
        new_state = state.copy()

        symbol_id = action[0]
        action_id = action[1]

        if action_id > 0:
            action_id = action_id - 1

            # check for the chosen symbol and quantity
            symbol = state.visible_symbols[symbol_id]
            quantity = allowed_discrete_actions_per_symbol[action_id]

            if abs(quantity) > 0.0:
                StockWorld.apply_action(new_state, symbol, quantity)

        # calculate new worth at the next day
        next_trading_date = StockWorld.get_next_trading_day_by_forwarding_price_open(new_state,
                                                                                     days_skipped_after_action)
        StockWorld.apply_dividends(new_state, next_trading_date)
        state.timestamp = next_trading_date
        forward_pointer_dict(state.internal_pointers, state.timestamp)
        new_worth = new_state.compute_current_worth()

        if new_state.worth == 0.0:
            reward = 0.0
        else:
            reward = (new_worth - new_state.worth) / new_state.worth
        new_state.worth = new_worth

        return new_state, StockWorldInfoSet(new_state), tf.constant([reward], dtype=tf.float32)

    @staticmethod
    def find_consistent_seed():
        symbols = list(archive.dict.keys())
        symbols.remove('GLOBAL')
        random.shuffle(symbols)
        while True:
            # pick a random symbol first
            seed_symbol = random.choice(symbols)
            visible_symbols = [seed_symbol]
            # now pick random starting time from its prices that has max_trading_days left
            seed_prices = archive[[seed_symbol, 'price', 'open']]
            num_possible_seed_prices = len(seed_prices.list) - max_trading_days_per_game - horizon_in_trading_days
            if num_possible_seed_prices <= 1:
                continue
            # There needs to be an offset of 1 so that there is allways a previous price
            starting_date_index = random.choice(range(1, num_possible_seed_prices))
            starting_date = seed_prices.list[starting_date_index].timestamp
            end_date = seed_prices.list[starting_date_index + max_trading_days_per_game].timestamp

            # now try to find max_symbols many other symbols that are available in this interval.
            for symbol in symbols:
                if len(visible_symbols) >= max_symbols_per_game:
                    break
                if symbol in visible_symbols:
                    continue
                local_prices = archive[[symbol, 'price', 'open']]
                first_price = local_prices.list[0]
                last_price = local_prices.list[-1]
                if first_price.timestamp < starting_date and last_price.timestamp >= end_date:
                    visible_symbols.append(symbol)

            if len(visible_symbols) >= max_symbols_per_game:
                break

        if len(visible_symbols) < max_symbols_per_game:
            raise AssertionError("seeding failed")

        return visible_symbols, starting_date

    @staticmethod
    def get_root() -> Tuple[State, InfoSet]:
        visible_symbols, starting_timestamp = StockWorld.find_consistent_seed()

        starting_budget = random.uniform(min_budget_at_start, max_budget_at_start)
        initial_shares = dict()
        for symbol in visible_symbols:
            initial_shares[symbol] = 0

        root_state = StockWorldState(starting_budget, initial_shares, starting_timestamp, visible_symbols)
        return root_state, StockWorldInfoSet(root_state)

    @staticmethod
    def test_performance(estimator: any):
        root_state, root_info = StockWorld.get_root()
        start_worth = root_state.worth
        traj = create_trajectory(
            StockWorld, root_state, root_info,
            estimator, identity_exploration_function, max_trajectory_length,
        )
        for state, _, v_est, on_p_s, _, _, _, action, dr in traj:
            tf.print(on_p_s.symbol_dist.probs, summarize=-1)
            tf.print(on_p_s.action_dist.probs, summarize=-1)

            symbol_id = action[0]
            action_id = action[1]
            if action_id == 0:
                tf.print("waited")
            else:
                action_id = action_id - 1

                # check for the chosen symbol and quantity
                symbol = state.visible_symbols[symbol_id]
                quantity = allowed_discrete_actions_per_symbol[action_id]

                tf.print(f'{symbol} : {quantity}')

        traj_random = create_trajectory(
            StockWorld, root_state, root_info,
            StockWorldBaselineEstimator(), identity_exploration_function, max_trajectory_length,
        )

        # write out the trajectory
        reward = traj[-1][0].worth - start_worth
        reward_random = traj_random[-1][0].worth - start_worth
        tf.print(f'{reward} (policy) vs. {reward_random} (random)')

        global performance_history_length
        global performance_history
        if len(performance_history) > 100:
            performance_history.pop(0)
        performance_history.append(
            [reward, reward_random]
        )

        performance_history_tensor = tf.constant(performance_history)
        mean_reward = tf.reduce_mean(performance_history_tensor[:, 0])
        mean_reward_random = tf.reduce_mean(performance_history_tensor[:, 1])
        tf.print(
            f'mean reward over last {performance_history_length} tests: {mean_reward} (policy) vs. {mean_reward_random} (random)'
        )


performance_history = []
performance_history_length = 100


def stock_world_exploration_function(action_schema: ActionSchema) -> ActionSchema:
    assert isinstance(action_schema, StockWorldActionSchema)

    smoothed_symbol_dist = categorical_smoothing_function(action_schema.symbol_dist, factor=0.2)
    smoothed_action_dist = categorical_smoothing_function(action_schema.action_dist, factor=0.2)

    return StockWorldActionSchema(smoothed_symbol_dist, smoothed_action_dist)


class StockWorldBaselineEstimator(Estimator):
    def __init__(self):
        pass

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        result = []
        for info_set in info_sets:
            result.append((
                tf.zeros(shape=(1,)),
                info_set.get_action_schema(),
            ))
        return result

    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        pass

    def apply_gradients(self, grads: VTraceGradients):
        pass

    def save(self, checkpoint_location: str) -> None:
        pass

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        pass


def fill_normed_pointer_copy(source, normed_archive_source, target):
    for key in source:
        e = source[key]
        if isinstance(e, TimelinePointer):
            if key in normed_archive_source:
                timeline = normed_archive_source[key].list
                event = e.value
                if event is not None:
                    event = timeline[e.index]
                target[key] = TimelinePointer(
                    event,
                    e.index,
                    normed_archive_source[key]
                )
            else:
                target[key] = e
        else:
            if key not in target:
                target[key] = dict()
            normed_sub_dict = dict()
            if key in normed_archive_source:
                normed_sub_dict = normed_archive_source[key]
            fill_normed_pointer_copy(e, normed_sub_dict, target[key])


def info_set_to_vector(info_set: InfoSet):
    global additional_data_sources
    assert isinstance(info_set, StockWorldInfoSet)

    state = info_set.internal_state
    normed_internal_pointers = dict()
    fill_normed_pointer_copy(state.internal_pointers, normed_archive.dict, normed_internal_pointers)

    symbol_profiles = []
    for symbol in state.visible_symbols:
        latest_price = normed_internal_pointers[symbol]['price']['close'].previous().value.description
        latest_market_cap = normed_internal_pointers[symbol]['market_cap']['marketCap'].previous().value.description

        if state.shares[symbol] == 0:
            owned_share = 0.0
        else:
            non_normed_market_cap = state.internal_pointers[symbol]['market_cap'][
                'marketCap'].previous().value.description
            non_normed_latest_price = state.internal_pointers[symbol]['market_cap'][
                'marketCap'].previous().value.description
            owned_share = non_normed_market_cap / (state.shares[symbol] * non_normed_latest_price)

        marked_cap_mean, marked_cap_std = normalizer.dict['market_cap']['marketCap']

        normed_budget = (state.budget - marked_cap_mean) / marked_cap_std

        profile = tf.constant([
            normed_budget,
            owned_share,
            latest_price,
            latest_market_cap,
        ], dtype=tf.float32)
        profile = tf.expand_dims(profile, axis=0)

        local_symbol_profile = [profile]

        for source_name, source_horizon in additional_data_sources:
            local_data = []
            source_data = normed_internal_pointers[symbol]['financial_statement']
            for key in source_data:
                pointer = source_data[key]
                for _ in range(source_horizon):
                    if pointer is None:
                        local_data.append(0.0)
                    else:
                        pointer = pointer.previous()
                        if pointer.value is None:
                            local_data.append(0.0)
                        else:
                            local_data.append(pointer.value.description)

            local_data = tf.constant(local_data, dtype=tf.float32)
            local_data = tf.reshape(local_data, shape=(-1, source_horizon))
            local_data = tf.transpose(local_data)

            local_symbol_profile.append(local_data)

        symbol_profiles.append(local_symbol_profile)

    # stack the symbol data source wise
    num_data_sources = len(additional_data_sources) + 1
    stacked_profiles = [None] * num_data_sources
    for i in range(num_data_sources):
        column_i = [row[i] for row in symbol_profiles]
        stacked_profiles[i] = tf.stack(column_i)

    global_profile = []
    global_data = normed_internal_pointers['GLOBAL']
    for indicator in global_data:
        values = []
        pointer = global_data[indicator]['value']
        for _ in range(max_global_indicators):
            if pointer is None:
                values.append(0.0)
            else:
                pointer = pointer.previous()
                if pointer.value is None:
                    values.append(0.0)
                else:
                    values.append(pointer.value.description)

        global_profile.append(tf.constant(values, dtype=tf.float32))

    global_profile = tf.stack(global_profile)
    global_profile = tf.transpose(global_profile)
    global_profile = tf.expand_dims(global_profile, axis=0)

    return global_profile, stacked_profiles


def to_batch(info_sets: List[InfoSet]):
    global_profile_batch, symbol_profiles_batch = zip(
        *[info_set_to_vector(info_set) for info_set in info_sets])
    global_profile_batch = tf.stack(global_profile_batch)

    # stack the symbol data source wise
    global additional_data_sources
    num_data_sources = len(additional_data_sources) + 1
    stacked_profile_batches = [None] * num_data_sources
    for i in range(num_data_sources):
        column_i = [row[i] for row in symbol_profiles_batch]
        stacked_profile_batches[i] = tf.stack(column_i)

    return global_profile_batch, stacked_profile_batches


class StockWorldCNNEncoder(tf.keras.Model):
    def __init__(self, num_layers, kernel_sizes: List[Tuple[int, int]], strides: List[Tuple[int, int]], filters,
                 filter_multiplier, dim_out, padding='same'):
        super().__init__()

        assert len(kernel_sizes) == len(strides) == num_layers

        self.cnn_layers = []
        for i in range(num_layers):
            self.cnn_layers.append(
                tf.keras.layers.Conv2D(
                    kernel_size=kernel_sizes[i], strides=strides[i], activation=tf.keras.layers.LeakyReLU(), filters=filters, padding=padding
                )
            )
            filters *= filter_multiplier
        self.dense_out = tf.keras.layers.Dense(activation=None, units=dim_out)

        self.dim_out = dim_out

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, x, training=False, *args, **kwargs):
        x = x  # (batch_size, num_symbols, horizon, dim_in)
        shape_x = tf.shape(x)
        batch_size = shape_x[0]
        num_symbols = shape_x[1]
        x = tf.reshape(x, shape=(-1, shape_x[2], shape_x[3], 1))  # (batch_size * num_symbols, horizon, dim_in, 1)

        for layer in self.cnn_layers:
            x = layer(x, training=training)

        x = tf.reshape(x, shape=(batch_size, num_symbols, -1))
        x = self.dense_out(x, training=training)  # (batch_size, num_symbols, dim_out)

        return x


class StockWorldFeedForward(tf.keras.Model):
    def __init__(self, num_layers, dim_hidden, dim_out):
        super().__init__()
        self.internal_layers = []
        for i in range(num_layers):
            self.internal_layers.append(
                tf.keras.layers.Dense(
                    units=dim_hidden, activation=tf.keras.layers.LeakyReLU()
                )
            )
        self.dense_out = tf.keras.layers.Dense(activation=None, units=dim_out)

        self.dim_out = dim_out

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, x, training=False, *args, **kwargs):
        x = x  # (batch_size, num_symbols, dim_in)

        for layer in self.internal_layers:
            x = layer(x, training=training)

        x = self.dense_out(x, training=training)  # (batch_size, num_symbols, dim_out)

        return x


class SequentialList(tf.keras.layers.Layer):
    def __init__(self, layers, residual=False, norm=False):
        super().__init__()
        self.internal_layers = layers

        self.residual = residual
        if self.residual:
            self.res_layer = tf.keras.layers.Dense(
                units=self.internal_layers[-1].dim_out, activation=None, use_bias=True
            )

        self.norm = norm
        if self.norm:
            self.norm_layer = tf.keras.layers.LayerNormalization()

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, x, training=False, *args, **kwargs):
        x_shape = tf.shape(x)
        x_in = tf.reshape(x, (x_shape[0], x_shape[1], -1))
        for layer in self.internal_layers:
            x = layer(x, training=training)
        if self.residual:
            x += self.res_layer(x_in, training=True)
        if self.norm:
            x = self.norm_layer(x, training=True)
        return x


class StockWorldEncoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.symbol_data_encoders = [
            # profile
            SequentialList([
                StockWorldCNNEncoder(
                    num_layers=0, kernel_sizes=[], strides=[], filters=1, filter_multiplier=1, dim_out=10,
                ),
            ]),
            # financial statements
            SequentialList([
                StockWorldCNNEncoder(
                    num_layers=4,
                    kernel_sizes=[(3, 6), (3, 6), (3, 6), (3, 6)], strides=[(1, 2), (2, 2), (2, 2), (1, 3)],
                    filters=10, filter_multiplier=1.5, dim_out=800,
                ),
                StockWorldFeedForward(num_layers=1, dim_hidden=800, dim_out=100),
            ])
        ]

        self.symbol_encoder = SequentialList([
            StockWorldFeedForward(num_layers=2, dim_hidden=200, dim_out=100),
        ], residual=True, norm=True)

        self.global_symbol_encoder = SequentialList([
            StockWorldCNNEncoder(
                num_layers=3,
                kernel_sizes=[(3, 6), (3, 6), (2, 6)], strides=[(1, 2), (2, 2), (1, 3)],
                filters=10, filter_multiplier=2, dim_out=800,
            ),
            StockWorldFeedForward(num_layers=1, dim_hidden=800, dim_out=200),
        ], norm=True)

        self.global_data_encoder = SequentialList([
            StockWorldCNNEncoder(
                num_layers=3,
                kernel_sizes=[(3, 6), (4, 4), (3, 3)], strides=[(1, 2), (2, 2), (2, 2)],
                filters=10, filter_multiplier=1.5, dim_out=200,
            ),
            StockWorldFeedForward(num_layers=2, dim_hidden=200, dim_out=100),
        ], norm=True)

        self.global_context_encoder = SequentialList([
            StockWorldFeedForward(num_layers=3, dim_hidden=512, dim_out=200),
        ], residual=True, norm=True)

    def __call__(self, x, training=False, *args, **kwargs):
        global_profile_batch, stacked_profile_batches = x
        global_profile_batch = global_profile_batch  # (batch_size, 1, horizon, dim)
        stacked_profile_batches = stacked_profile_batches  # List[(batch_size, num_symbols, horizon, dim)]
        num_symbols = stacked_profile_batches[0].shape[1]

        # Encode symbol information into a single vector
        # first encode source wise
        global additional_data_sources
        num_data_sources = len(additional_data_sources) + 1
        symbol_data_encodings = [None] * num_data_sources
        for i in range(num_data_sources):
            data = stacked_profile_batches[i]
            symbol_data_encodings[i] = self.symbol_data_encoders[i](data,
                                                                    training=training)  # (batch_size, num_symbols, dim_out)
        # Now combine this information into a single vector
        encodings = tf.concat(symbol_data_encodings, axis=-1)
        symbol_encodings = self.symbol_encoder(encodings, training=training)  # (batch_size, num_symbol, dim)
        # Now combine information of all symbols into a single global symbol context vector
        global_symbol_encoding = self.global_symbol_encoder(tf.expand_dims(symbol_encodings, axis=1),
                                                            training=training)  # (batch_size, 1, dim_out)

        # Encode global data into single vector
        global_data_encoding = self.global_data_encoder(global_profile_batch,
                                                        training=training)  # (batch_size, 1, dim_out)
        # Add the global symbol information to this embedding
        global_context = tf.concat([global_data_encoding, global_symbol_encoding], axis=-1)
        global_context_vector = self.global_context_encoder(global_context,
                                                            training=training)  # (batch_size, 1, dim_out)

        local_profile = symbol_data_encodings[0]  # (batch_size, num_symbols, dim)
        symbol_encodings = symbol_encodings  # (batch_size, num_symbol, dim)
        global_context_vector = tf.tile(global_context_vector, [1, num_symbols, 1])  # (batch_size, 1, dim_out)
        final_symbol_encodings = tf.concat([local_profile, symbol_encodings, global_context_vector], axis=-1)

        return final_symbol_encodings


class StockWorldValueNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = StockWorldEncoder()

        self.value_head = SequentialList([
            StockWorldFeedForward(num_layers=3, dim_hidden=1000, dim_out=1),
        ])

    def __call__(self, x, training=False, *args, **kwargs):
        x, local_policies, policy_estimate = x
        _, stacked_profile_batches = x
        num_symbols = stacked_profile_batches[0].shape[1]

        policy_head_input = self.encoder(x, trainin=training)

        # now compute the state value by computing a value for each of the symbols while using information about the
        # global policy and than adding those values together.
        value_head_input = tf.concat([
            local_policies,
            tf.tile(tf.expand_dims(policy_estimate, axis=1), [1, num_symbols, 1]),
            policy_head_input
        ], axis=-1)
        local_values = self.value_head(value_head_input, training=training)  # (batch_size, num_symbols, 1)
        value_estimate = tf.expand_dims(tf.reduce_sum(local_values, axis=[2, 1]), axis=-1)  # (batch_size, 1)

        return value_estimate

    def get_variables(self):
        return self.trainable_variables


class StockWorldPolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = StockWorldEncoder()

        self.policy_head = SequentialList([
            StockWorldFeedForward(num_layers=3, dim_hidden=1000, dim_out=1 + num_actions),
        ])

    def __call__(self, x, training=False, *args, **kwargs):
        policy_head_input = self.encoder(x, trainin=training)

        # now for each symbol compute a score (how interesting this symbol is) and a policy over the available actions
        local_policies = self.policy_head(policy_head_input,
                                          training=training)  # (batch_size, num_symbols, 1 + actions)

        symbols_scores = local_policies[:, :, 0]
        symbol_policy = tf.math.softmax(symbols_scores, axis=-1)  # (batch_size, num_symbols)
        action_scores = tf.expand_dims(symbol_policy, axis=-1) * local_policies[:, :, 1:]
        action_scores = tf.reduce_sum(action_scores, axis=1)  # (batch_size, actions)
        policy_estimate = tf.concat([symbols_scores, action_scores], axis=-1)  # (batch_size, num_symbols + actions)

        return local_policies, policy_estimate

    def get_variables(self):
        return self.trainable_variables


class StockWorldEstimator(Estimator):
    def __init__(self):
        self.weight_decay = 0.0000001

        self.network_value = StockWorldValueNetwork()
        self.network_policy = StockWorldPolicyNetwork()

        self.optimizer_value = tfa.optimizers.SGDW(
            weight_decay=self.weight_decay,
            learning_rate=0.0005,
            momentum=0.9,
            nesterov=True,
        )

        self.optimizer_policy = tfa.optimizers.SGDW(
            weight_decay=self.weight_decay,
            learning_rate=0.0005,
            momentum=0.9,
            nesterov=True,
        )

        self.version = 0

    def get_variables(self):
        tv = []
        tv += self.network_value.get_variables()
        tv += self.network_policy.get_variables()
        return tv

    def vector_to_action_schema(self, logits):
        symbol_dist = Categorical(logits[:max_symbols_per_game])
        action_dist = Categorical(logits[max_symbols_per_game:])
        return StockWorldActionSchema(symbol_dist, action_dist)

    def evaluate(self, info_sets: List[InfoSet], training=False) -> List[Tuple[tf.Tensor, ActionSchema]]:
        global_profile_batch, symbol_profiles_batch = to_batch(info_sets)

        local_policies, output_policy = self.network_policy((global_profile_batch, symbol_profiles_batch))

        output_value = self.network_value(
            (
                (global_profile_batch, symbol_profiles_batch),
                tf.stop_gradient(local_policies),
                tf.stop_gradient(output_policy),
            )
        )

        action_schemas: List[ActionSchema] = [
            self.vector_to_action_schema(vector) for vector in tf.unstack(output_policy)
        ]
        values = tf.unstack(output_value, axis=0)
        return list(zip(values, action_schemas))

    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        with tf.GradientTape(persistent=True) as tape:
            info_sets, reach_weights, value_targets, q_value_targets = zip(*targets)
            value_estimates, on_policy_action_schemas = zip(*self.evaluate(info_sets, training=True))
            value_losses = reach_weights * tf.keras.losses.huber(
                y_pred=tf.stack(value_estimates), y_true=tf.stack(value_targets)
            )
            value_loss = tf.reduce_mean(value_losses)

            max_id = tf.squeeze(tf.argmax(value_targets))
            tf.print(f'{value_estimates[max_id]} vs. {value_targets[max_id]}')
            tf.print(value_loss)

            policy_losses = []
            for i in range(len(targets)):
                local_policy_losses = []
                action_schema = on_policy_action_schemas[i]
                assert isinstance(action_schema, StockWorldActionSchema)
                value_estimate = value_estimates[i]
                for action, importance, q_value in q_value_targets[i]:
                    advantage = q_value - value_estimate
                    on_policy_log_prob = action_schema.log_prob(action)
                    policy_loss = importance * on_policy_log_prob * advantage
                    local_policy_losses.append(policy_loss)
                policy_losses.append(-tf.reduce_mean(tf.stack(local_policy_losses)))

            policy_loss = tf.reduce_mean(reach_weights * tf.stack(policy_losses))

        tv = self.network_value.get_variables()
        value_grads = tape.gradient(value_loss, tv)

        tp = self.network_policy.get_variables()
        policy_grads = tape.gradient(policy_loss, tp)

        return value_grads, policy_grads

    def apply_gradients(self, grads: VTraceGradients):
        value_grads, policy_grads = grads

        tv = self.network_value.get_variables()
        value_grads, _ = tf.clip_by_global_norm(value_grads, 5.0)
        self.optimizer_value.apply_gradients(zip(value_grads, tv))

        tp = self.network_policy.get_variables()
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, 5.0)
        self.optimizer_policy.apply_gradients(zip(policy_grads, tp))

    def save(self, checkpoint_location: str) -> None:
        value_net_location = os.path.dirname(checkpoint_location) + "/data/value"
        policy_net_location = os.path.dirname(checkpoint_location) + "/data/policy"
        checkpoint_object = jsonpickle.encode({
            "version": random.randint(1, 10000000),
            "value_net_location": value_net_location,
            "policy_net_location": policy_net_location
        })
        success = False
        while success is False:
            try:
                os.makedirs(os.path.dirname(value_net_location), exist_ok=True)
                self.network_value.save_weights(value_net_location)
                self.network_policy.save_weights(policy_net_location)
                with open(checkpoint_location, 'w') as f:
                    f.write(checkpoint_object)
                success = True
                tf.print("saved")
            except:
                tf.print("save error")
                sleep(1)

    def load(self, checkpoint_location: str, blocking: bool = True) -> None:
        _, info_set = StockWorld.get_root()
        self.evaluate([info_set])
        success = False
        while success is False:
            try:
                with open(checkpoint_location, 'r') as f:
                    json_object = f.read()
                    checkpoint = jsonpickle.decode(json_object)
                    version = checkpoint["version"]
                    if self.version != version:
                        tf.print("loaded new version")
                        self.network_value.load_weights(checkpoint["value_net_location"])
                        self.network_policy.load_weights(checkpoint["policy_net_location"])
                        self.version = version
                    success = True
            except:
                tf.print("load error")
                sleep(10)
                if blocking is False:
                    success = True