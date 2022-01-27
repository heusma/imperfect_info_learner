import copy
import os

from Games.financial_model.archive.Archive import Archive
from Games.financial_model.archive.Normalizer import Normalizer
from Games.financial_model.archive.structures.Timeline import Timeline, TimelinePointer
from Games.financial_model.kraken.FinModPrep.stocks.datatypes.entities.Dividend import DividendType

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
from datetime import datetime
from time import sleep
from typing import Tuple, List

import jsonpickle
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

import matplotlib.pyplot as plt

from Distributions.CategoricalDistribution import categorical_smoothing_function
from EvaluationTool import Estimator, VTraceTarget, VTraceGradients, create_trajectory, identity_exploration_function
from MDP import Game, State, InfoSet, Leaf, ActionSchema

"""Config"""
max_symbols_per_game = 5
max_trajectory_length = 100
max_horizon = 10

max_budget_at_start = 10000

allowed_discrete_actions_per_symbol = [
    +2000, +9999999999999,
    -2000, -9999999999999,
]
days_skipped_after_wait_action = 7
days_skipped_after_transaction = 30

archive = Archive('./Games/financial_model/Games/archive.json')
normalizer = Normalizer('./Games/financial_model/Games/norm.json')
normalizer.apply(archive)

# Hier fängt die conversion config an:
max_financial_statements = 3 * 4  # Jahre * Quartale

"""------"""

num_actions = 1 + len(allowed_discrete_actions_per_symbol) * max_symbols_per_game
max_skip = max(days_skipped_after_transaction, days_skipped_after_wait_action)
max_trading_days_per_game = 200 * max_skip
horizon_in_trading_days = max_horizon * max_skip


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
    def __init__(self, categorical_distribution: tfp.distributions.Categorical):
        self.dist = categorical_distribution

    def sample(self) -> Tuple[float, tf.Tensor]:
        s = self.dist.sample()
        p = self.dist.prob(s)
        return p, s

    def prob(self, action: tf.Tensor) -> tf.Tensor:
        p = self.dist.prob(action)
        return p

    def log_prob(self, action: tf.Tensor) -> tf.Tensor:
        l_p = self.dist.log_prob(action)
        return l_p


class StockWorldInfoSet(InfoSet):
    def __init__(self, state: StockWorldState):
        super().__init__(0)

        self.internal_state = state

    def get_action_schema(self) -> ActionSchema:
        return StockWorldActionSchema(tfp.distributions.Categorical(
            logits=tf.zeros(shape=(num_actions,)))
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
    def apply_dividends(state: StockWorldState):
        # Gehe durch jedes symbol.
        # Finde heraus, ob der nächste Handelstag ein ExDate für eine Dividende ist.
        # Wenn ja führe sie aus und sete den Pointer eins weiter.

        next_trading_day = state.internal_pointers[state.visible_symbols[0]]['price']['open'].next().value.timestamp

        for symbol in state.visible_symbols:
            if 'dividend' not in state.internal_pointers[symbol]:
                continue
            upcomming_dividend_dict = state.internal_pointers[symbol]['dividend']
            if upcomming_dividend_dict['dividend'].value is None:
                continue
            upcomming_dividend_timestamp = upcomming_dividend_dict['dividend'].value.timestamp
            if upcomming_dividend_timestamp <= next_trading_day:
                # apply this dividend
                upcomming_dividend_dividend = upcomming_dividend_dict['dividend']
                upcomming_dividend_dividend_type = upcomming_dividend_dict['dividend_type']
                factor = upcomming_dividend_dividend.value.description
                type = upcomming_dividend_dividend_type.value.description
                if type == DividendType.cash.name:
                    state.budget += state.shares[symbol] * factor
                else:
                    state.shares[symbol] *= factor

    @staticmethod
    def fast_forward_to_next_trading_day(state: StockWorldState):
        # first fast forward prices only then everything else
        next_trading_date = None
        for symbol in state.visible_symbols:
            state.internal_pointers[symbol]['price']['open'] = state.internal_pointers[symbol]['price']['open'].next()
            if next_trading_date is None:
                next_trading_date = state.internal_pointers[symbol]['price']['open'].value.timestamp
            assert state.internal_pointers[symbol]['price']['open'].value.timestamp == next_trading_date

        state.timestamp = next_trading_date

        forward_pointer_dict(state.internal_pointers, state.timestamp)

    @staticmethod
    def act(state: StockWorldState, info_set: InfoSet, action: tf.Tensor):
        new_state = state.copy()

        if action == 0:
            # This is the waiting action
            days_to_skip = days_skipped_after_wait_action
        else:
            action = action - 1

            # check for the chosen symbol and quantity
            symbol_id = tf.cast(action / len(allowed_discrete_actions_per_symbol), dtype=tf.int32)
            action_id = tf.cast(action % len(allowed_discrete_actions_per_symbol), dtype=tf.int32)
            symbol = state.visible_symbols[symbol_id]
            quantity = allowed_discrete_actions_per_symbol[action_id]

            if abs(quantity) > 0.0:
                StockWorld.apply_action(new_state, symbol, quantity)

            days_to_skip = days_skipped_after_transaction

        # calculate new worth at the next day
        for _ in range(days_to_skip):
            StockWorld.apply_dividends(new_state)
            StockWorld.fast_forward_to_next_trading_day(new_state)
        new_worth = new_state.compute_current_worth()

        reward = (new_worth - new_state.worth) / new_state.worth
        new_state.worth = new_worth

        return new_state, StockWorldInfoSet(new_state), reward

    @staticmethod
    def find_consistent_seed():
        symbols = list(archive.dict.keys())
        random.shuffle(symbols)
        while True:
            # pick a random symbol first
            seed_symbol = random.choice(symbols)
            visible_symbols = [seed_symbol]
            # now pick random starting time from its prices that has max_trading_days left
            seed_prices = archive[[seed_symbol, 'price', 'open']]
            num_possible_seed_prices = len(seed_prices.list) - max_trading_days_per_game - horizon_in_trading_days
            if num_possible_seed_prices <= 0:
                continue
            starting_date_index = random.choice(range(num_possible_seed_prices))
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
                if first_price.timestamp <= starting_date and last_price.timestamp >= end_date:
                    visible_symbols.append(symbol)

            if len(visible_symbols) >= max_symbols_per_game:
                break

        if len(visible_symbols) < max_symbols_per_game:
            raise AssertionError("seeding failed")

        return visible_symbols, starting_date

    @staticmethod
    def get_root() -> Tuple[State, InfoSet]:
        visible_symbols, starting_timestamp = StockWorld.find_consistent_seed()

        starting_budget = random.random() * max_budget_at_start
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
            estimator, identity_exploration_function, max_trading_days_per_game,
        )
        tf.print(f'estimator made: {traj[-1][0].worth - start_worth}')


def stock_world_exploration_function(action_schema: ActionSchema) -> ActionSchema:
    assert isinstance(action_schema, StockWorldActionSchema)

    smoothed_dist = categorical_smoothing_function(action_schema.dist, factor=0.4)

    return StockWorldActionSchema(smoothed_dist)


class StockWorldBaselineEstimator(Estimator):
    def __init__(self):
        pass

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        result = []
        for info_set in info_sets:
            result.append((
                tf.zeros(shape=(1,)),
                info_set.get_action_schema()
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


class StockWorldNetwork(tf.keras.Model):
    def __init__(self, num_layers: int, dff: int, outputs: int):
        super().__init__()

        self.internal_layers = []
        self.internal_layer_norms = []
        for _ in range(num_layers):
            self.internal_layers.append(
                tf.keras.layers.Dense(
                    dff,
                    activation='relu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0, mode='fan_in', distribution='truncated_normal'))
            )
            self.internal_layer_norms.append(
                tf.keras.layers.LayerNormalization()
            )
        self.output_layer = tf.keras.layers.Dense(
            outputs,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, input, *args, **kwargs):
        activation = input
        for i in range(len(self.internal_layers)):
            l = self.internal_layers[i]
            ln = self.internal_layer_norms[i]
            activation = ln(l(activation))
        return self.output_layer(activation)


class StockWorldEstimator(Estimator):
    def __init__(self):
        self.weight_decay = 1e-4

        # Network config
        self.symbol_encoding_dim = 40
        self.symbol_encoding_dim_scale = 4
        self.global_encoding_dim = 80
        #

        self.internal_symbol_encoder = StockWorldNetwork(
            num_layers=3, dff=self.symbol_encoding_dim * self.symbol_encoding_dim_scale, outputs=self.symbol_encoding_dim
        )
        self.internal_global_state_encoder = StockWorldNetwork(
            num_layers=3, dff=self.symbol_encoding_dim * max_symbols_per_game, outputs=self.global_encoding_dim
        )

        self.internal_network_policy = StockWorldNetwork(
            num_layers=3, dff=self.symbol_encoding_dim * max_symbols_per_game + self.global_encoding_dim,
            outputs=num_actions
        )
        self.internal_network_value = StockWorldNetwork(
            num_layers=3, dff=self.symbol_encoding_dim * max_symbols_per_game + self.global_encoding_dim, outputs=1
        )

        self.optimizer = tfa.optimizers.SGDW(
            weight_decay=self.weight_decay,
            learning_rate=0.005,
            momentum=0.9,
            nesterov=True,
        )

        self.version = 0

    def info_set_to_vector(self, info_set: InfoSet):
        assert isinstance(info_set, StockWorldInfoSet)

        state = info_set.internal_state

        symbol_profiles = []
        for symbol in state.visible_symbols:
            latest_price = state.internal_pointers[symbol]['price']['close'].previous().value.description
            latest_market_cap = state.internal_pointers[symbol]['market_cap']['marketCap'].previous().value.description
            profile = tf.constant([
                state.budget,
                state.shares[symbol],
                latest_price,
                latest_market_cap,
            ], dtype=tf.float32)

            financial_statements = []
            financial_statement_data = state.internal_pointers[symbol]['financial_statement']
            for key in financial_statement_data:
                pointer = financial_statement_data[key]
                for _ in range(max_financial_statements):
                    if pointer is None:
                        financial_statements.append(0.0)
                    else:
                        pointer = pointer.previous()
                        if pointer.value is None:
                            financial_statements.append(0.0)
                        else:
                            financial_statements.append(pointer.value.description)

            financial_statements = tf.constant(financial_statements, dtype=tf.float32)
            symbol_profiles.append(
                tf.concat([profile, financial_statements], axis=0)
            )

        symbol_profiles = tf.stack(symbol_profiles)

        return symbol_profiles

    def vector_to_action_schema(self, logits):
        c_dist = tfp.distributions.Categorical(logits)
        return StockWorldActionSchema(c_dist)

    def get_variables(self):
        tv = []
        tv += self.internal_symbol_encoder.trainable_variables
        tv += self.internal_global_state_encoder.trainable_variables
        tv += self.internal_network_value.trainable_variables
        tv += self.internal_network_policy.trainable_variables

        return tv

    def evaluate(self, info_sets: List[InfoSet]) -> List[Tuple[tf.Tensor, ActionSchema]]:
        batch = tf.stack([self.info_set_to_vector(info_set) for info_set in info_sets])

        batch_shape = tf.shape(batch)
        symbol_encodings = self.internal_symbol_encoder(tf.reshape(batch, shape=(-1, batch_shape[-1])))
        symbol_encodings = tf.reshape(
            symbol_encodings, shape=(batch_shape[0], max_symbols_per_game, self.symbol_encoding_dim)
        )

        global_encodings = self.internal_global_state_encoder(tf.reshape(symbol_encodings, shape=(batch_shape[0], -1)))

        complete_data = tf.concat(
            [
                tf.reshape(symbol_encodings, shape=(batch_shape[0], -1)),
                tf.reshape(global_encodings, shape=(batch_shape[0], -1))
            ],
            axis=-1,
        )
        output_policy = self.internal_network_policy(complete_data)
        output_value = self.internal_network_value(complete_data)

        action_schemas: List[ActionSchema] = [
            self.vector_to_action_schema(vector) for vector in tf.unstack(output_policy)
        ]
        values = tf.unstack(output_value, axis=0)
        return list(zip(values, action_schemas))

    def compute_gradients(self, targets: List[VTraceTarget]) -> VTraceGradients:
        with tf.GradientTape() as tape:
            info_sets, reach_weights, value_targets, q_value_targets = zip(*targets)
            value_estimates, on_policy_action_schemas = zip(*self.evaluate(info_sets))

            value_losses = reach_weights * tf.keras.losses.huber(
                y_pred=tf.stack(value_estimates), y_true=tf.stack(value_targets)
            )
            value_loss = tf.reduce_mean(value_losses)

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

            loss = value_loss + policy_loss

        tv = self.get_variables()
        grads = tape.gradient(loss, tv)

        return grads, []

    def apply_gradients(self, grads: VTraceGradients):
        grads, _ = grads

        tv = self.get_variables()
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, tv))

    def save(self, checkpoint_location: str) -> None:
        value_net_location = os.path.dirname(checkpoint_location) + "/data/value"
        policy_net_location = os.path.dirname(checkpoint_location) + "/data/policy"
        symbol_encoder_location = os.path.dirname(checkpoint_location) + "/data/symbol_encoder"
        global_encoder_location = os.path.dirname(checkpoint_location) + "/data/global_encoder"
        checkpoint_object = jsonpickle.encode({
            "version": random.randint(1, 10000000),
            "value_net_location": value_net_location,
            "policy_net_location": policy_net_location,
            "symbol_encoder_location": symbol_encoder_location,
            "global_encoder_location": global_encoder_location,
        })
        success = False
        while success is False:
            try:
                os.makedirs(os.path.dirname(value_net_location), exist_ok=True)
                self.internal_network_value.save_weights(value_net_location)
                self.internal_network_policy.save_weights(policy_net_location)
                self.internal_symbol_encoder.save_weights(symbol_encoder_location)
                self.internal_global_state_encoder.save_weights(global_encoder_location)
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
                        self.internal_network_value.load_weights(checkpoint["value_net_location"])
                        self.internal_network_policy.load_weights(checkpoint["policy_net_location"])
                        self.internal_symbol_encoder.load_weights(checkpoint["symbol_encoder_location"])
                        self.internal_global_state_encoder.load_weights(checkpoint["global_encoder_location"])
                        self.version = version
                    success = True
            except:
                tf.print("load error")
                sleep(10)
                if blocking is False:
                    success = True


"""
basic playing example

root_state, root_info = StockWorld.get_root()
start_worth = root_state.worth
traj = create_trajectory(
    StockWorld, root_state, root_info,
    StockWorldEstimator(), identity_exploration_function, max_trading_days_per_game - 10,
)
tf.print(f'random trader made: {traj[-1][0].worth - start_worth}')
"""
