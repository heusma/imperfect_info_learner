from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Tuple, Callable

import requests

from Games.financial_model.archive.structures.Timeline import Timeline
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_assertions import assert_is_float, assert_is_string
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_entity_functions import day_to_timestamp, \
    load_history


class DividendType(Enum):
    stock = 0,
    cash = 1,


mapping: List[Tuple[str, Callable]] = [
    ('dividend', assert_is_float),
    # Dividends can be paid in various ways, but the big two are cash and stock.
    # Cash Dividend: You will receive dividend * shares_owned in cash.
    # Stock Dividend: You will receive int(dividend * shares_owned) new shares. The remainder is payed out in cash.
    ('dividend_type', assert_is_string),
]


def pre_processor_dividend(obj: dict):
    assert isinstance(obj, dict)
    assert 'date' in obj

    obj['date'] = day_to_timestamp(obj['date'])

    # Unser Partner hat derzeit keine Informationen über den dividenden typ.
    # Daher nehmen wir eine Cash Dividende an.
    obj['dividend_type'] = DividendType.cash.name

    return [obj['date']], obj


def pre_processor_split(obj: dict):
    assert isinstance(obj, dict)
    assert 'date' in obj

    # Es handlet sich hier um das exDate.
    # Das exDate ist der erste Handelstag nach dem stocksplit.
    # Unser derzeitiger Partner gibt uns nur das exDate for splits.
    # Also der erste Handelstag nach dem split.
    obj['date'] = day_to_timestamp(obj['date'])

    assert 'numerator' in obj and isinstance(obj['numerator'], float)
    assert 'denominator' in obj and isinstance(obj['denominator'], float)
    obj['dividend'] = obj['numerator'] / obj['denominator']

    obj['dividend_type'] = DividendType.stock.name

    return [obj['date']], obj


def request_dividends(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert response['symbol'] == symbol
    assert isinstance(response['historical'], List) and all(isinstance(item, dict) for item in response['historical'])
    history = response['historical']
    return load_history(history, pre_processor_dividend, mapping)


def request_split(symbol: str):
    assert isinstance(symbol, str)

    response = requests.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/stock_split/{symbol}?apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert response['symbol'] == symbol
    assert isinstance(response['historical'], List) and all(isinstance(item, dict) for item in response['historical'])
    history = response['historical']
    return load_history(history, pre_processor_split, mapping)


def request_by(symbol: str):
    dividends = request_dividends(symbol)
    splits = request_split(symbol)
    # Beide sind Dividenden. Da sehr unwahrscheinlich am selben Tag dividenden gezahlt
    # und gesplittet wird können auch beide in die selbe Kategorie.
    result = dict()
    keys = []
    for key in dividends:
        if key not in keys:
            keys.append(key)
    for key in splits:
        if key not in keys:
            keys.append(key)
    for key in keys:
        if key in dividends:
            a = dividends[key]
        else:
            a = Timeline()
        if key in splits:
            b = splits[key]
        else:
            b = Timeline()
        result[key] = a + b
    return result
