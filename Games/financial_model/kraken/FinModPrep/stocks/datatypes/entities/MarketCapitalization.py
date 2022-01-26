from datetime import datetime, timezone, timedelta
from typing import List, Callable, Tuple

import requests

from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_assertions import assert_float_greater_zero
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_entity_functions import day_to_timestamp, \
    load_history

mapping: List[Tuple[str, Callable]] = [
    ('marketCap', assert_float_greater_zero),
]


def pre_processor(obj: dict):
    obj['date'] = day_to_timestamp(obj['date'])
    return [obj['date']], obj


def request_by(symbol: str):
    assert isinstance(symbol, str)
    response = requests.get(
        'https://financialmodelingprep.com/api/v3/historical-market-capitalization/{symbol}?limit=9999999&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol)
    ).json()
    if not response:
        return []
    assert isinstance(response, List)
    assert all(statement['symbol'] == symbol for statement in response)
    return load_history(response, pre_processor, mapping)
