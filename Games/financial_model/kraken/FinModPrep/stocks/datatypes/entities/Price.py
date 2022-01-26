from datetime import datetime, timezone, timedelta
from typing import List, Callable, Tuple

import requests

from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_assertions import assert_float_greater_zero, \
    assert_is_int, assert_is_float
from Games.financial_model.kraken.FinModPrep.stocks.utilities.common_entity_functions import day_to_timestamp, \
    load_history

mapping: List[Tuple[str, Callable]] = [
    ('open', assert_float_greater_zero),
    ('close', assert_float_greater_zero),
    ('high', assert_float_greater_zero),
    ('low', assert_float_greater_zero),
    ('volume', assert_is_int),
    ('vwap', assert_is_float),
]

# Erlärung zum timestamp:
# timestamp: Gespeichert werden Preisaggregate, haben also eigentlich einen Start- und Endzeitpunkt.
#            Wir speichern nur den Endzeitpunkt als timestamp.
def pre_processor(obj: dict):
    obj['date'] = day_to_timestamp(obj['date'])
    return [obj['date']], obj


def request_by(symbol: str):
    assert isinstance(symbol, str)

    today = datetime.now().strftime('%Y-%m-%d')
    response = requests.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from=1900-01-01&to={current_date}&apikey=ae94d52fd1ea5b77c7614381ced3d130'.format(
            symbol=symbol, current_date=today)
    ).json()
    if not response:
        return []
    assert response['symbol'] == symbol
    assert response['historical']
    history = response['historical']
    return load_history(history, pre_processor, mapping)
