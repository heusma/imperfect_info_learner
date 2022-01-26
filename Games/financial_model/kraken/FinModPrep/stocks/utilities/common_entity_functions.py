from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Callable

from Games.financial_model.archive.structures.Event import Event
from Games.financial_model.archive.structures.Timeline import Timeline


def day_to_timestamp(day_str: str):
    assert isinstance(day_str, str)

    timestamp = datetime.strptime(day_str, '%Y-%m-%d')
    return timestamp


def load_history(history: List[dict], pre_processor: Callable, mapping: List[Tuple[str, Callable]]):
    assert isinstance(history, List) and all(isinstance(item, dict) for item in history)
    assert isinstance(pre_processor, Callable)

    history_length = history.__len__()
    result = dict()
    # Die add_multiple methode der Timeline wird die Liste aufsteigend sortieren müssen.
    # Die history von unserem Partner ist aber absteigen sortiert.
    for index in reversed(range(history_length)):
        timestamps, reference = pre_processor(history[index])
        for timestamp in timestamps:
            for key, assertion in mapping:
                assert key in reference
                value = assertion(reference[key])
                if key not in result:
                    result[key] = Timeline()
                result[key].list.append(
                    Event(timestamp, value)
                )
    return result
