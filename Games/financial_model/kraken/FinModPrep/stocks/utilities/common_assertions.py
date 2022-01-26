from datetime import datetime, timezone
from enum import Enum


def assert_is_float(value):
    if value is None:
        value = 0.0
    value = float(value)
    assert isinstance(value, float)
    return value


def assert_float_greater_zero(value):
    value = assert_is_float(value)
    assert value > 0
    return value


def assert_is_int(value):
    value = int(value)
    assert isinstance(value, int)
    return value


def assert_is_datetime(value):
    assert isinstance(value, datetime)
    return value


def assert_is_string(value):
    assert isinstance(value, str)
    return value
