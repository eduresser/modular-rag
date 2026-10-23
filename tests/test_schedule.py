from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.schedule import InvalidScheduleError, next_fire


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def test_next_fire_finds_the_same_day_occurrence_when_it_has_not_passed_yet():
    assert next_fire("0 3 * * *", now=_utc(2026, 9, 10, 2, 0)) == _utc(2026, 9, 10, 3, 0)


def test_next_fire_rolls_over_to_the_next_day_once_the_time_has_passed():
    assert next_fire("0 3 * * *", now=_utc(2026, 9, 10, 4, 0)) == _utc(2026, 9, 11, 3, 0)


def test_next_fire_is_strictly_after_now_not_at_or_after():
    assert next_fire("0 3 * * *", now=_utc(2026, 9, 10, 3, 0)) == _utc(2026, 9, 11, 3, 0)


def test_step_values_count_from_the_start_of_the_field_not_from_now():
    assert next_fire("*/15 * * * *", now=_utc(2026, 9, 10, 2, 7)) == _utc(2026, 9, 10, 2, 15)


def test_a_sunday_is_read_the_same_way_whether_named_0_or_7():
    from_zero = next_fire("0 3 * * 0", now=_utc(2026, 9, 10, 0, 0))
    from_seven = next_fire("0 3 * * 7", now=_utc(2026, 9, 10, 0, 0))

    assert from_zero == from_seven == _utc(2026, 9, 13, 3, 0)


def test_the_result_carries_the_timezone_of_now():
    result = next_fire("0 3 * * *", now=_utc(2026, 9, 10, 2, 0))

    assert result.tzinfo == timezone.utc


@pytest.mark.parametrize(
    "expression",
    [
        "not a schedule",
        "* * * *",
        "* * * * * *",
        "* * * * * * *",
        "@daily",
        "@hourly",
    ],
)
def test_anything_other_than_five_fields_is_rejected(expression):
    with pytest.raises(InvalidScheduleError) as excinfo:
        next_fire(expression, now=_utc(2026, 9, 10, 0, 0))

    assert expression in str(excinfo.value)
    assert "five-field" in str(excinfo.value)


@pytest.mark.parametrize(
    "expression",
    [
        "60 * * * *",
        "* 24 * * *",
        "* * 32 * *",
        "* * * 13 *",
        "gibberish here now and more",
    ],
)
def test_a_five_field_expression_with_an_unreadable_value_is_rejected(expression):
    with pytest.raises(InvalidScheduleError) as excinfo:
        next_fire(expression, now=_utc(2026, 9, 10, 0, 0))

    assert expression in str(excinfo.value)
