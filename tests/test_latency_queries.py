from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from src.analytics.latency import (
    DEFAULT_PERCENTILE,
    LatencyGroup,
    largest_step,
    latency_by_index_effort,
)
from src.search_events import SEARCH_OUTCOME_FOUND, SearchEvent, record_search
from tests.conftest import TEST_CLIENT

START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(days=7)


def _searches(conn: psycopg.Connection, effort: int, durations: tuple[int, ...], at=None) -> None:
    for index, duration in enumerate(durations):
        record_search(
            conn,
            SearchEvent(
                caller="assistente-juridico",
                question=f"pergunta {effort} {index}",
                dense_weight=0.5,
                lexical_weight=0.5,
                candidate_depth=50,
                result_count=10,
                index_search_effort=effort,
                duration_ms=duration,
                outcome=SEARCH_OUTCOME_FOUND,
                document_ids=[1],
                chunk_ids=[1],
                occurred_at=(at or START) + timedelta(minutes=index),
            ),
            client=TEST_CLIENT,
        )


def _group(effort: int, high: str) -> LatencyGroup:
    return LatencyGroup(
        index_search_effort=effort,
        searches=10,
        median_ms=Decimal(high) / 2,
        high_percentile_ms=Decimal(high),
        slowest_ms=Decimal(high) * 2,
    )


def test_no_history_reports_no_groups(conn: psycopg.Connection):
    assert latency_by_index_effort(conn, since=START, until=END) == ()


def test_the_groups_come_back_in_the_order_the_knob_turns(conn: psycopg.Connection):
    for effort in (200, 4, 40):
        _searches(conn, effort, (10, 20, 30))

    groups = latency_by_index_effort(conn, since=START, until=END)

    assert [group.index_search_effort for group in groups] == [4, 40, 200]


def test_the_percentile_asked_for_is_the_one_returned(conn: psycopg.Connection):
    _searches(conn, 40, (10, 20, 30, 40, 50))

    at_median = latency_by_index_effort(conn, since=START, until=END, percentile=0.5)[0]
    at_high = latency_by_index_effort(conn, since=START, until=END, percentile=0.9)[0]

    assert at_median.high_percentile_ms == at_median.median_ms == Decimal("30")
    assert at_high.high_percentile_ms == Decimal("46")


def test_the_default_percentile_follows_the_tail_an_average_would_flatten(conn: psycopg.Connection):
    _searches(conn, 40, (10, 10, 10, 10, 10, 10, 10, 10, 10, 2_000))

    group = latency_by_index_effort(conn, since=START, until=END)[0]

    assert DEFAULT_PERCENTILE == 0.95
    assert group.median_ms == Decimal("10")
    assert group.high_percentile_ms > Decimal("500")
    assert group.slowest_ms == Decimal("2000")


@pytest.mark.parametrize("percentile", [0, 1, -0.1, 1.5])
def test_a_percentile_outside_zero_to_one_is_refused(conn: psycopg.Connection, percentile):
    with pytest.raises(ValueError) as excinfo:
        latency_by_index_effort(conn, since=START, until=END, percentile=percentile)

    assert "fraction between 0 and 1" in str(excinfo.value)


def test_only_searches_inside_the_window_are_counted(conn: psycopg.Connection):
    _searches(conn, 40, (10, 20), at=START - timedelta(days=1))
    _searches(conn, 40, (30, 40, 50))

    group = latency_by_index_effort(conn, since=START, until=END)[0]

    assert group.searches == 3


def test_a_single_effort_group_has_no_step_to_report():
    assert largest_step(()) is None
    assert largest_step((_group(40, "30"),)) is None


def test_the_step_names_the_lowest_effort_that_still_keeps_the_index():
    groups = (_group(4, "1000"), _group(40, "35"), _group(200, "55"))

    step = largest_step(groups)

    assert step.below.index_search_effort == 4
    assert step.above.index_search_effort == 40
    assert step.effort_floor == 40
    assert step.factor > 20


def test_a_flat_curve_reports_a_step_near_one(conn: psycopg.Connection):
    groups = (_group(40, "30"), _group(100, "32"), _group(200, "34"))

    step = largest_step(groups)

    assert step is not None
    assert Decimal("0.8") < step.factor < Decimal("1.2")


def test_a_group_with_no_measurable_latency_is_skipped_rather_than_dividing_by_zero():
    groups = (_group(4, "1000"), _group(40, "0"))

    assert largest_step(groups) is None
