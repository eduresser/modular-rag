from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from src.analytics.rollup import (
    days_with_raw_rows,
    prune_raw,
    roll_up_usage,
    run_rollup,
    search_aggregate,
    usage_aggregate,
)
from src.prices import OPERATION_READ, OPERATION_CHUNK, UNIT_PAGE, PriceBook
from src.search_events import SEARCH_OUTCOME_FOUND, SEARCH_OUTCOME_NOTHING_MATCHED, SearchEvent, record_search
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_CONFIGURATION_CHANGED,
    REASON_NEW_DOCUMENT,
    priced,
    record_usage,
)
from tests.conftest import TEST_CLIENT

READER = "pretrained-layout-parser-v1.0-2024-06-03"
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _usage(conn, book, *, at, units=100, reason=REASON_NEW_DOCUMENT, operation=OPERATION_READ,
           outcome=OUTCOME_SUCCEEDED, duration_ms=100, provider="google") -> int:
    return record_usage(
        conn,
        priced(
            book,
            operation=operation,
            reason=reason,
            provider=provider,
            model=READER,
            unit=UNIT_PAGE,
            units=units,
            duration_ms=duration_ms,
            outcome=outcome,
            occurred_at=at,
        ),
        client=TEST_CLIENT,
    )


def _search(conn, *, at, duration_ms=30, outcome=SEARCH_OUTCOME_FOUND, document_ids=(1, 2)) -> int:
    return record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question=f"pergunta {at.isoformat()} {duration_ms}",
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=duration_ms,
            outcome=outcome,
            document_ids=list(document_ids),
            chunk_ids=[1] if document_ids else [],
            occurred_at=at,
        ),
        client=TEST_CLIENT,
    )


def _raw_counts(conn: psycopg.Connection) -> tuple[int, int]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        usage = int(cursor.fetchone()[0])
        cursor.execute("SELECT count(*) FROM search_events")
        return usage, int(cursor.fetchone()[0])


def test_a_retention_of_zero_days_is_refused(conn: psycopg.Connection):
    for retention in (0, -1):
        with pytest.raises(ValueError) as excinfo:
            run_rollup(conn, retention_days=retention, now=NOW)
        assert "at least one day" in str(excinfo.value)


def test_nothing_to_roll_up_is_not_an_error(conn: psycopg.Connection):
    result = run_rollup(conn, retention_days=30, now=NOW)

    assert result.days_rolled_up == ()
    assert result.usage_aggregates == 0
    assert result.pruned_anything is False


def test_only_whole_days_are_pruned(conn: psycopg.Connection, price_book: PriceBook):
    oldest_retained = date(2026, 8, 31)
    just_inside = datetime(2026, 8, 31, 0, 30, tzinfo=timezone.utc)
    just_outside = datetime(2026, 8, 30, 23, 30, tzinfo=timezone.utc)
    _usage(conn, price_book, at=just_inside)
    _usage(conn, price_book, at=just_outside)

    result = run_rollup(conn, retention_days=30, now=NOW)

    assert result.oldest_retained_day == oldest_retained
    assert result.usage_rows_pruned == 1
    assert _raw_counts(conn)[0] == 1


def test_a_pruned_days_aggregate_survives_every_later_run(conn: psycopg.Connection, price_book: PriceBook):
    august = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=august, units=300)
    run_rollup(conn, retention_days=30, now=NOW)
    august_first = date(2026, 8, 1)
    assert usage_aggregate(conn, august_first)[0].units == Decimal("300")

    _usage(conn, price_book, at=datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc), units=10)
    run_rollup(conn, retention_days=30, now=NOW)
    run_rollup(conn, retention_days=30, now=NOW + timedelta(days=1))

    assert usage_aggregate(conn, august_first)[0].units == Decimal("300")


def test_the_aggregate_is_written_before_anything_is_deleted(conn: psycopg.Connection, price_book: PriceBook):
    august = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=august)
    _search(conn, at=august)

    result = run_rollup(conn, retention_days=30, now=NOW)

    assert result.usage_rows_pruned == 1
    assert result.search_rows_pruned == 1
    assert _raw_counts(conn) == (0, 0)
    assert usage_aggregate(conn, date(2026, 8, 1))
    assert search_aggregate(conn, date(2026, 8, 1))


def test_the_usage_aggregate_is_grouped_by_every_way_it_will_be_asked_about(
    conn: psycopg.Connection, price_book: PriceBook
):
    day = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=day, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT, units=100)
    _usage(conn, price_book, at=day, operation=OPERATION_READ, reason=REASON_CONFIGURATION_CHANGED, units=50)
    _usage(conn, price_book, at=day, operation=OPERATION_CHUNK, reason=REASON_CONFIGURATION_CHANGED, units=50)
    _usage(conn, price_book, at=day, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT, units=25,
           provider="docling")

    run_rollup(conn, retention_days=30, now=NOW)

    rows = usage_aggregate(conn, date(2026, 9, 29))

    assert {(row.operation, row.reason, row.provider) for row in rows} == {
        (OPERATION_READ, REASON_NEW_DOCUMENT, "google"),
        (OPERATION_READ, REASON_CONFIGURATION_CHANGED, "google"),
        (OPERATION_CHUNK, REASON_CONFIGURATION_CHANGED, "google"),
        (OPERATION_READ, REASON_NEW_DOCUMENT, "docling"),
    }
    assert [row.cost for row in rows] == sorted((row.cost for row in rows), reverse=True)


def test_the_error_count_survives_into_the_aggregate(conn: psycopg.Connection, price_book: PriceBook):
    day = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=day, outcome=OUTCOME_SUCCEEDED)
    _usage(conn, price_book, at=day, outcome=OUTCOME_FAILED)
    _usage(conn, price_book, at=day, outcome=OUTCOME_FAILED)

    run_rollup(conn, retention_days=30, now=NOW)

    row = usage_aggregate(conn, date(2026, 9, 29))[0]
    assert row.events == 3
    assert row.errors == 2


def test_the_search_aggregate_separates_the_three_outcomes(conn: psycopg.Connection):
    day = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    _search(conn, at=day, duration_ms=30)
    _search(conn, at=day, duration_ms=40)
    _search(conn, at=day, duration_ms=15, outcome=SEARCH_OUTCOME_NOTHING_MATCHED, document_ids=())

    run_rollup(conn, retention_days=30, now=NOW)

    rows = {row.outcome: row for row in search_aggregate(conn, date(2026, 9, 29))}
    assert rows[SEARCH_OUTCOME_FOUND].searches == 2
    assert rows[SEARCH_OUTCOME_FOUND].documents_returned == 4
    assert rows[SEARCH_OUTCOME_NOTHING_MATCHED].searches == 1
    assert rows[SEARCH_OUTCOME_NOTHING_MATCHED].documents_returned == 0


def test_a_day_is_cut_in_utc_whatever_the_session_carries(conn: psycopg.Connection, price_book: PriceBook):
    late = datetime(2026, 9, 29, 23, 30, tzinfo=timezone.utc)
    with conn.cursor() as cursor:
        cursor.execute("SET TIME ZONE 'America/Sao_Paulo'")
    _usage(conn, price_book, at=late)

    assert days_with_raw_rows(conn) == (date(2026, 9, 29),)
    run_rollup(conn, retention_days=30, now=NOW)
    assert usage_aggregate(conn, date(2026, 9, 29))
    assert usage_aggregate(conn, date(2026, 9, 30)) == ()


def test_a_targeted_rerun_aggregates_only_the_day_it_was_given(
    conn: psycopg.Connection, price_book: PriceBook
):
    first = datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc)
    second = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=first)
    _usage(conn, price_book, at=second)

    result = run_rollup(conn, retention_days=30, now=NOW, days=(date(2026, 9, 27),))

    assert result.days_rolled_up == (date(2026, 9, 27),)
    assert usage_aggregate(conn, date(2026, 9, 27))
    assert usage_aggregate(conn, date(2026, 9, 28)) == ()


def test_a_targeted_rerun_still_prunes_what_is_past_retention(
    conn: psycopg.Connection, price_book: PriceBook
):
    august = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=august)
    roll_up_usage(conn, (date(2026, 8, 1),))

    result = run_rollup(conn, retention_days=30, now=NOW, days=(date(2026, 9, 29),))

    assert result.usage_rows_pruned == 1
    assert usage_aggregate(conn, date(2026, 8, 1))


def test_days_with_raw_rows_unions_both_tables(conn: psycopg.Connection, price_book: PriceBook):
    _usage(conn, price_book, at=datetime(2026, 9, 27, 9, 0, tzinfo=timezone.utc))
    _search(conn, at=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc))
    _search(conn, at=datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc))

    assert days_with_raw_rows(conn) == (date(2026, 9, 27), date(2026, 9, 28))


def test_pruning_reports_what_each_table_lost(conn: psycopg.Connection, price_book: PriceBook):
    august = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=august)
    _usage(conn, price_book, at=august)
    _search(conn, at=august)

    assert prune_raw(conn, date(2026, 8, 31)) == (2, 1)


def test_the_aggregate_keeps_the_pricing_gap_the_raw_rows_would_have_shown(
    conn: psycopg.Connection, price_book: PriceBook
):
    august = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=august, units=100)
    _usage(conn, price_book, at=august, units=5_000, provider="fornecedor-novo")

    run_rollup(conn, retention_days=30, now=NOW)

    rows = {row.provider: row for row in usage_aggregate(conn, date(2026, 8, 1))}
    assert rows["google"].cost == Decimal("1.00")
    assert rows["google"].complete
    assert rows["fornecedor-novo"].cost == Decimal("0")
    assert rows["fornecedor-novo"].complete is False
    assert rows["fornecedor-novo"].unpriced_units == Decimal("5000")
    assert rows["fornecedor-novo"].unpriced_events == 1
    assert _raw_counts(conn)[0] == 0


def test_a_rerun_corrects_the_pricing_gap_once_a_price_arrives(
    conn: psycopg.Connection, price_book: PriceBook
):
    day = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    _usage(conn, price_book, at=day, units=5_000, provider="fornecedor-novo")
    run_rollup(conn, retention_days=30, now=NOW)
    assert usage_aggregate(conn, date(2026, 9, 29))[0].unpriced_units == Decimal("5000")

    run_rollup(conn, retention_days=30, now=NOW)

    row = usage_aggregate(conn, date(2026, 9, 29))[0]
    assert row.unpriced_units == Decimal("5000")
    assert row.unpriced_events == 1
