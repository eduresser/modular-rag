
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from src.prices import OPERATION_EMBED, OPERATION_READ, UNIT_PAGE, UNIT_TOKEN, PriceBook
from src.store import upsert_document
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_QUESTION_ASKED,
    REPROCESSING_REASONS,
    USAGE_REASONS,
    UsageError,
    UsageEvent,
    priced,
    read_usage,
    record_usage,
    usage_between,
)
from tests.conftest import TEST_CLIENT, configured_prices

READER = "pretrained-layout-parser-v1.0-2024-06-03"


def _event(**overrides) -> UsageEvent:
    base = dict(
        operation=OPERATION_READ,
        reason=REASON_NEW_DOCUMENT,
        provider="google",
        model=READER,
        unit=UNIT_PAGE,
        units=Decimal("100"),
        unit_price=Decimal("0.01"),
        duration_ms=Decimal("1200"),
    )
    base.update(overrides)
    return UsageEvent(**base)


def test_an_operation_outside_the_billable_set_is_refused_and_lists_them():
    with pytest.raises(UsageError) as excinfo:
        _event(operation="transcribe")

    message = str(excinfo.value)
    assert "transcribe" in message
    assert OPERATION_EMBED in message


def test_an_outcome_that_is_neither_success_nor_failure_is_refused():
    with pytest.raises(UsageError):
        _event(outcome="partially")


@pytest.mark.parametrize(
    "overrides",
    [
        {"units": Decimal("-1")},
        {"unit_price": Decimal("-0.01")},
        {"duration_ms": Decimal("-5")},
    ],
)
def test_negative_quantities_are_refused(overrides):
    with pytest.raises(UsageError):
        _event(**overrides)


def test_a_row_without_a_unit_is_refused_because_its_units_would_mean_nothing():
    with pytest.raises(UsageError) as excinfo:
        _event(unit="")

    assert "what a unit is" in str(excinfo.value)


def test_zero_units_are_allowed_because_a_failure_can_consume_nothing():
    event = _event(units=Decimal("0"), outcome=OUTCOME_FAILED)

    assert event.cost == Decimal("0")


def test_the_reason_vocabulary_separates_reprocessing_from_a_question():
    assert REASON_QUESTION_ASKED in USAGE_REASONS
    assert REASON_QUESTION_ASKED not in REPROCESSING_REASONS
    assert set(REPROCESSING_REASONS) < set(USAGE_REASONS)


def test_the_cost_column_is_the_databases_own_multiplication(conn: psycopg.Connection, wider_price_book):
    book = wider_price_book
    event = priced(
        book,
        operation=OPERATION_EMBED,
        reason=REASON_CONTENT_CHANGED,
        provider="openai",
        model="text-embedding-3-large",
        unit=UNIT_TOKEN,
        units=91_000,
        duration_ms=430,
    )

    row = read_usage(conn, record_usage(conn, event, client=TEST_CLIENT))

    rate = book.price_for(
        operation=OPERATION_EMBED, provider="openai",
        model="text-embedding-3-large", unit=UNIT_TOKEN,
    ).amount
    assert row.cost == event.cost == rate * 91_000


def test_nobody_can_write_a_cost_that_disagrees_with_its_price(conn: psycopg.Connection):
    with pytest.raises(psycopg.errors.GeneratedAlways), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_events
                    (client, operation, reason, provider, model, unit, units, unit_price,
                     duration_ms, outcome, cost)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (TEST_CLIENT, OPERATION_READ, REASON_NEW_DOCUMENT, "google", READER, UNIT_PAGE,
                 100, Decimal("0.01"), 10, OUTCOME_SUCCEEDED, Decimal("999.99")),
            )


def test_occurred_at_defaults_to_now_when_the_caller_says_nothing(conn: psycopg.Connection):
    before = datetime.now(timezone.utc) - timedelta(seconds=5)

    row = read_usage(conn, record_usage(conn, _event(), client=TEST_CLIENT))

    assert row.occurred_at >= before


def test_the_reading_window_is_half_open(conn: psycopg.Connection):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)
    record_usage(conn, _event(occurred_at=start), client=TEST_CLIENT)
    record_usage(conn, _event(occurred_at=end), client=TEST_CLIENT)

    inside = usage_between(conn, since=start, until=end)

    assert [row.occurred_at for row in inside] == [start]


def test_reading_a_row_that_retention_already_pruned_returns_none(conn: psycopg.Connection):
    assert read_usage(conn, 10_000_000) is None


def test_a_row_can_be_attributed_to_a_document(conn: psycopg.Connection):
    document_id = upsert_document(conn, external_id="rescisao", title="Contrato")

    row = read_usage(
        conn,
        record_usage(conn, _event(document_id=document_id), client=TEST_CLIENT),
    )

    assert row.document_id == document_id


def test_a_search_row_is_about_a_question_not_a_document(conn: psycopg.Connection):
    row = read_usage(conn, record_usage(conn, _event(), client=TEST_CLIENT))

    assert row.document_id is None


def test_the_database_refuses_an_unattributed_row_even_past_the_writer(conn: psycopg.Connection):
    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_events
                    (client, operation, reason, provider, model, unit, units, unit_price,
                     duration_ms, outcome)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                ("  ", OPERATION_READ, REASON_NEW_DOCUMENT, "google", READER, UNIT_PAGE,
                 1, Decimal("0.01"), 1, OUTCOME_SUCCEEDED),
            )


def test_an_operation_from_an_unpriced_provider_is_still_recorded(conn: psycopg.Connection, wider_price_book):
    book = wider_price_book
    event = priced(
        book,
        operation=OPERATION_READ,
        reason=REASON_NEW_DOCUMENT,
        provider="fornecedor-novo",
        model="parser-v1",
        unit=UNIT_PAGE,
        units=412,
        duration_ms=900,
    )

    assert event.priced is False
    assert event.cost is None

    row = read_usage(conn, record_usage(conn, event, client=TEST_CLIENT))

    assert row.units == Decimal("412")
    assert row.unit == UNIT_PAGE
    assert row.priced is False
    assert row.unit_price is None
    assert row.cost is None


def test_a_known_price_still_lands_on_the_row(conn: psycopg.Connection, wider_price_book):
    book = wider_price_book
    event = priced(
        book,
        operation=OPERATION_READ,
        reason=REASON_NEW_DOCUMENT,
        provider="google",
        model=READER,
        unit=UNIT_PAGE,
        units=412,
        duration_ms=900,
    )

    row = read_usage(conn, record_usage(conn, event, client=TEST_CLIENT))

    assert row.priced is True
    assert row.unit_price == Decimal("0.01")
    assert row.cost == Decimal("4.12")


def test_a_negative_price_is_still_refused_even_though_none_is_allowed():
    with pytest.raises(UsageError):
        _event(unit_price=Decimal("-0.01"))


def test_the_units_are_never_optional(conn: psycopg.Connection):
    with pytest.raises(psycopg.errors.NotNullViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_events
                    (client, operation, reason, provider, model, unit, units, unit_price,
                     duration_ms, outcome)
                VALUES (%s, %s, %s, %s, %s, %s, NULL, %s, %s, %s)
                """,
                (TEST_CLIENT, OPERATION_READ, REASON_NEW_DOCUMENT, "google", READER, UNIT_PAGE,
                 Decimal("0.01"), 1, OUTCOME_SUCCEEDED),
            )
