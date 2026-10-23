from decimal import Decimal

import psycopg
import pytest

from src.prices import OPERATION_READ, OPERATION_CHUNK, UNIT_PAGE, PriceBook
from src.processing.cost import (
    MODEL_LOCAL_RECUT,
    PROVIDER_LOCAL,
    PROVIDER_READER,
    record_read,
    record_recut,
    record_refused_read,
)
from src.processing.recut import RECUT_LOCAL, RECUT_PROVIDER
from src.store import upsert_document
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_QUESTION_ASKED,
    REASON_WATERMARK_MOVED,
    UsageError,
    read_usage,
)
from tests.conftest import configured_amount, DOCAI_VERSION, TEST_CLIENT


def _document(conn: psycopg.Connection, external_id: str = "contrato") -> int:
    return upsert_document(conn, external_id=external_id, title="Contrato")


def test_a_read_records_pages_at_the_configured_rate(conn: psycopg.Connection, price_book: PriceBook):
    document_id = _document(conn)

    row = read_usage(
        conn,
        record_read(
            conn, price_book, client=TEST_CLIENT, document_id=document_id,
            reader_version=DOCAI_VERSION, pages=412, reason=REASON_NEW_DOCUMENT, duration_ms=8_400,
        ),
    )

    assert row.operation == OPERATION_READ
    assert row.provider == PROVIDER_READER
    assert row.model == DOCAI_VERSION
    assert row.unit == UNIT_PAGE
    assert row.units == Decimal(412)
    assert row.reason == REASON_NEW_DOCUMENT
    assert row.document_id == document_id
    assert row.unit_price == configured_amount(operation="read", provider="google", model="any", unit="page")
    assert row.cost == Decimal("4.12")


def test_a_recut_of_the_same_document_costs_nothing_and_a_read_does_not(
    conn: psycopg.Connection, price_book: PriceBook
):
    document_id = _document(conn)
    read = read_usage(
        conn,
        record_read(
            conn, price_book, client=TEST_CLIENT, document_id=document_id,
            reader_version=DOCAI_VERSION, pages=1_000, reason=REASON_NEW_DOCUMENT, duration_ms=1,
        ),
    )
    recut = read_usage(
        conn,
        record_recut(
            conn, price_book, client=TEST_CLIENT, document_id=document_id,
            reader_version=DOCAI_VERSION, pages=1_000, reason=REASON_CONFIGURATION_CHANGED,
            duration_ms=1, path=RECUT_LOCAL,
        ),
    )

    assert recut.operation == OPERATION_CHUNK
    assert recut.provider == PROVIDER_LOCAL
    assert read.cost > 0
    assert recut.cost == 0
    assert recut.unit_price == 0


def test_a_local_recut_is_billed_to_nobody_and_still_leaves_a_row(conn: psycopg.Connection, price_book: PriceBook):
    row = read_usage(
        conn,
        record_recut(
            conn, price_book, client=TEST_CLIENT, document_id=_document(conn),
            reader_version=DOCAI_VERSION, pages=250, reason=REASON_CONFIGURATION_CHANGED,
            duration_ms=90, path=RECUT_LOCAL,
        ),
    )

    assert row.provider == PROVIDER_LOCAL
    assert row.model == MODEL_LOCAL_RECUT
    assert row.units == Decimal(250)
    assert row.unit_price == Decimal(0)
    assert row.cost == Decimal(0)


def test_a_failed_read_still_writes_a_row(conn: psycopg.Connection, price_book: PriceBook):
    row = read_usage(
        conn,
        record_read(
            conn, price_book, client=TEST_CLIENT, document_id=_document(conn),
            reader_version=DOCAI_VERSION, pages=200, reason=REASON_CONTENT_CHANGED,
            duration_ms=30_000, outcome=OUTCOME_FAILED,
        ),
    )

    assert row.outcome == OUTCOME_FAILED
    assert row.cost == Decimal("2.00")


def test_a_document_refused_before_submission_costs_nothing_and_is_still_recorded(
    conn: psycopg.Connection, price_book: PriceBook
):
    row = read_usage(
        conn,
        record_refused_read(
            conn, price_book, client=TEST_CLIENT, document_id=_document(conn),
            reader_version=DOCAI_VERSION, reason=REASON_NEW_DOCUMENT,
        ),
    )

    assert row.units == Decimal(0)
    assert row.cost == Decimal(0)
    assert row.outcome == OUTCOME_FAILED


def test_a_read_never_happens_because_somebody_asked_a_question(conn: psycopg.Connection, price_book: PriceBook):
    for recorder in (record_read, record_recut):
        with pytest.raises(UsageError, match="question"):
            recorder(
                conn, price_book, client=TEST_CLIENT, document_id=None,
                reader_version=DOCAI_VERSION, pages=1, reason=REASON_QUESTION_ASKED, duration_ms=1,
            )


def test_a_recut_by_no_recognised_path_is_refused(conn: psycopg.Connection, price_book: PriceBook):
    with pytest.raises(UsageError, match="locally or at the provider"):
        record_recut(
            conn, price_book, client=TEST_CLIENT, document_id=None,
            reader_version=DOCAI_VERSION, pages=1, reason=REASON_CONFIGURATION_CHANGED,
            duration_ms=1, path="magia",
        )


def test_a_deployment_that_has_not_been_told_its_rates_records_the_pages_anyway(
    conn: psycopg.Connection, empty_price_book: PriceBook
):
    row = read_usage(
        conn,
        record_read(
            conn, empty_price_book, client=TEST_CLIENT, document_id=_document(conn),
            reader_version=DOCAI_VERSION, pages=77, reason=REASON_WATERMARK_MOVED, duration_ms=5,
        ),
    )

    assert row.units == Decimal(77)
    assert row.unit_price is None
    assert row.cost is None
    assert row.outcome == OUTCOME_SUCCEEDED
