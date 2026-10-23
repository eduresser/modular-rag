"""Unit-level coverage for src/analytics/cost.py.

AC-161 and AC-162 are proven in tests/test_spec_usage_observability.py — see
CLAUDE.md for why the proofs live there. This file covers the other named
queries, the empty window, and the currency boundary.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg

from src.analytics.cost import (
    cost_by_operation,
    cost_by_provider,
    cost_by_reason,
    cost_of_unchanged_content,
    total_cost,
    unchanged_content_breakdown,
)
from src.prices import OPERATION_EMBED, OPERATION_READ, OPERATION_CHUNK, UNIT_PAGE, UNIT_TOKEN, PriceBook
from src.usage import (
    OUTCOME_FAILED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_WATERMARK_MOVED,
    priced,
    record_usage,
)
from tests.conftest import TEST_CLIENT

READER = "pretrained-layout-parser-v1.0-2024-06-03"
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(days=30)


def _write(conn, book, *, operation, reason, provider, model, unit, units, outcome=None, at=None):
    return record_usage(
        conn,
        priced(
            book,
            operation=operation,
            reason=reason,
            provider=provider,
            model=model,
            unit=unit,
            units=units,
            duration_ms=100,
            outcome=outcome or "succeeded",
            occurred_at=at or START + timedelta(days=1),
        ),
        client=TEST_CLIENT,
    )


def _a_month(conn: psycopg.Connection, book: PriceBook) -> None:
    """Reading, re-cutting and embedding, for three different reasons."""
    _write(conn, book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="google", model=READER, unit=UNIT_PAGE, units=400)
    _write(conn, book, operation=OPERATION_READ, reason=REASON_WATERMARK_MOVED,
           provider="google", model=READER, unit=UNIT_PAGE, units=120)
    _write(conn, book, operation=OPERATION_CHUNK, reason=REASON_WATERMARK_MOVED,
           provider="google", model=READER, unit=UNIT_PAGE, units=120)
    _write(conn, book, operation=OPERATION_EMBED, reason=REASON_CONTENT_CHANGED,
           provider="openai", model="text-embedding-3-large", unit=UNIT_TOKEN, units=1_000_000)
    _write(conn, book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="docling", model="docling-2", unit=UNIT_PAGE, units=1_000,
           outcome=OUTCOME_FAILED)


def test_an_empty_window_totals_zero_rather_than_nothing(conn: psycopg.Connection):
    # A caller that has to distinguish "no spend" from "no answer" gets it
    # wrong; there is no such thing as an unknown total here.
    empty = total_cost(conn, since=START, until=END)
    assert empty.cost == Decimal("0")
    assert empty.units == Decimal("0")
    # Nothing spent is complete information, not a gap.
    assert empty.complete
    assert cost_of_unchanged_content(conn, since=START, until=END).cost == Decimal("0")
    assert cost_by_reason(conn, since=START, until=END) == ()


def test_cost_by_reason_puts_the_most_expensive_line_first(conn: psycopg.Connection, wider_price_book):
    _a_month(conn, wider_price_book)

    slices = cost_by_reason(conn, since=START, until=END)

    assert [slice_.label for slice_ in slices] == [
        REASON_NEW_DOCUMENT,   # 400 pages read at one rate, 1000 at another
        REASON_WATERMARK_MOVED,  # 120 pages at 0.01, plus the re-cut
        REASON_CONTENT_CHANGED,  # 1M tokens at 0.13/1M
    ]
    assert [slice_.cost for slice_ in slices] == sorted(
        (slice_.cost for slice_ in slices), reverse=True
    )


def test_cost_by_provider_is_the_invoice_view(conn: psycopg.Connection, wider_price_book):
    _a_month(conn, wider_price_book)

    by_provider = {slice_.label: slice_ for slice_ in cost_by_provider(conn, since=START, until=END)}

    def rate(operation, provider, model, unit):
        return wider_price_book.price_for(
            operation=operation, provider=provider, model=model, unit=unit
        ).amount

    # The figures come from the book rather than from products written out here,
    # so this stays a test about the INVOICE VIEW — one row per who is paid —
    # rather than a second copy of the price book.
    read = rate(OPERATION_READ, "google", READER, UNIT_PAGE)
    chunk = rate(OPERATION_CHUNK, "google", READER, UNIT_PAGE)
    other = rate(OPERATION_EMBED, "openai", "text-embedding-3-large", UNIT_TOKEN)
    local = rate(OPERATION_READ, "docling", "docling-2", UNIT_PAGE)

    assert set(by_provider) == {"google", "openai", "docling"}
    assert by_provider["google"].cost == read * 520 + chunk * 120
    assert by_provider["openai"].cost == other * 1_000_000
    assert by_provider["docling"].cost == local * 1_000


def test_cost_by_operation_shows_which_step_spent_it(conn: psycopg.Connection, wider_price_book):
    _a_month(conn, wider_price_book)

    by_operation = {slice_.label: slice_ for slice_ in cost_by_operation(conn, since=START, until=END)}

    assert set(by_operation) == {OPERATION_READ, OPERATION_CHUNK, OPERATION_EMBED}
    assert by_operation[OPERATION_READ].events == 3
    assert by_operation[OPERATION_CHUNK].cost == Decimal("0.0024")


def test_a_failure_is_counted_in_its_slice_and_still_costs(conn: psycopg.Connection, wider_price_book):
    # A read that burned its pages and then failed cost what it cost.
    _a_month(conn, wider_price_book)

    by_provider = {slice_.label: slice_ for slice_ in cost_by_provider(conn, since=START, until=END)}

    assert by_provider["docling"].failures == 1
    assert by_provider["docling"].cost > 0
    assert by_provider["google"].failures == 0


def test_the_units_are_summed_alongside_the_cost(conn: psycopg.Connection, wider_price_book):
    _a_month(conn, wider_price_book)

    by_operation = {slice_.label: slice_ for slice_ in cost_by_operation(conn, since=START, until=END)}

    assert by_operation[OPERATION_EMBED].units == Decimal("1000000")
    assert by_operation[OPERATION_READ].units == Decimal("1520")


def test_the_window_excludes_what_falls_outside_it(conn: psycopg.Connection, wider_price_book):
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="google", model=READER, unit=UNIT_PAGE, units=100,
           at=START - timedelta(days=1))
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="google", model=READER, unit=UNIT_PAGE, units=100,
           at=START + timedelta(days=1))

    assert total_cost(conn, since=START, until=END).cost == Decimal("1.00")


def test_a_total_is_asked_for_in_one_currency(conn: psycopg.Connection, wider_price_book):
    # Vendor prices are quoted in dollars and the business is Brazilian; adding
    # the two into one number is wrong in a way nobody notices until the
    # invoice disagrees.
    _a_month(conn, wider_price_book)

    assert total_cost(conn, since=START, until=END, currency="USD").cost > 0
    assert total_cost(conn, since=START, until=END, currency="BRL").cost == Decimal("0")


def test_the_unchanged_content_breakdown_shows_reading_dominating(conn: psycopg.Connection, wider_price_book):
    # Which is what tells an operator the saving is in avoiding the read, not
    # in a cheaper embedding model.
    _a_month(conn, wider_price_book)

    breakdown = unchanged_content_breakdown(conn, since=START, until=END)

    assert [slice_.label for slice_ in breakdown] == [OPERATION_READ, OPERATION_CHUNK]
    assert breakdown[0].cost == Decimal("1.20")
    assert breakdown[1].cost == Decimal("0.0024")
    # The breakdown and the single figure are the same money.
    assert sum(slice_.cost for slice_ in breakdown) == cost_of_unchanged_content(
        conn, since=START, until=END
    ).cost


def test_a_total_says_how_much_of_it_could_not_be_priced(conn: psycopg.Connection, wider_price_book):
    # sum() skips nulls without a word, so a month in which a new provider was
    # never priced would total as though that provider had been free (ASM-075).
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="google", model=READER, unit=UNIT_PAGE, units=400)
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="fornecedor-novo", model="parser-v1", unit=UNIT_PAGE, units=1_000)

    figure = total_cost(conn, since=START, until=END)

    assert figure.cost == Decimal("4.00")
    assert figure.units == Decimal("1400")
    assert figure.unpriced_units == Decimal("1000")
    assert figure.unpriced_events == 1
    assert figure.complete is False


def test_a_slice_says_which_of_its_units_it_could_not_price(conn: psycopg.Connection, wider_price_book):
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_NEW_DOCUMENT,
           provider="google", model=READER, unit=UNIT_PAGE, units=400)
    _write(conn, wider_price_book, operation=OPERATION_READ, reason=REASON_WATERMARK_MOVED,
           provider="fornecedor-novo", model="parser-v1", unit=UNIT_PAGE, units=1_000)

    by_reason = {slice_.label: slice_ for slice_ in cost_by_reason(conn, since=START, until=END)}

    assert by_reason[REASON_NEW_DOCUMENT].complete
    assert by_reason[REASON_WATERMARK_MOVED].complete is False
    assert by_reason[REASON_WATERMARK_MOVED].unpriced_units == Decimal("1000")
    # The unpriced line still shows its units, so it is not invisible just
    # because it sorted to the bottom on a cost of zero.
    assert by_reason[REASON_WATERMARK_MOVED].units == Decimal("1000")
    assert by_reason[REASON_WATERMARK_MOVED].cost == Decimal("0")


def test_a_deployment_that_configured_no_prices_reports_zero_and_says_so(
    conn: psycopg.Connection, empty_price_book: PriceBook
):
    # There is no built-in price table any more (ASM-075), so this is the state
    # a fresh deployment is in until someone fills in RAG_USAGE_PRICES. It must
    # read as "nobody priced this", never as "this was free" — and the units
    # are still all there, because the units were never the uncertain part.
    _a_month(conn, empty_price_book)

    figure = total_cost(conn, since=START, until=END)

    assert figure.cost == Decimal("0")
    assert figure.complete is False
    assert figure.unpriced_events == 5
    assert figure.units == Decimal("1000000") + Decimal("400") + Decimal("120") * 2 + Decimal("1000")

    # And every slice of it says the same, rather than one of them looking
    # settled because it happened to sort first.
    assert all(not slice_.complete for slice_ in cost_by_reason(conn, since=START, until=END))
