
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from src.analytics.corpus import never_returned
from src.analytics.cost import (
    cost_by_reason,
    cost_of_unchanged_content,
    total_cost,
)
from src.analytics.latency import largest_step, latency_by_index_effort
from src.analytics.rollup import run_rollup, search_aggregate, usage_aggregate
from src.config import SearchSettings
from src.otel import CLIENT_ATTRIBUTE, CollectedSpans, Telemetry
from src.prices import (
    BILLABLE_OPERATIONS,
    OPERATION_EMBED,
    OPERATION_READ,
    OPERATION_CHUNK,
    OPERATION_RERANK,
    OPERATION_SEARCH,
    UNIT_PAGE,
    UNIT_REQUEST,
    UNIT_TOKEN,
    PriceBook,
)
from src.search import SearchRequest, search_on
from src.search_events import (
    SEARCH_OUTCOME_FOUND,
    SEARCH_OUTCOME_NOTHING_MATCHED,
    SearchEvent,
    SearchEventError,
    question_digest,
    read_search,
    record_search,
)
from src.search_result import Found
from src.store_versions import activate_version
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_OPERATOR_REQUESTED,
    REASON_QUESTION_ASKED,
    REASON_WATERMARK_MOVED,
    UsageError,
    priced,
    read_usage,
    record_usage,
)
from tests.conftest import (
    configured_amount,
    TEST_CLIENT,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    configured_prices,
    direction,
    index_corpus,
)

READER = "pretrained-layout-parser-v1.0-2024-06-03"
EMBED_MODEL = "text-embedding-3-large"

CONTRACT = DocumentSpec(
    "rescisao",
    "Contrato de locação — Sala 402",
    (
        ChunkSpec(
            0,
            "Cláusula quinta: da rescisão antecipada do contrato pelo locatário",
            degrees=0,
            pages=(4, 5),
            headings=("Cláusula Quinta", "Rescisão antecipada"),
        ),
    ),
)


def _read_row(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    reason: str,
    units: int = 100,
    outcome: str = OUTCOME_SUCCEEDED,
    document_id: int | None = None,
    at: datetime | None = None,
    duration_ms: int = 4200,
) -> int:
    return record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_READ,
            reason=reason,
            provider="google",
            model=READER,
            unit=UNIT_PAGE,
            units=units,
            duration_ms=duration_ms,
            outcome=outcome,
            document_id=document_id,
            occurred_at=at,
        ),
        client=TEST_CLIENT,
    )


def test_ac_160(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-160: Every billable operation writes a record with its reason @spec:AC-160"""
    operations = (
        (OPERATION_READ, "google", READER, UNIT_PAGE, 412, REASON_NEW_DOCUMENT, OUTCOME_SUCCEEDED),
        (OPERATION_CHUNK, "google", READER, UNIT_PAGE, 412, REASON_CONFIGURATION_CHANGED, OUTCOME_SUCCEEDED),
        (OPERATION_EMBED, "openai", EMBED_MODEL, UNIT_TOKEN, 91_000, REASON_CONTENT_CHANGED, OUTCOME_SUCCEEDED),
        (OPERATION_RERANK, "self-hosted", "zerank-2", UNIT_REQUEST, 1, REASON_QUESTION_ASKED, OUTCOME_SUCCEEDED),
        (OPERATION_SEARCH, "internal", "postgres-pgvector", UNIT_REQUEST, 1, REASON_QUESTION_ASKED, OUTCOME_SUCCEEDED),
        (OPERATION_READ, "google", READER, UNIT_PAGE, 200, REASON_WATERMARK_MOVED, OUTCOME_FAILED),
    )

    for operation, provider, model, unit, units, reason, outcome in operations:
        event = priced(
            wider_price_book,
            operation=operation,
            reason=reason,
            provider=provider,
            model=model,
            unit=unit,
            units=units,
            duration_ms=1500,
            outcome=outcome,
        )
        usage_id = record_usage(conn, event, client=TEST_CLIENT)

        row = read_usage(conn, usage_id)
        assert row is not None, f"{operation} wrote no record"
        assert row.operation == operation
        assert row.units == Decimal(units)
        assert row.unit == unit
        expected_price = wider_price_book.price_for(
            operation=operation, provider=provider, model=model, unit=unit
        )
        assert row.unit_price == expected_price.amount
        assert row.cost == expected_price.amount * Decimal(units)
        assert row.provider == provider
        assert row.model == model
        assert row.duration_ms == Decimal(1500)
        assert row.outcome == outcome
        assert row.reason == reason

    five_cases = (
        REASON_NEW_DOCUMENT,
        REASON_CONTENT_CHANGED,
        REASON_CONFIGURATION_CHANGED,
        REASON_WATERMARK_MOVED,
        REASON_OPERATOR_REQUESTED,
    )
    written = {reason: _read_row(conn, wider_price_book, reason=reason) for reason in five_cases}
    assert {read_usage(conn, usage_id).reason for usage_id in written.values()} == set(five_cases)

    with pytest.raises(UsageError) as excinfo:
        _read_row(conn, wider_price_book, reason="just because")
    assert "just because" in str(excinfo.value)

    unpriced_id = record_usage(
        conn,
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason=REASON_NEW_DOCUMENT,
            provider="fornecedor-novo",
            model="parser-v1",
            unit=UNIT_PAGE,
            units=412,
            duration_ms=1500,
        ),
        client=TEST_CLIENT,
    )
    unpriced = read_usage(conn, unpriced_id)
    assert unpriced.units == Decimal("412")
    assert unpriced.unit == UNIT_PAGE
    assert unpriced.reason == REASON_NEW_DOCUMENT
    assert unpriced.duration_ms == Decimal("1500")
    assert unpriced.priced is False
    assert unpriced.unit_price is None
    assert unpriced.cost is None


def test_ac_161(conn: psycopg.Connection, wider_price_book):
    """AC-161: Cost is computed from the price stored on the record @spec:AC-161"""
    august = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)
    september = datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc)

    before = wider_price_book
    old_price = before.price_for(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    ).amount
    assert old_price == configured_amount(operation="read", provider="google", model="any", unit="page")
    _read_row(conn, before, reason=REASON_NEW_DOCUMENT, units=100, at=august)

    august_total_when_it_happened = total_cost(
        conn, since=august - timedelta(days=1), until=august + timedelta(days=1)
    )
    assert august_total_when_it_happened.cost == Decimal("1.00")
    assert august_total_when_it_happened.complete

    after = PriceBook([
        *(entry for entry in configured_prices() if entry["operation"] != OPERATION_READ),
        {"operation": OPERATION_READ, "provider": "google", "unit": UNIT_PAGE, "amount": "0.05"},
    ])
    assert after.price_for(
        operation=OPERATION_READ, provider="google", model=READER, unit=UNIT_PAGE
    ).amount == Decimal("0.05")
    _read_row(conn, after, reason=REASON_NEW_DOCUMENT, units=100, at=september)

    recomputed = total_cost(conn, since=august - timedelta(days=1), until=august + timedelta(days=1))

    august_rows = [
        row
        for row in (read_usage(conn, usage_id) for usage_id in _ids(conn))
        if row.occurred_at.month == 8
    ]
    assert [row.unit_price for row in august_rows] == [configured_amount(operation="read", provider="google", model="any", unit="page")]

    assert recomputed.cost == august_total_when_it_happened.cost == Decimal("1.00")

    assert total_cost(
        conn, since=september - timedelta(days=1), until=september + timedelta(days=1)
    ).cost == Decimal("5.00")


def _ids(conn: psycopg.Connection) -> tuple[int, ...]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM usage_events ORDER BY id")
        return tuple(int(row[0]) for row in cursor.fetchall())


def test_ac_162(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-162: The cost of reprocessing content that did not change is answerable @spec:AC-162"""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=30)

    _read_row(conn, wider_price_book, reason=REASON_NEW_DOCUMENT, units=400, at=start + timedelta(days=1))
    _read_row(conn, wider_price_book, reason=REASON_CONTENT_CHANGED, units=150, at=start + timedelta(days=3))

    for day in (5, 12, 22):
        moment = start + timedelta(days=day)
        _read_row(conn, wider_price_book, reason=REASON_WATERMARK_MOVED, units=120, at=moment)
        record_usage(
            conn,
            priced(
                wider_price_book,
                operation=OPERATION_CHUNK,
                reason=REASON_WATERMARK_MOVED,
                provider="google",
                model=READER,
                unit=UNIT_PAGE,
                units=120,
                duration_ms=800,
                occurred_at=moment,
            ),
            client=TEST_CLIENT,
        )

    figure = cost_of_unchanged_content(conn, since=start, until=end)

    assert isinstance(figure.cost, Decimal)

    assert figure.cost == Decimal("3.6") + Decimal("0.0072")
    assert figure.complete

    by_reason = {slice_.label: slice_.cost for slice_ in cost_by_reason(conn, since=start, until=end)}
    assert by_reason[REASON_WATERMARK_MOVED] == figure.cost
    assert by_reason[REASON_NEW_DOCUMENT] == Decimal("4.00")
    assert by_reason[REASON_CONTENT_CHANGED] == Decimal("1.50")
    assert figure.cost < by_reason[REASON_NEW_DOCUMENT]

    record_usage(
        conn,
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason=REASON_WATERMARK_MOVED,
            provider="fornecedor-novo",
            model="parser-v1",
            unit=UNIT_PAGE,
            units=500,
            duration_ms=900,
            occurred_at=start + timedelta(days=25),
        ),
        client=TEST_CLIENT,
    )
    with_a_gap = cost_of_unchanged_content(conn, since=start, until=end)
    assert with_a_gap.cost == figure.cost
    assert with_a_gap.complete is False
    assert with_a_gap.unpriced_units == Decimal("500")
    assert with_a_gap.unpriced_events == 1


def test_ac_163(conn: psycopg.Connection):
    """AC-163: Every search records what it returned @spec:AC-163"""
    written = index_corpus(conn, [CONTRACT])
    activate_version(conn, TEST_STAMP)
    question = "rescisão   antecipada"
    deployment = SearchSettings(index_search_effort=137, dense_weight=0.7, lexical_weight=0.3)
    outcome = search_on(
        conn,
        SearchRequest(question=question, query_vector=direction(0), result_count=3),
        defaults=deployment,
    )
    assert isinstance(outcome, Found)

    event_id = record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question=question,
            dense_weight=outcome.diagnostics.weights.dense,
            lexical_weight=outcome.diagnostics.weights.lexical,
            candidate_depth=outcome.diagnostics.candidate_depth,
            result_count=outcome.diagnostics.result_count,
            index_search_effort=outcome.diagnostics.index_search_effort,
            duration_ms=outcome.diagnostics.duration_ms,
            outcome=SEARCH_OUTCOME_FOUND,
            document_ids=[hit.provenance.document_id for hit in outcome.hits],
            chunk_ids=[hit.chunk_id for hit in outcome.hits],
        ),
        client=TEST_CLIENT,
    )

    row = read_search(conn, event_id)

    assert row is not None
    assert row.caller == "assistente-juridico"
    assert row.question_digest == question_digest(question)
    assert row.dense_weight == Decimal("0.7")
    assert row.lexical_weight == Decimal("0.3")
    assert row.document_ids == (written["rescisao"],)
    assert row.chunk_ids == tuple(hit.chunk_id for hit in outcome.hits)
    assert row.index_search_effort == 137
    assert row.duration_ms > 0

    assert row.question_text is None

    enabled = record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question=question,
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=12,
            outcome=SEARCH_OUTCOME_NOTHING_MATCHED,
        ),
        client=TEST_CLIENT,
        record_question_text=True,
    )
    assert read_search(conn, enabled).question_text == question

    assert question_digest("  Rescisão Antecipada  ") == question_digest("rescisão antecipada")


def test_ac_164(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-164: Documents never retrieved are identifiable @spec:AC-164"""
    consulted = DocumentSpec("consultado", "Contrato consultado", (ChunkSpec(0, "cláusula", pages=(1, 1)),))
    expensive = DocumentSpec("caro", "Laudo de 400 páginas", (ChunkSpec(0, "laudo técnico", pages=(1, 1)),))
    cheap = DocumentSpec("barato", "Memorando de duas páginas", (ChunkSpec(0, "memorando", pages=(1, 1)),))
    written = index_corpus(conn, [consulted, expensive, cheap])

    for external_id, pages in (("consultado", 30), ("caro", 400), ("barato", 2)):
        _read_row(
            conn,
            wider_price_book,
            reason=REASON_NEW_DOCUMENT,
            units=pages,
            document_id=written[external_id],
        )

    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question="cláusula de rescisão",
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=31,
            outcome=SEARCH_OUTCOME_FOUND,
            document_ids=[written["consultado"]],
            chunk_ids=[1],
            occurred_at=start + timedelta(days=2),
        ),
        client=TEST_CLIENT,
    )

    unused = never_returned(conn, since=start, until=end)

    assert [document.external_id for document in unused] == ["caro", "barato"]
    assert {document.external_id: document.cost for document in unused} == {
        "caro": Decimal("4.00"),
        "barato": Decimal("0.02"),
    }

    assert [document.cost for document in unused] == sorted(
        (document.cost for document in unused), reverse=True
    )

    assert "consultado" not in {document.external_id for document in unused}

    earlier = never_returned(conn, since=start - timedelta(days=30), until=start)
    assert {document.external_id for document in earlier} == {"consultado", "caro", "barato"}


def test_ac_165(conn: psycopg.Connection):
    """AC-165: Latency is reportable by index effort @spec:AC-165"""
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    latencies = {
        4: (900, 950, 1000, 1100, 4800),
        40: (30, 32, 35, 38, 60),
        200: (48, 52, 55, 58, 90),
    }
    for effort, durations in latencies.items():
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
                    occurred_at=start + timedelta(hours=index),
                ),
                client=TEST_CLIENT,
            )

    groups = latency_by_index_effort(conn, since=start, until=end)

    assert [group.index_search_effort for group in groups] == [4, 40, 200]
    assert all(group.searches == 5 for group in groups)
    by_effort = {group.index_search_effort: group for group in groups}
    assert by_effort[4].high_percentile_ms > by_effort[40].high_percentile_ms
    assert by_effort[4].high_percentile_ms > by_effort[4].median_ms

    step = largest_step(groups)
    assert step is not None
    assert step.below.index_search_effort == 4
    assert step.effort_floor == 40
    assert step.factor > 10

    higher = largest_step(groups[1:])
    assert higher is not None
    assert higher.factor < 2


def test_ac_166(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-166: Records are rolled up daily and the raw rows are pruned @spec:AC-166"""
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    retention_days = 30
    old = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    recent = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)

    for units, outcome in ((100, OUTCOME_SUCCEEDED), (250, OUTCOME_SUCCEEDED), (40, OUTCOME_FAILED)):
        _read_row(
            conn,
            wider_price_book,
            reason=REASON_NEW_DOCUMENT,
            units=units,
            outcome=outcome,
            at=old,
            duration_ms=units * 10,
        )
    kept_id = _read_row(conn, wider_price_book, reason=REASON_CONTENT_CHANGED, units=10, at=recent)

    for duration in (20, 40, 90):
        record_search(
            conn,
            SearchEvent(
                caller="assistente-juridico",
                question=f"pergunta antiga {duration}",
                dense_weight=0.5,
                lexical_weight=0.5,
                candidate_depth=50,
                result_count=10,
                index_search_effort=40,
                duration_ms=duration,
                outcome=SEARCH_OUTCOME_FOUND,
                document_ids=[1, 2],
                chunk_ids=[1],
                occurred_at=old,
            ),
            client=TEST_CLIENT,
        )

    result = run_rollup(conn, retention_days=retention_days, now=now)

    august_first = date(2026, 8, 1)
    aggregate = usage_aggregate(conn, august_first)
    assert len(aggregate) == 1
    day = aggregate[0]
    assert day.client == TEST_CLIENT
    assert day.operation == OPERATION_READ
    assert day.reason == REASON_NEW_DOCUMENT
    assert day.events == 3
    assert day.units == Decimal("390")
    assert day.cost == Decimal("3.90")
    assert day.errors == 1
    assert day.complete
    assert day.unpriced_units == Decimal("0")
    assert day.duration_p50_ms == Decimal("1000")
    assert day.duration_p95_ms >= day.duration_p50_ms

    searches = search_aggregate(conn, august_first)
    assert len(searches) == 1
    assert searches[0].searches == 3
    assert searches[0].documents_returned == 6
    assert searches[0].duration_p50_ms == Decimal("40")

    assert result.usage_rows_pruned == 3
    assert result.search_rows_pruned == 3
    assert result.oldest_retained_day == date(2026, 8, 31)
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT count(*) FROM search_events")
        assert cursor.fetchone()[0] == 0
    assert read_usage(conn, kept_id) is not None
    assert usage_aggregate(conn, date(2026, 9, 29))
    assert usage_aggregate(conn, august_first)


def test_ac_167(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-167: The rollup is idempotent @spec:AC-167"""
    now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    yesterday = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    for units in (100, 250):
        _read_row(conn, wider_price_book, reason=REASON_NEW_DOCUMENT, units=units, at=yesterday)
    record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question="pergunta de ontem",
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=44,
            outcome=SEARCH_OUTCOME_FOUND,
            document_ids=[7],
            chunk_ids=[7],
            occurred_at=yesterday,
        ),
        client=TEST_CLIENT,
    )

    run_rollup(conn, retention_days=30, now=now)
    day = date(2026, 9, 29)
    first = usage_aggregate(conn, day)
    first_searches = search_aggregate(conn, day)
    assert first[0].events == 2
    assert first[0].units == Decimal("350")

    run_rollup(conn, retention_days=30, now=now)

    second = usage_aggregate(conn, day)
    assert len(second) == len(first) == 1
    assert second[0].events == 2
    assert second[0].units == Decimal("350")
    assert second[0].cost == first[0].cost
    assert search_aggregate(conn, day)[0].searches == first_searches[0].searches == 1

    _read_row(conn, wider_price_book, reason=REASON_NEW_DOCUMENT, units=50, at=yesterday)
    run_rollup(conn, retention_days=30, now=now)
    third = usage_aggregate(conn, day)
    assert len(third) == 1
    assert third[0].events == 3
    assert third[0].units == Decimal("400")


def test_ac_168(conn: psycopg.Connection, wider_price_book: PriceBook):
    """AC-168: Every deployment's records identify their client @spec:AC-168"""
    usage_id = _read_row(conn, wider_price_book, reason=REASON_NEW_DOCUMENT)
    search_id = record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question="quais os prazos de rescisão",
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=27,
            outcome=SEARCH_OUTCOME_NOTHING_MATCHED,
        ),
        client=TEST_CLIENT,
    )
    collected = CollectedSpans()
    telemetry = Telemetry(client=TEST_CLIENT, endpoint="http://collector.internal:4318", exporter=collected)
    with telemetry.span("search", caller="assistente-juridico"):
        pass

    assert read_usage(conn, usage_id).client == TEST_CLIENT
    assert read_search(conn, search_id).client == TEST_CLIENT
    assert len(collected.spans) == 1
    assert collected.spans[0].client == TEST_CLIENT
    assert collected.spans[0].attributes[CLIENT_ATTRIBUTE] == TEST_CLIENT

    other = "segundo-escritorio"
    record_usage(
        conn,
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason=REASON_OPERATOR_REQUESTED,
            provider="google",
            model=READER,
            unit=UNIT_PAGE,
            units=250,
            duration_ms=900,
        ),
        client=other,
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT client, sum(cost) FROM usage_events GROUP BY client ORDER BY client")
        assert cursor.fetchall() == [(TEST_CLIENT, Decimal("1.00")), (other, Decimal("2.50"))]

    unattributed = priced(
        wider_price_book,
        operation=OPERATION_READ,
        reason=REASON_NEW_DOCUMENT,
        provider="google",
        model=READER,
        unit=UNIT_PAGE,
        units=1,
        duration_ms=1,
    )
    with pytest.raises(UsageError):
        record_usage(conn, unattributed, client="   ")
    with pytest.raises(SearchEventError):
        record_search(
            conn,
            SearchEvent(
                caller="assistente-juridico",
                question="pergunta sem dono",
                dense_weight=0.5,
                lexical_weight=0.5,
                candidate_depth=50,
                result_count=10,
                index_search_effort=40,
                duration_ms=5,
                outcome=SEARCH_OUTCOME_NOTHING_MATCHED,
            ),
            client="",
        )
    with pytest.raises(ValueError):
        Telemetry(client="")


def test_p_010(conn: psycopg.Connection, wider_price_book: PriceBook):
    billable = (
        (OPERATION_READ, "google", READER, UNIT_PAGE, 412, REASON_CONTENT_CHANGED),
        (OPERATION_CHUNK, "google", READER, UNIT_PAGE, 412, REASON_CONFIGURATION_CHANGED),
        (OPERATION_EMBED, "openai", EMBED_MODEL, UNIT_TOKEN, 91_000, REASON_CONFIGURATION_CHANGED),
        (OPERATION_RERANK, "self-hosted", "zerank-2", UNIT_REQUEST, 1, REASON_QUESTION_ASKED),
        (OPERATION_SEARCH, "internal", "postgres-pgvector", UNIT_REQUEST, 1, REASON_QUESTION_ASKED),
    )
    assert {operation for operation, *_ in billable} == set(BILLABLE_OPERATIONS)

    for operation, provider, model, unit, units, reason in billable:
        row = read_usage(
            conn,
            record_usage(
                conn,
                priced(
                    wider_price_book,
                    operation=operation,
                    reason=reason,
                    provider=provider,
                    model=model,
                    unit=unit,
                    units=units,
                    duration_ms=250,
                ),
                client=TEST_CLIENT,
            ),
        )
        assert row.units == Decimal(units), operation
        assert row.unit_price >= 0, operation
        assert row.reason == reason, operation

    with pytest.raises(UsageError):
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason="",
            provider="google",
            model=READER,
            unit=UNIT_PAGE,
            units=1,
            duration_ms=1,
        )
    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_events
                    (client, operation, reason, provider, model, unit, units, unit_price,
                     duration_ms, outcome)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (TEST_CLIENT, OPERATION_READ, "porque sim", "google", READER, UNIT_PAGE,
                 1, Decimal("0.01"), 1, OUTCOME_SUCCEEDED),
            )
