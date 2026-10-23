"""Unit-level coverage for src/analytics/corpus.py.

AC-164 is proven in tests/test_spec_usage_observability.py — see CLAUDE.md for
why the proof lives there. AC-208 is proven in
tests/test_spec_read_fidelity.py::test_ac_208. This file covers the edges of
"never returned" (a document that is gone, a document with no cost history
left, and the two different windows the query uses on purpose) and the edges
of "refused for want of a locator" (a failure for another reason, and one
already fixed by a later read).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg

from src.analytics.corpus import never_returned, refused_for_want_of_locator, returned_document_ids
from src.prices import OPERATION_READ, UNIT_PAGE, PriceBook
from src.processing.artifact import record_read_failure, store_parsed_artifact
from src.processing.to_chunks import NO_LOCATOR_REASON
from src.search_events import SEARCH_OUTCOME_FOUND, SEARCH_OUTCOME_NOTHING_MATCHED, SearchEvent, record_search
from src.store import upsert_document
from src.usage import REASON_NEW_DOCUMENT, priced, record_usage
from tests.conftest import (
    DOCAI_PROCESSOR,
    DOCAI_VERSION,
    TEST_CLIENT,
    ChunkSpec,
    DocumentSpec,
    FakeObjectStorage,
    index_corpus,
)

READER = "pretrained-layout-parser-v1.0-2024-06-03"
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(days=7)


def _document(external_id: str, title: str) -> DocumentSpec:
    return DocumentSpec(external_id, title, (ChunkSpec(0, f"conteúdo de {external_id}", pages=(1, 1)),))


def _cost(conn, book, document_id: int, pages: int) -> None:
    record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_READ,
            reason=REASON_NEW_DOCUMENT,
            provider="google",
            model=READER,
            unit=UNIT_PAGE,
            units=pages,
            duration_ms=100,
            document_id=document_id,
        ),
        client=TEST_CLIENT,
    )


def _search(conn, *, document_ids, at=None, outcome=SEARCH_OUTCOME_FOUND) -> int:
    return record_search(
        conn,
        SearchEvent(
            caller="assistente-juridico",
            question="cláusula de rescisão",
            dense_weight=0.5,
            lexical_weight=0.5,
            candidate_depth=50,
            result_count=10,
            index_search_effort=40,
            duration_ms=30,
            outcome=outcome,
            document_ids=document_ids,
            chunk_ids=[1] if document_ids else [],
            occurred_at=at or START + timedelta(days=1),
        ),
        client=TEST_CLIENT,
    )


def test_an_archive_with_no_search_history_reports_everything_unused(conn: psycopg.Connection):
    written = index_corpus(conn, [_document("a", "A"), _document("b", "B")])

    unused = never_returned(conn, since=START, until=END)

    assert {document.external_id for document in unused} == set(written)


def test_a_document_marked_gone_is_not_reported_as_unused(conn: psycopg.Connection, wider_price_book):
    # It is not costing anything to keep indexed, because it is not indexed.
    written = index_corpus(
        conn,
        [
            _document("vivo", "Ainda indexado"),
            DocumentSpec("removido", "Removido", (ChunkSpec(0, "texto", pages=(1, 1)),), gone=True),
        ],
    )
    _cost(conn, wider_price_book, written["removido"], 900)

    unused = never_returned(conn, since=START, until=END)

    assert [document.external_id for document in unused] == ["vivo"]


def test_a_document_whose_cost_history_was_pruned_still_appears_at_zero(conn: psycopg.Connection):
    # Leaving it out would read as "consulted", which is the opposite of true.
    index_corpus(conn, [_document("antigo", "Indexado antes da retenção")])

    unused = never_returned(conn, since=START, until=END)

    assert [(document.external_id, document.cost, document.operations) for document in unused] == [
        ("antigo", Decimal("0"), 0)
    ]


def test_the_cost_is_all_of_it_while_the_consultation_window_is_the_callers(
    conn: psycopg.Connection, wider_price_book: PriceBook
):
    # Two different windows, deliberately: narrowing the cost to the same
    # window would report a document that was expensive last quarter and idle
    # this one as free.
    written = index_corpus(conn, [_document("caro", "Laudo caro")])
    record_usage(
        conn,
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason=REASON_NEW_DOCUMENT,
            provider="google",
            model=READER,
            unit=UNIT_PAGE,
            units=400,
            duration_ms=100,
            document_id=written["caro"],
            occurred_at=START - timedelta(days=90),
        ),
        client=TEST_CLIENT,
    )

    unused = never_returned(conn, since=START, until=END)

    assert unused[0].cost == Decimal("4.00")


def test_a_search_that_matched_nothing_marks_nothing_as_used(conn: psycopg.Connection):
    written = index_corpus(conn, [_document("a", "A")])
    _search(conn, document_ids=[], outcome=SEARCH_OUTCOME_NOTHING_MATCHED)

    unused = never_returned(conn, since=START, until=END)

    assert [document.document_id for document in unused] == [written["a"]]


def test_the_limit_keeps_the_most_expensive_ones(conn: psycopg.Connection, wider_price_book: PriceBook):
    written = index_corpus(conn, [_document("caro", "Caro"), _document("medio", "Médio"), _document("barato", "Barato")])
    for external_id, pages in (("caro", 400), ("medio", 100), ("barato", 2)):
        _cost(conn, wider_price_book, written[external_id], pages)

    top = never_returned(conn, since=START, until=END, limit=2)

    assert [document.external_id for document in top] == ["caro", "medio"]


def test_the_documents_returned_in_a_window_are_deduplicated_and_ordered(conn: psycopg.Connection):
    written = index_corpus(conn, [_document("a", "A"), _document("b", "B")])
    _search(conn, document_ids=[written["b"], written["a"]])
    _search(conn, document_ids=[written["b"]], at=START + timedelta(days=2))

    assert returned_document_ids(conn, since=START, until=END) == tuple(
        sorted((written["a"], written["b"]))
    )


def test_the_two_queries_are_complements_of_each_other(conn: psycopg.Connection, wider_price_book: PriceBook):
    written = index_corpus(conn, [_document("a", "A"), _document("b", "B"), _document("c", "C")])
    _search(conn, document_ids=[written["a"], written["c"]])

    consulted = set(returned_document_ids(conn, since=START, until=END))
    unused = {document.document_id for document in never_returned(conn, since=START, until=END)}

    assert consulted == {written["a"], written["c"]}
    assert unused == {written["b"]}
    assert consulted & unused == set()


def test_a_search_outside_the_window_does_not_count_as_a_consultation(conn: psycopg.Connection):
    written = index_corpus(conn, [_document("a", "A")])
    _search(conn, document_ids=[written["a"]], at=END + timedelta(days=1))

    unused = never_returned(conn, since=START, until=END)

    assert [document.external_id for document in unused] == ["a"]


def test_a_cost_in_another_currency_is_not_added_to_the_dollar_figure(conn: psycopg.Connection, wider_price_book: PriceBook):
    written = index_corpus(conn, [_document("a", "A")])
    _cost(conn, wider_price_book, written["a"], 400)

    in_dollars = never_returned(conn, since=START, until=END, currency="USD")
    in_reais = never_returned(conn, since=START, until=END, currency="BRL")

    assert in_dollars[0].cost == Decimal("4.00")
    assert in_reais[0].cost == Decimal("0")
    assert in_reais[0].currency == "BRL"


def test_a_document_nobody_priced_says_so_instead_of_sorting_as_cheap(
    conn: psycopg.Connection, wider_price_book: PriceBook
):
    # The ordering is by cost, so a document processed entirely by an unpriced
    # provider sums to zero and sorts last while possibly being the most
    # expensive thing in the archive (ASM-075). The row has to say so.
    written = index_corpus(conn, [_document("caro", "Caro"), _document("sem_preco", "Sem preço")])
    _cost(conn, wider_price_book, written["caro"], 400)
    record_usage(
        conn,
        priced(
            wider_price_book,
            operation=OPERATION_READ,
            reason=REASON_NEW_DOCUMENT,
            provider="fornecedor-novo",
            model="parser-v1",
            unit=UNIT_PAGE,
            units=5_000,
            duration_ms=100,
            document_id=written["sem_preco"],
        ),
        client=TEST_CLIENT,
    )

    unused = {document.external_id: document for document in never_returned(conn, since=START, until=END)}

    assert unused["caro"].cost == Decimal("4.00")
    assert unused["caro"].complete
    # Zero cost, five thousand pages, and the row does not pretend otherwise.
    assert unused["sem_preco"].cost == Decimal("0")
    assert unused["sem_preco"].complete is False
    assert unused["sem_preco"].unpriced_units == Decimal("5000")


def test_a_document_with_no_history_at_all_reads_as_complete(conn: psycopg.Connection):
    # Zero cost and nothing unpriced: already pruned, or never processed. Not
    # the same as a gap in pricing, and it should not look like one.
    index_corpus(conn, [_document("antigo", "Antigo")])

    document = never_returned(conn, since=START, until=END)[0]

    assert document.cost == Decimal("0")
    assert document.complete


# ── refused_for_want_of_locator (AC-208) ───────────────────────────────────
# AC-208 itself is proven end to end, from the raise in
# src/processing/to_chunks.py through to this query, in
# tests/test_spec_read_fidelity.py::test_ac_208. These cover the query's own
# edges: a failure for another reason, and a refusal later fixed by a read.


def _fail(conn, external_id: str, title: str, reason: str) -> int:
    document_id = upsert_document(conn, external_id=external_id, title=title)
    record_read_failure(
        conn, document_id=document_id, reason=reason,
        reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )
    return document_id


def test_a_document_refused_for_want_of_a_locator_is_listed(conn: psycopg.Connection):
    document_id = _fail(
        conn, "sem-localizador", "Sem localizador",
        f"document 'sem-localizador': {NO_LOCATOR_REASON}: none of its 4 pieces carry a "
        "page or a heading (principle P-002)",
    )

    refused = refused_for_want_of_locator(conn)

    assert [item.document_id for item in refused] == [document_id]
    assert refused[0].external_id == "sem-localizador"
    assert NO_LOCATOR_REASON in refused[0].failure_reason


def test_a_failure_for_another_reason_does_not_count_as_want_of_a_locator(
    conn: psycopg.Connection,
):
    _fail(conn, "formato-recusado", "Formato recusado", "has 640 pages, above the 500-page limit")

    assert refused_for_want_of_locator(conn) == ()


def test_a_refusal_later_fixed_by_a_successful_read_is_no_longer_listed(
    conn: psycopg.Connection,
):
    # document_reads is one row per document (AC-208 counts a current verdict,
    # not a history of attempts): a later, successful read replaces the
    # failure rather than adding to it.
    document_id = upsert_document(conn, external_id="corrigido", title="Corrigido")
    record_read_failure(
        conn, document_id=document_id, reason=f"{NO_LOCATOR_REASON}: none of its 2 pieces...",
        reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )
    store_parsed_artifact(
        conn, FakeObjectStorage().store(), document_id=document_id, external_id="corrigido",
        payload={"document": {}}, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )

    assert refused_for_want_of_locator(conn) == ()


def test_documents_refused_for_want_of_a_locator_can_be_listed_together(conn: psycopg.Connection):
    first = _fail(conn, "a", "A", f"{NO_LOCATOR_REASON}: none of its 2 pieces...")
    second = _fail(conn, "b", "B", f"{NO_LOCATOR_REASON}: none of its 3 pieces...")
    _fail(conn, "c", "C", "has 900 pages, above the 500-page limit")

    refused = refused_for_want_of_locator(conn)

    assert {item.document_id for item in refused} == {first, second}
