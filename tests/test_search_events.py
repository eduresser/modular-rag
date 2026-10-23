from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
import pytest

from src.search_events import (
    SEARCH_OUTCOME_FAILED,
    SEARCH_OUTCOME_FOUND,
    SEARCH_OUTCOME_NOTHING_MATCHED,
    SearchEvent,
    SearchEventError,
    question_digest,
    read_search,
    record_search,
)
from tests.conftest import TEST_CLIENT


def _event(**overrides) -> SearchEvent:
    base = dict(
        caller="assistente-juridico",
        question="quais os prazos de rescisão",
        dense_weight=0.5,
        lexical_weight=0.5,
        candidate_depth=50,
        result_count=10,
        index_search_effort=40,
        duration_ms=31,
        outcome=SEARCH_OUTCOME_NOTHING_MATCHED,
    )
    base.update(overrides)
    return SearchEvent(**base)


def test_the_digest_ignores_spacing_and_case_so_repetition_is_visible():
    canonical = question_digest("qual o prazo de rescisão")

    assert question_digest("Qual o Prazo de Rescisão") == canonical
    assert question_digest("  qual   o prazo\tde  rescisão  ") == canonical


def test_the_digest_distinguishes_two_different_questions():
    assert question_digest("prazo de rescisão") != question_digest("prazo de renovação")


def test_the_digest_is_a_sha256_hex_digest():
    digest = question_digest("qualquer pergunta")

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_the_question_text_is_withheld_unless_the_deployment_asks(conn: psycopg.Connection):
    withheld = read_search(conn, record_search(conn, _event(), client=TEST_CLIENT))
    kept = read_search(
        conn, record_search(conn, _event(), client=TEST_CLIENT, record_question_text=True)
    )

    assert withheld.question_text is None
    assert kept.question_text == "quais os prazos de rescisão"
    assert withheld.question_digest == kept.question_digest


def test_a_record_without_a_caller_is_refused():
    with pytest.raises(SearchEventError) as excinfo:
        _event(caller="   ")

    assert "caller" in str(excinfo.value)


def test_an_unknown_outcome_is_refused_and_names_the_three():
    with pytest.raises(SearchEventError) as excinfo:
        _event(outcome="empty")

    message = str(excinfo.value)
    assert SEARCH_OUTCOME_FOUND in message
    assert SEARCH_OUTCOME_NOTHING_MATCHED in message
    assert SEARCH_OUTCOME_FAILED in message


def test_a_found_outcome_that_returned_nothing_is_refused():
    with pytest.raises(SearchEventError) as excinfo:
        _event(outcome=SEARCH_OUTCOME_FOUND, chunk_ids=())

    assert SEARCH_OUTCOME_NOTHING_MATCHED in str(excinfo.value)


def test_a_candidate_depth_below_the_result_count_is_refused():
    with pytest.raises(SearchEventError) as excinfo:
        _event(candidate_depth=5, result_count=10)

    assert "nothing to fuse" in str(excinfo.value)


def test_an_index_effort_below_one_is_refused():
    with pytest.raises(SearchEventError):
        _event(index_search_effort=0)


def test_a_negative_duration_is_refused():
    with pytest.raises(SearchEventError):
        _event(duration_ms=-1)


def test_the_documents_returned_are_deduplicated_and_ordered(conn: psycopg.Connection):
    event = _event(
        outcome=SEARCH_OUTCOME_FOUND,
        document_ids=[7, 3, 7, 3, 11],
        chunk_ids=[70, 31, 72, 33, 110],
    )

    row = read_search(conn, record_search(conn, event, client=TEST_CLIENT))

    assert row.document_ids == (3, 7, 11)
    assert row.chunk_ids == (70, 31, 72, 33, 110)


def test_a_failed_search_is_recorded_with_no_results(conn: psycopg.Connection):
    row = read_search(
        conn,
        record_search(conn, _event(outcome=SEARCH_OUTCOME_FAILED), client=TEST_CLIENT),
    )

    assert row.outcome == SEARCH_OUTCOME_FAILED
    assert row.document_ids == ()
    assert row.chunk_ids == ()


def test_the_reranking_model_is_recorded_when_there_was_one(conn: psycopg.Connection):
    without = read_search(conn, record_search(conn, _event(), client=TEST_CLIENT))
    with_rerank = read_search(
        conn, record_search(conn, _event(rerank_model="zerank-2"), client=TEST_CLIENT)
    )

    assert without.rerank_model is None
    assert with_rerank.rerank_model == "zerank-2"


def test_the_weights_are_stored_as_written(conn: psycopg.Connection):
    row = read_search(
        conn,
        record_search(conn, _event(dense_weight=0.7, lexical_weight=0.3), client=TEST_CLIENT),
    )

    assert row.dense_weight == Decimal("0.7")
    assert row.lexical_weight == Decimal("0.3")


def test_occurred_at_defaults_to_now(conn: psycopg.Connection):
    before = datetime.now(timezone.utc) - timedelta(seconds=5)

    row = read_search(conn, record_search(conn, _event(), client=TEST_CLIENT))

    assert row.occurred_at >= before


def test_reading_a_pruned_search_returns_none(conn: psycopg.Connection):
    assert read_search(conn, 10_000_000) is None


def test_a_search_carrying_a_filter_records_it_beside_the_knobs(conn: psycopg.Connection):
    applied = {"all": [{"field": "media_type", "op": "is", "value": "application/pdf"}]}

    row = read_search(
        conn, record_search(conn, _event(filter=applied), client=TEST_CLIENT)
    )

    assert row.filter == applied


def test_an_unfiltered_search_records_the_absence_never_an_empty_filter(conn: psycopg.Connection):
    row = read_search(conn, record_search(conn, _event(), client=TEST_CLIENT))

    assert row.filter is None
    assert row.filter != {}
