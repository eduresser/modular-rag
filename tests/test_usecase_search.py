
from __future__ import annotations

import psycopg
import pytest
from conftest import (
    API_CORPUS,
    TEST_STAMP,
    FakeQueryEmbedder,
    FakeScorer,
    api_config,
    index_corpus,
)

from src.embedding.client import EmbeddingError
from src.responses import documents_search_payload
from src.search import FoundWithOmitted
from src.search_result import Failed, Found, NothingMatched
from src.store_versions import activate_version
from src.usecases.search import (
    UNNAMED_CALLER,
    KnobOutOfRange,
    SearchQuery,
    plan,
    run_search,
)

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2"


@pytest.fixture
def corpus(conn: psycopg.Connection) -> psycopg.Connection:
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    return conn


def _config(monkeypatch, **overrides):
    return api_config(monkeypatch, "postgresql://localhost/unused", **overrides)


def test_a_request_that_asks_for_nothing_gets_the_deployments_own_values(monkeypatch):
    config = _config(monkeypatch, RAG_SEARCH_RESULT_COUNT="7", RAG_SEARCH_DENSE_WEIGHT="0.4")

    settled = plan(SearchQuery(question="rescisão"), config)

    assert settled.result_count == 7
    assert settled.weights.dense == 0.4
    assert settled.candidate_depth == config.search.candidate_depth


def test_setting_one_weight_leaves_the_other_at_the_deployments_value(monkeypatch):
    config = _config(monkeypatch, RAG_SEARCH_DENSE_WEIGHT="0.4", RAG_SEARCH_LEXICAL_WEIGHT="0.6")

    settled = plan(SearchQuery(question="rescisão", lexical_weight=0.1), config)

    assert (settled.weights.dense, settled.weights.lexical) == (0.4, 0.1)


def test_two_weights_of_zero_are_refused_rather_than_run(monkeypatch):
    config = _config(monkeypatch)

    with pytest.raises(KnobOutOfRange):
        plan(SearchQuery(question="rescisão", dense_weight=0, lexical_weight=0), config)


def test_an_empty_question_is_refused_before_anything_is_paid_for(monkeypatch):
    config = _config(monkeypatch)

    with pytest.raises(KnobOutOfRange) as excinfo:
        plan(SearchQuery(question="   "), config)

    assert "needs a question" in str(excinfo.value)


def test_a_depth_below_the_result_count_is_refused_as_one_kind_of_refusal(monkeypatch):
    config = _config(monkeypatch)

    with pytest.raises(KnobOutOfRange) as excinfo:
        plan(SearchQuery(question="rescisão", result_count=20, candidate_depth=5), config)

    assert "5" in str(excinfo.value) and "20" in str(excinfo.value)


@pytest.mark.parametrize(
    "knob,value,setting",
    [
        ("result_count", 999, "api.limits.search.limit"),
        ("candidate_depth", 9_999, "search.max_candidate_depth"),
        ("index_search_effort", 99_999, "search.max_index_search_effort"),
    ],
)
def test_every_ceiling_names_itself_and_the_value_it_refused(monkeypatch, knob, value, setting):
    config = _config(monkeypatch)

    with pytest.raises(KnobOutOfRange) as excinfo:
        plan(SearchQuery(question="rescisão", **{knob: value}), config)

    message = str(excinfo.value)
    assert setting in message and str(value) in message


def test_a_search_runs_both_legs_and_returns_citable_hits(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, Found)
    assert all(hit.provenance.external_id for hit in outcome.hits)
    assert "drive-removido" not in {hit.provenance.external_id for hit in outcome.hits}


def test_a_result_count_below_the_matches_found_carries_what_it_left_out(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test", result_count=1),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, FoundWithOmitted)
    assert len(outcome.hits) == 1
    assert outcome.omitted
    assert all(hit.provenance.ordinal is not None for hit in outcome.hits)
    assert all(hit.provenance.ordinal is None for hit in outcome.omitted)
    assert all(hit.provenance.document_id for hit in outcome.omitted)


def test_a_result_count_that_covers_every_match_carries_nothing_omitted(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test", result_count=50),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, Found)
    assert not isinstance(outcome, FoundWithOmitted)


def test_the_documents_view_scores_each_document_and_carries_no_passage_text(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, Found)
    documents = documents_search_payload(outcome)["results"]

    assert [doc["document"] for doc in documents] == ["drive-rescisao", "drive-reajuste"]
    rescisao = documents[0]
    assert rescisao["match_count"] == 2
    assert rescisao["score"] >= rescisao["average_score"]
    assert documents[1]["match_count"] == 1
    assert documents[1]["score"] == documents[1]["average_score"]

    for doc in documents:
        assert "passages" not in doc
        assert "text" not in doc


def test_a_question_matching_nothing_is_an_explicit_empty(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="xilografia bizantina"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=200),
        connection=corpus,
    )

    assert isinstance(outcome, NothingMatched)
    assert outcome.hits == ()


def test_an_embedding_provider_that_refuses_becomes_a_failure_and_never_an_exception(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(fails_with=EmbeddingError("401 invalid api key")),
        connection=corpus,
    )

    assert isinstance(outcome, Failed)
    assert "401" in outcome.reason


def test_an_embedder_that_fails_in_a_way_nobody_anticipated_is_still_a_failure(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(fails_with=TimeoutError("the provider went away")),
        connection=corpus,
    )

    assert isinstance(outcome, Failed)
    assert "TimeoutError" in outcome.reason


def test_an_unreachable_archive_is_a_failure_carrying_the_reason(monkeypatch):
    config = _config(monkeypatch, RAG_DB_URL="postgresql://127.0.0.1:1/nowhere")

    outcome = run_search(
        SearchQuery(question="rescisão"), config=config, embedder=FakeQueryEmbedder()
    )

    assert isinstance(outcome, Failed)
    assert "could not reach the archive" in outcome.reason


def test_an_archive_with_no_active_version_is_a_failure_and_not_an_empty(monkeypatch, conn):
    index_corpus(conn, API_CORPUS)
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )

    assert isinstance(outcome, Failed)


def _last_event(conn: psycopg.Connection):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT caller, outcome, result_count, candidate_depth, rerank_model, question_text "
            "FROM search_events ORDER BY id DESC LIMIT 1"
        )
        return cursor.fetchone()


def test_a_search_that_failed_is_recorded_with_the_knobs_it_would_have_used(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    run_search(
        SearchQuery(question="rescisão", result_count=3, candidate_depth=30),
        config=config,
        embedder=FakeQueryEmbedder(fails_with=EmbeddingError("401")),
        connection=corpus,
    )

    caller, outcome, result_count, depth, _, _ = _last_event(corpus)
    assert outcome == "failed"
    assert (result_count, depth) == (3, 30)
    assert caller == UNNAMED_CALLER


def test_the_questions_text_is_not_recorded_unless_the_deployment_says_so(monkeypatch, corpus):
    config = _config(monkeypatch)

    run_search(
        SearchQuery(question="qual o prazo de rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert _last_event(corpus)[5] is None


def test_a_deployment_that_asks_for_the_question_text_gets_it(monkeypatch, corpus):
    config = _config(monkeypatch, RAG_USAGE_RECORD_QUESTION_TEXT="true")

    run_search(
        SearchQuery(question="qual o prazo de rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert _last_event(corpus)[5] == "qual o prazo de rescisão"


def test_a_record_that_cannot_be_written_never_costs_the_caller_the_answer(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    def refuse(*args, **kwargs):
        raise psycopg.errors.InsufficientPrivilege("no write for you")

    monkeypatch.setattr("src.usecases.search.record_search", refuse)
    outcome = run_search(
        SearchQuery(question="rescisão antecipada"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, Found)


def test_reranking_sees_the_candidates_and_not_the_already_cut_answer(monkeypatch, corpus):
    scorer = FakeScorer(RERANK_MODEL)
    config = _config(monkeypatch, RAG_RERANK_MODEL=RERANK_MODEL)

    run_search(
        SearchQuery(question="rescisão antecipada", result_count=1, rerank=True),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        scorer=scorer,
        connection=corpus,
    )

    (_, passages), = scorer.calls
    assert len(passages) > 1


def test_a_reranking_that_fails_costs_the_ordering_and_never_the_answer(monkeypatch, corpus):
    class Broken:
        model = RERANK_MODEL

        def score(self, question, passages):
            raise RuntimeError("the reranking endpoint is down")

    config = _config(monkeypatch, RAG_RERANK_MODEL=RERANK_MODEL)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", rerank=True),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        scorer=Broken(),
        connection=corpus,
    )

    assert isinstance(outcome, Found)
    assert outcome.hits
    assert _last_event(corpus)[4] is None


def test_the_record_names_the_model_that_actually_reordered_the_results(monkeypatch, corpus):
    config = _config(monkeypatch, RAG_RERANK_MODEL=RERANK_MODEL)

    run_search(
        SearchQuery(question="rescisão antecipada", rerank=True),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        scorer=FakeScorer(RERANK_MODEL),
        connection=corpus,
    )

    assert _last_event(corpus)[4] == RERANK_MODEL


def test_a_question_embedded_with_another_model_than_the_archive_is_refused(monkeypatch, corpus):
    config = _config(monkeypatch, RAG_EMBED_MODEL="qwen/qwen3-embedding-4b")

    outcome = run_search(
        SearchQuery(question="prazo de vigência", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert not outcome.succeeded
    assert "qwen/qwen3-embedding-4b" in outcome.reason
    assert TEST_STAMP.model in outcome.reason
    assert "cutover" in outcome.reason


def test_the_archive_answers_normally_once_the_two_agree(monkeypatch, corpus):
    config = _config(monkeypatch)

    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=corpus,
    )

    assert isinstance(outcome, Found)
