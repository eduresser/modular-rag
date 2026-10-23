from __future__ import annotations

import pytest
from conftest import FakeScorer

from src.config import RerankSettings
from src.rerank import RerankError, Reranking, rerank, scorer_for, wanted
from src.search_result import Provenance, SearchHit

MODEL = "cross-encoder/mmarco-mMiniLMv2"


def _hit(chunk_id: int, content: str, score: float = 0.1) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content=content,
        provenance=Provenance(document_id=1, external_id=f"doc-{chunk_id}", page_start=1),
        score=score,
        found_by=("dense",),
    )


HITS = (_hit(1, "primeiro", score=0.9), _hit(2, "segundo", score=0.5), _hit(3, "terceiro", score=0.1))


def _enabled() -> RerankSettings:
    return RerankSettings(enabled=True, model=MODEL)


def test_the_stage_is_off_unless_a_deployment_or_a_request_turns_it_on():
    off = RerankSettings()

    assert wanted(off, None) is False
    assert wanted(off, True) is True
    assert wanted(_enabled(), None) is True
    assert wanted(_enabled(), False) is False


def test_a_stage_that_is_off_hands_the_fusions_order_back_untouched():
    result = rerank(HITS, "pergunta", settings=RerankSettings())

    assert result.hits == HITS
    assert result.model == ""
    assert result.happened is False


def test_the_model_decides_the_order_and_is_named_on_the_way_out():
    scorer = FakeScorer(MODEL, {"terceiro": 9.0, "primeiro": 1.0, "segundo": 5.0})

    result = rerank(HITS, "pergunta", settings=_enabled(), scorer=scorer)

    assert [hit.chunk_id for hit in result.hits] == [3, 2, 1]
    assert result.model == MODEL
    assert result.happened is True
    assert scorer.calls == [("pergunta", ("primeiro", "segundo", "terceiro"))]


def test_a_tie_is_broken_by_the_piece_and_never_by_chance():
    scorer = FakeScorer(MODEL, {"primeiro": 1.0, "segundo": 1.0, "terceiro": 1.0})

    first = rerank(HITS, "pergunta", settings=_enabled(), scorer=scorer)
    again = rerank(HITS, "pergunta", settings=_enabled(), scorer=scorer)

    assert [hit.chunk_id for hit in first.hits] == [1, 2, 3]
    assert [hit.chunk_id for hit in first.hits] == [hit.chunk_id for hit in again.hits]


def test_a_single_result_is_reported_as_reranked_without_paying_for_a_call():
    scorer = FakeScorer(MODEL)

    result = rerank(HITS[:1], "pergunta", settings=_enabled(), scorer=scorer)

    assert result.hits == HITS[:1]
    assert result.model == MODEL
    assert scorer.calls == []


def test_reranking_with_no_model_configured_is_refused_by_name():
    with pytest.raises(RerankError) as excinfo:
        scorer_for(RerankSettings(enabled=True), FakeScorer(MODEL))

    assert "rerank.model" in str(excinfo.value)


def test_reranking_with_no_scorer_wired_in_is_refused():
    with pytest.raises(RerankError) as excinfo:
        scorer_for(_enabled(), None)

    assert MODEL in str(excinfo.value)
    assert "no scorer" in str(excinfo.value)


def test_a_scorer_that_is_not_the_configured_model_is_refused():
    with pytest.raises(RerankError) as excinfo:
        scorer_for(_enabled(), FakeScorer("some/other-model"))

    assert MODEL in str(excinfo.value)
    assert "some/other-model" in str(excinfo.value)


def test_a_model_that_scores_only_some_of_the_passages_is_refused():
    class Partial:
        model = MODEL

        def score(self, question, passages):
            return [1.0]

    with pytest.raises(RerankError) as excinfo:
        rerank(HITS, "pergunta", settings=_enabled(), scorer=Partial())

    assert "1 of 3" in str(excinfo.value)


def test_an_empty_reranking_reports_nothing_happened():
    assert Reranking(hits=()).happened is False
