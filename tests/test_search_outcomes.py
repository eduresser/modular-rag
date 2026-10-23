from __future__ import annotations

import pytest

from src.search_result import (
    Failed,
    Found,
    FusionWeights,
    NothingMatched,
    Provenance,
    SearchDiagnostics,
    SearchHit,
)


def _diagnostics() -> SearchDiagnostics:
    return SearchDiagnostics(
        index_search_effort=40,
        weights=FusionWeights(),
        candidate_depth=50,
        result_count=10,
    )


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id=1,
        content="Cláusula de rescisão antecipada",
        provenance=Provenance(document_id=1, external_id="drive-1", page_start=3, page_end=3),
        score=0.5,
        found_by=("dense",),
    )


def test_an_empty_match_carries_a_present_empty_list_not_an_absent_field() -> None:
    outcome = NothingMatched(diagnostics=_diagnostics())
    assert outcome.succeeded is True
    assert outcome.hits == ()
    assert hasattr(outcome, "hits")


def test_a_failure_carries_no_hits_attribute_at_all() -> None:
    outcome = Failed(reason="could not reach the archive")
    assert outcome.succeeded is False
    assert not hasattr(outcome, "hits")


def test_a_failure_without_a_reason_is_refused() -> None:
    with pytest.raises(ValueError):
        Failed(reason="   ")


def test_a_found_outcome_with_no_hits_is_refused() -> None:
    with pytest.raises(ValueError):
        Found(hits=(), diagnostics=_diagnostics())


def test_a_found_outcome_carries_its_hits_and_reports_success() -> None:
    outcome = Found(hits=(_hit(),), diagnostics=_diagnostics())
    assert outcome.succeeded is True
    assert len(outcome.hits) == 1


def test_the_effort_used_is_recorded_whatever_the_outcome() -> None:
    diagnostics = SearchDiagnostics(
        index_search_effort=200,
        weights=FusionWeights(dense=0.7, lexical=0.3),
        candidate_depth=80,
        result_count=5,
    )
    assert Found(hits=(_hit(),), diagnostics=diagnostics).diagnostics.index_search_effort == 200
    assert NothingMatched(diagnostics=diagnostics).diagnostics.index_search_effort == 200


def test_a_hit_must_name_the_leg_that_found_it() -> None:
    with pytest.raises(ValueError):
        SearchHit(
            chunk_id=1,
            content="x",
            provenance=Provenance(document_id=1, external_id="drive-1", page_start=1),
            score=0.1,
            found_by=(),
        )


@pytest.mark.parametrize("weights", [(-0.1, 0.5), (0.5, -0.1), (0.0, 0.0)])
def test_weights_that_would_switch_off_retrieval_are_refused(weights: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        FusionWeights(dense=weights[0], lexical=weights[1])


def test_weights_default_to_an_even_split() -> None:
    assert FusionWeights() == FusionWeights(dense=0.5, lexical=0.5)
