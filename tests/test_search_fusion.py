from __future__ import annotations

import pytest

from src.config import SearchSettings
from src.search import RRF_DAMPING, SearchRequest, fuse, fused_order, settle
from src.search_result import Candidate, FusionWeights, Provenance


def _candidate(chunk_id: int, rank: int) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        content=f"piece {chunk_id}",
        provenance=Provenance(document_id=chunk_id, external_id=f"doc-{chunk_id}", page_start=1),
        rank=rank,
        leg_score=1.0 / rank,
    )


def test_a_piece_both_legs_found_outscores_the_same_position_found_by_one(
) -> None:
    both = fuse([_candidate(1, 1)], [_candidate(1, 1)], weights=FusionWeights(), result_count=10)
    one = fuse([_candidate(2, 1)], [], weights=FusionWeights(), result_count=10)
    assert both[0].score > one[0].score
    assert both[0].found_by == ("dense", "lexical")
    assert one[0].found_by == ("dense",)


def test_every_hit_carries_a_score_on_the_same_scale() -> None:
    hits = fuse([_candidate(1, 1), _candidate(2, 2)], [_candidate(3, 1)], weights=FusionWeights(), result_count=10)
    assert all(hit.score > 0 for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_the_score_is_the_weighted_reciprocal_of_each_rank() -> None:
    weights = FusionWeights(dense=0.25, lexical=0.75)
    (hit,) = fuse([_candidate(1, 2)], [_candidate(1, 4)], weights=weights, result_count=10)
    assert hit.score == pytest.approx(0.25 / (RRF_DAMPING + 2) + 0.75 / (RRF_DAMPING + 4))


def test_raising_a_leg_s_weight_reorders_the_result() -> None:
    dense = [_candidate(1, 1), _candidate(2, 5)]
    lexical = [_candidate(2, 1), _candidate(1, 5)]
    dense_led = fuse(dense, lexical, weights=FusionWeights(dense=0.9, lexical=0.1), result_count=10)
    lexical_led = fuse(dense, lexical, weights=FusionWeights(dense=0.1, lexical=0.9), result_count=10)
    assert dense_led[0].chunk_id == 1
    assert lexical_led[0].chunk_id == 2


def test_the_returned_count_never_exceeds_what_was_asked_for() -> None:
    hits = fuse([_candidate(i, i) for i in range(1, 21)], [], weights=FusionWeights(), result_count=3)
    assert len(hits) == 3


def test_a_leg_that_found_nothing_does_not_stop_the_other() -> None:
    hits = fuse([], [_candidate(1, 1)], weights=FusionWeights(), result_count=10)
    assert [hit.chunk_id for hit in hits] == [1]
    assert hits[0].found_by == ("lexical",)


def test_two_pieces_on_the_same_score_come_back_in_a_stable_order() -> None:
    first = fuse([_candidate(9, 1), _candidate(3, 1)], [], weights=FusionWeights(), result_count=10)
    second = fuse([_candidate(3, 1), _candidate(9, 1)], [], weights=FusionWeights(), result_count=10)
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second] == [3, 9]


def test_nothing_found_by_either_leg_fuses_to_nothing() -> None:
    assert fuse([], [], weights=FusionWeights(), result_count=10) == ()


def test_fuse_is_the_full_ordering_cut_to_the_result_count() -> None:
    candidates = [_candidate(i, i) for i in range(1, 6)]
    whole = fused_order(candidates, [], weights=FusionWeights())
    assert fuse(candidates, [], weights=FusionWeights(), result_count=2) == whole[:2]


def test_fused_order_keeps_every_candidate_either_leg_found() -> None:
    whole = fused_order([_candidate(i, i) for i in range(1, 6)], [], weights=FusionWeights())
    assert [hit.chunk_id for hit in whole] == [1, 2, 3, 4, 5]


def _defaults(**overrides) -> SearchSettings:
    return SearchSettings(**overrides)


def test_a_request_that_asks_for_nothing_gets_the_deployment_s_defaults() -> None:
    defaults = _defaults(dense_weight=0.8, lexical_weight=0.2, index_search_effort=90, candidate_depth=60)
    settled = settle(SearchRequest(question="q", query_vector=[0.0], result_count=7), defaults)
    assert settled.result_count == 7
    assert settled.weights == FusionWeights(dense=0.8, lexical=0.2)
    assert settled.index_search_effort == 90
    assert settled.candidate_depth == 60


def test_each_knob_can_be_overridden_on_its_own() -> None:
    settled = settle(
        SearchRequest(
            question="q",
            query_vector=[0.0],
            result_count=7,
            weights=FusionWeights(dense=0.1, lexical=0.9),
        ),
        _defaults(index_search_effort=90),
    )
    assert settled.weights == FusionWeights(dense=0.1, lexical=0.9)
    assert settled.result_count == 7
    assert settled.index_search_effort == 90


def test_a_depth_below_the_result_count_is_refused() -> None:
    with pytest.raises(ValueError, match="below the"):
        settle(
            SearchRequest(question="q", query_vector=[0.0], result_count=100),
            _defaults(candidate_depth=50),
        )


@pytest.mark.parametrize("bad", [{"result_count": 0}, {"index_search_effort": 0}])
def test_an_incoherent_knob_is_refused_by_name(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        settle(SearchRequest(question="q", query_vector=[0.0], **{"result_count": 10, **bad}), _defaults())
