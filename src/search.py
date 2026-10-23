from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from src.config import SearchSettings
from src.db import DatabaseUnavailable, connect
from src.filters import Clause, Combinator, FilterNode
from src.search_dense import dense_candidates
from src.search_lexical import lexical_candidates
from src.search_result import (
    DENSE_LEG,
    LEXICAL_LEG,
    Candidate,
    Failed,
    Found,
    FusionWeights,
    NothingMatched,
    SearchDiagnostics,
    SearchHit,
    SearchOutcome,
)
from src.store import ModelStamp
from src.store_versions import NoActiveVersion, active_version

RRF_DAMPING = 60


@dataclass(frozen=True, slots=True)
class SearchRequest:
    question: str
    query_vector: Sequence[float]
    result_count: int | None = None
    weights: FusionWeights | None = None
    candidate_depth: int | None = None
    index_search_effort: int | None = None
    filter: FilterNode | None = None


@dataclass(frozen=True, slots=True)
class FoundWithOmitted(Found):
    omitted: tuple[SearchHit, ...] = ()


@dataclass(frozen=True, slots=True)
class _Settled:
    result_count: int
    weights: FusionWeights
    candidate_depth: int
    index_search_effort: int


def settle(request: SearchRequest, defaults: SearchSettings) -> _Settled:
    if request.result_count is None:
        raise ValueError(
            "a search request reaching the archive has to name its result count; "
            "it is the serving surface's setting now (api.defaults.search.limit or "
            "mcp.defaults.search.limit), not the search stage's"
        )
    result_count = request.result_count
    candidate_depth = (
        request.candidate_depth if request.candidate_depth is not None else defaults.candidate_depth
    )
    weights = request.weights or FusionWeights(
        dense=defaults.dense_weight, lexical=defaults.lexical_weight
    )
    effort = (
        request.index_search_effort
        if request.index_search_effort is not None
        else defaults.index_search_effort
    )
    if result_count < 1:
        raise ValueError(f"a search has to return at least one result if it finds one, got {result_count}")
    if candidate_depth < result_count:
        raise ValueError(
            f"candidate depth {candidate_depth} is below the {result_count} results asked for; "
            "the fusion has to look at more than it returns, or it is not fusing anything"
        )
    if effort < 1:
        raise ValueError(f"index search effort must be at least 1, got {effort}")
    return _Settled(
        result_count=result_count,
        weights=weights,
        candidate_depth=candidate_depth,
        index_search_effort=effort,
    )


def fused_order(
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    weights: FusionWeights,
) -> tuple[SearchHit, ...]:
    scores: dict[int, float] = {}
    legs: dict[int, list[str]] = {}
    pieces: dict[int, Candidate] = {}
    for leg_name, weight, candidates in (
        (DENSE_LEG, weights.dense, dense),
        (LEXICAL_LEG, weights.lexical, lexical),
    ):
        for candidate in candidates:
            contribution = weight / (RRF_DAMPING + candidate.rank)
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + contribution
            legs.setdefault(candidate.chunk_id, []).append(leg_name)
            pieces.setdefault(candidate.chunk_id, candidate)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return tuple(
        SearchHit(
            chunk_id=chunk_id,
            content=pieces[chunk_id].content,
            provenance=pieces[chunk_id].provenance,
            score=scores[chunk_id],
            found_by=tuple(legs[chunk_id]),
        )
        for chunk_id in ordered
    )


def fuse(
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    weights: FusionWeights,
    result_count: int,
) -> tuple[SearchHit, ...]:
    return fused_order(dense, lexical, weights=weights)[:result_count]


def search_on(
    conn: psycopg.Connection,
    request: SearchRequest,
    *,
    defaults: SearchSettings,
    stamp: ModelStamp | None = None,
) -> SearchOutcome:
    settled = settle(request, defaults)
    started = time.perf_counter()
    scope = stamp or active_version(conn)
    dense = dense_candidates(
        conn,
        query_vector=request.query_vector,
        stamp=scope,
        depth=settled.candidate_depth,
        index_search_effort=settled.index_search_effort,
        filter_=request.filter,
        filter_timeout_ms=defaults.filter_timeout_ms,
    )
    lexical = lexical_candidates(
        conn,
        question=request.question,
        stamp=scope,
        depth=settled.candidate_depth,
        filter_=request.filter,
        filter_timeout_ms=defaults.filter_timeout_ms,
    )
    ordered = fused_order(dense, lexical, weights=settled.weights)
    hits = ordered[: settled.result_count]
    omitted = ordered[settled.result_count :]
    diagnostics = SearchDiagnostics(
        index_search_effort=settled.index_search_effort,
        weights=settled.weights,
        candidate_depth=settled.candidate_depth,
        result_count=settled.result_count,
        dense_candidates=len(dense),
        lexical_candidates=len(lexical),
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
    if not hits:
        return NothingMatched(diagnostics=diagnostics)
    if omitted:
        return FoundWithOmitted(hits=hits, diagnostics=diagnostics, omitted=omitted)
    return Found(hits=hits, diagnostics=diagnostics)


def _filter_field_names(node: FilterNode) -> tuple[str, ...]:
    if isinstance(node, Clause):
        return (node.field,)
    names: list[str] = []
    for child in node.children:
        for name in _filter_field_names(child):
            if name not in names:
                names.append(name)
    return tuple(names)


def search(dsn: str, request: SearchRequest, *, defaults: SearchSettings) -> SearchOutcome:
    try:
        with connect(dsn) as conn:
            return search_on(conn, request, defaults=defaults)
    except DatabaseUnavailable as exc:
        return Failed(reason=str(exc))
    except NoActiveVersion as exc:
        return Failed(reason=str(exc))
    except psycopg.errors.QueryCanceled as exc:
        if request.filter is None:
            return Failed(reason=f"the archive refused the query: {exc}".strip())
        fields = ", ".join(_filter_field_names(request.filter))
        return Failed(
            reason=(
                "the filter timed out against this deployment's filter_timeout_ms "
                f"({defaults.filter_timeout_ms}ms), evaluating {fields}"
            )
        )
    except psycopg.Error as exc:
        return Failed(reason=f"the archive refused the query: {exc}".strip())
