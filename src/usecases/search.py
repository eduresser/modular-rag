from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import psycopg

from src.config import AppConfig
from src.db import DatabaseUnavailable, connect
from src.embedding.client import EmbeddingError
from src.filters import Clause, FilterNode, compile_filter
from src.rerank import Reranking, Scorer, rerank, wanted
from src.responses import ASSISTANT, SERVICE, Audience, shorten_to_sentence
from src.search import FoundWithOmitted, SearchRequest, search_on, settle
from src.search_events import (
    SEARCH_OUTCOME_FAILED,
    SEARCH_OUTCOME_FOUND,
    SEARCH_OUTCOME_NOTHING_MATCHED,
    SearchEvent,
    record_search,
)
from src.search_result import (
    Failed,
    FilterMatchedNothing,
    Found,
    FusionWeights,
    NothingMatched,
    SearchHit,
    SearchOutcome,
)
from src.embedding.stamp import stamp_from_config
from src.store import DOCUMENT_STATE_GONE, ModelStamp
from src.store_versions import NoActiveVersion, active_version

logger = logging.getLogger(__name__)

UNNAMED_CALLER = "direct"


class KnobOutOfRange(ValueError):
    ...


class QueryEmbedder(Protocol):

    def embed_query(self, question: str) -> Sequence[float]:
        ...


@dataclass(frozen=True, slots=True)
class SearchQuery:

    question: str
    caller: str = UNNAMED_CALLER
    offset: int = 0
    result_count: int | None = None
    dense_weight: float | None = None
    lexical_weight: float | None = None
    candidate_depth: int | None = None
    index_search_effort: int | None = None
    rerank: bool | None = None
    brief: bool | None = None
    filter: FilterNode | None = None


@dataclass(frozen=True, slots=True)
class _Plan:

    request: SearchRequest
    result_count: int
    weights: FusionWeights
    candidate_depth: int
    index_search_effort: int
    brief: bool
    # The surface's own window setting, not the archive's — resolved once here
    # so `_on` never has to ask which surface it is serving (AC-294).
    brief_length: int
    offset: int = 0


def _within(value: int | None, *, knob: str, setting: str, ceiling: int) -> None:
    if value is not None and value > ceiling:
        raise KnobOutOfRange(
            f"{knob} {value} is above this deployment's maximum of {ceiling} "
            f"({setting}); the request is refused rather than reduced to the limit, "
            "because a silently reduced request answers a question nobody asked"
        )


def plan(query: SearchQuery, config: AppConfig, *, audience: Audience = SERVICE) -> _Plan:
    if not query.question.strip():
        raise KnobOutOfRange("a search needs a question; an empty one matches nothing by construction")
    limits = config.search
    surface = config.mcp if audience == ASSISTANT else config.api
    window = surface.defaults.search
    limit_ceiling = surface.limits.search.limit
    result_count = query.result_count if query.result_count is not None else window.limit
    _within(
        query.result_count,
        knob="result count",
        setting="mcp.limits.search.limit" if audience == ASSISTANT else "api.limits.search.limit",
        ceiling=limit_ceiling,
    )
    _within(
        query.candidate_depth,
        knob="candidate depth",
        setting="search.max_candidate_depth",
        ceiling=limits.max_candidate_depth,
    )
    _within(
        query.index_search_effort,
        knob="index search effort",
        setting="search.max_index_search_effort",
        ceiling=limits.max_index_search_effort,
    )
    request = SearchRequest(
        question=query.question,
        query_vector=(),
        result_count=result_count,
        weights=_weights(query, config),
        candidate_depth=query.candidate_depth,
        index_search_effort=query.index_search_effort,
        filter=query.filter,
    )
    try:
        settled = settle(request, limits)
    except ValueError as exc:
        raise KnobOutOfRange(str(exc)) from exc
    if query.offset < 0:
        raise KnobOutOfRange(
            f"a search cannot start before its first result, got offset {query.offset}"
        )
    return _Plan(
        request=request,
        offset=query.offset,
        result_count=settled.result_count,
        weights=settled.weights,
        candidate_depth=settled.candidate_depth,
        index_search_effort=settled.index_search_effort,
        brief=bool(query.brief),
        brief_length=window.brief_length,
    )


def _weights(query: SearchQuery, config: AppConfig) -> FusionWeights:
    dense = query.dense_weight if query.dense_weight is not None else config.search.dense_weight
    lexical = (
        query.lexical_weight if query.lexical_weight is not None else config.search.lexical_weight
    )
    try:
        return FusionWeights(dense=dense, lexical=lexical)
    except ValueError as exc:
        raise KnobOutOfRange(str(exc)) from exc


def run_search(
    query: SearchQuery,
    *,
    config: AppConfig,
    embedder: QueryEmbedder,
    audience: Audience = SERVICE,
    scorer: Scorer | None = None,
    connection: psycopg.Connection | None = None,
) -> SearchOutcome:
    prepared = plan(query, config, audience=audience)
    started = time.perf_counter()
    try:
        vector = embedder.embed_query(query.question)
    except Exception as exc:
        reason = (
            f"the question could not be embedded: {exc}"
            if isinstance(exc, EmbeddingError)
            else f"the question could not be embedded: {type(exc).__name__}: {exc}"
        )
        failure = Failed(reason=reason)
        elapsed = (time.perf_counter() - started) * 1000.0
        if connection is not None:
            return _record(connection, query, config, prepared, failure, elapsed)
        return _record_apart(query, config, prepared, failure, elapsed)
    prepared = replace(prepared, request=replace(prepared.request, query_vector=vector))
    if connection is not None:
        return _on(connection, query, prepared, config, scorer, started)
    try:
        with connect(config.db.url) as conn:
            return _on(conn, query, prepared, config, scorer, started)
    except DatabaseUnavailable as exc:
        return Failed(reason=str(exc))


def _on(
    conn: psycopg.Connection,
    query: SearchQuery,
    prepared: _Plan,
    config: AppConfig,
    scorer: Scorer | None,
    started: float,
) -> SearchOutcome:
    reranking_wanted = wanted(config.rerank, query.rerank)
    request = (
        replace(prepared.request, result_count=prepared.candidate_depth)
        if reranking_wanted
        else prepared.request
    )
    mismatch = _generation_mismatch(conn, config)
    try:
        if mismatch is not None:
            outcome: SearchOutcome = Failed(reason=mismatch)
        else:
            outcome = search_on(conn, request, defaults=config.search)
    except NoActiveVersion as exc:
        outcome = Failed(reason=str(exc))
    except psycopg.Error as exc:
        outcome = Failed(reason=f"the archive refused the query: {exc}".strip())
    outcome = _with_filter_counts(conn, request.filter, outcome)
    reranking = _reranked(outcome, query, config, scorer)
    if isinstance(outcome, Found):
        ranked = tuple(reranking.hits) + tuple(getattr(outcome, "omitted", ()))
        visible = ranked[prepared.offset : prepared.offset + prepared.result_count]
        leftover = ranked[prepared.offset + prepared.result_count :]
        enriched_visible = _enrich_provenance(conn, visible)
        if prepared.brief:
            brief_hits = []
            for hit in enriched_visible:
                shortened = shorten_to_sentence(hit.content, prepared.brief_length)
                brief_hits.append(
                    replace(
                        hit,
                        content=shortened,
                        truncated=shortened != hit.content,
                        full_length=len(hit.content),
                    )
                )
            enriched_visible = tuple(brief_hits)
        diagnostics = replace(
            outcome.diagnostics, result_count=prepared.result_count, offset=prepared.offset
        )
        if not enriched_visible:
            return _record(
                conn,
                query,
                config,
                prepared,
                NothingMatched(diagnostics=diagnostics),
                (time.perf_counter() - started) * 1000.0,
                rerank_model=reranking.model,
            )
        outcome = (
            FoundWithOmitted(hits=enriched_visible, diagnostics=diagnostics, omitted=leftover)
            if leftover
            else Found(hits=enriched_visible, diagnostics=diagnostics)
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    return _record(conn, query, config, prepared, outcome, elapsed, rerank_model=reranking.model)


def _enrich_provenance(
    conn: psycopg.Connection, hits: Sequence[SearchHit]
) -> tuple[SearchHit, ...]:
    if not hits:
        return tuple(hits)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, ordinal FROM chunks WHERE id = ANY(%s)",
            ([hit.chunk_id for hit in hits],),
        )
        ordinals = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.execute(
            "SELECT id, link, location FROM documents WHERE id = ANY(%s)",
            (list({hit.provenance.document_id for hit in hits}),),
        )
        facts = {row[0]: (row[1], tuple(row[2] or ())) for row in cursor.fetchall()}
    enriched = []
    for hit in hits:
        link, location = facts.get(hit.provenance.document_id, (None, ()))
        enriched.append(
            replace(
                hit,
                provenance=replace(
                    hit.provenance,
                    ordinal=ordinals.get(hit.chunk_id),
                    link=link,
                    location=location,
                ),
            )
        )
    return tuple(enriched)


def _filter_selectivity(
    conn: psycopg.Connection, filter_: FilterNode, stamp: ModelStamp
) -> tuple[int, int]:
    fragment, params = compile_filter(filter_)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT count(DISTINCT c.document_id), count(*)
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE d.state <> %s
               AND c.embed_model = %s
               AND c.embed_dimension = %s
               AND ({fragment})
            """,
            [DOCUMENT_STATE_GONE, stamp.model, stamp.dimension, *params],
        )
        documents, pieces = cursor.fetchone()
    return documents, pieces


def _leaf_clauses(node: FilterNode) -> list[Clause]:
    if isinstance(node, Clause):
        return [node]
    leaves: list[Clause] = []
    for child in node.children:
        leaves.extend(_leaf_clauses(child))
    return leaves


def _describe_clause(clause: Clause) -> str:
    return f"{clause.field} {clause.op} {clause.value!r}"


def _responsible_clause(conn: psycopg.Connection, filter_: FilterNode, stamp: ModelStamp) -> str:
    leaves = _leaf_clauses(filter_)
    for leaf in leaves:
        documents, _ = _filter_selectivity(conn, leaf, stamp)
        if documents == 0:
            return _describe_clause(leaf)
    return "; ".join(_describe_clause(leaf) for leaf in leaves)


def _with_filter_counts(
    conn: psycopg.Connection, filter_: FilterNode | None, outcome: SearchOutcome
) -> SearchOutcome:
    if filter_ is None or not isinstance(outcome, (Found, NothingMatched)):
        return outcome
    try:
        stamp = active_version(conn)
        documents_considered, pieces_considered = _filter_selectivity(conn, filter_, stamp)
    except Exception as exc:
        logger.warning("the filter's selectivity was not counted: %s", exc)
        return outcome
    diagnostics = replace(
        outcome.diagnostics,
        documents_considered=documents_considered,
        pieces_considered=pieces_considered,
    )
    if isinstance(outcome, NothingMatched) and documents_considered == 0:
        return FilterMatchedNothing(
            diagnostics=diagnostics,
            clause=_responsible_clause(conn, filter_, stamp),
        )
    return replace(outcome, diagnostics=diagnostics)


def _generation_mismatch(conn: psycopg.Connection, config: AppConfig) -> str | None:
    try:
        active = active_version(conn)
    except NoActiveVersion:
        return None
    configured = stamp_from_config(config)
    if active == configured:
        return None
    return (
        f"this deployment embeds a question with {configured.model!r} at {configured.dimension} "
        f"dimensions while the archive is serving vectors from {active.model!r} at "
        f"{active.dimension}; comparing the two is comparing things that are not comparable, "
        "so nothing is returned rather than something that cannot be trusted (AC-052). "
        "Either put the configured model back, or finish the switch by activating the "
        "generation that was written: make cutover APP=<app>"
    )


def _reranked(
    outcome: SearchOutcome,
    query: SearchQuery,
    config: AppConfig,
    scorer: Scorer | None,
) -> Reranking:
    if not isinstance(outcome, Found):
        return Reranking(hits=())
    try:
        return rerank(
            outcome.hits,
            query.question,
            settings=config.rerank,
            scorer=scorer,
            requested=query.rerank,
        )
    except Exception as exc:
        logger.warning("reranking was skipped: %s", exc)
        return Reranking(hits=outcome.hits)


def _outcome_name(outcome: SearchOutcome) -> str:
    if isinstance(outcome, Found):
        return SEARCH_OUTCOME_FOUND
    if isinstance(outcome, (NothingMatched, FilterMatchedNothing)):
        return SEARCH_OUTCOME_NOTHING_MATCHED
    return SEARCH_OUTCOME_FAILED


def _event(
    query: SearchQuery,
    prepared: _Plan,
    outcome: SearchOutcome,
    elapsed_ms: float,
    rerank_model: str,
) -> SearchEvent:
    hits = getattr(outcome, "hits", ())
    return SearchEvent(
        caller=query.caller or UNNAMED_CALLER,
        question=query.question,
        dense_weight=prepared.weights.dense,
        lexical_weight=prepared.weights.lexical,
        candidate_depth=prepared.candidate_depth,
        result_count=prepared.result_count,
        index_search_effort=prepared.index_search_effort,
        duration_ms=max(elapsed_ms, 0.0),
        outcome=_outcome_name(outcome),
        document_ids=[hit.provenance.document_id for hit in hits],
        chunk_ids=[hit.chunk_id for hit in hits],
        rerank_model=rerank_model or None,
    )


def _record(
    conn: psycopg.Connection,
    query: SearchQuery,
    config: AppConfig,
    prepared: _Plan,
    outcome: SearchOutcome,
    elapsed_ms: float,
    *,
    rerank_model: str = "",
) -> SearchOutcome:
    try:
        record_search(
            conn,
            _event(query, prepared, outcome, elapsed_ms, rerank_model),
            client=config.telemetry.client,
            record_question_text=config.usage.record_question_text,
        )
    except Exception as exc:
        logger.warning("the search was not recorded: %s", exc)
    return outcome


def _record_apart(
    query: SearchQuery,
    config: AppConfig,
    prepared: _Plan,
    outcome: SearchOutcome,
    elapsed_ms: float,
) -> SearchOutcome:
    try:
        with connect(config.db.url) as conn:
            return _record(conn, query, config, prepared, outcome, elapsed_ms)
    except Exception as exc:
        logger.warning("the failed search was not recorded: %s", exc)
    return outcome
