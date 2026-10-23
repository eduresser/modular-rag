from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Literal

from src.search_events import (
    SEARCH_OUTCOME_FAILED,
    SEARCH_OUTCOME_FOUND,
    SEARCH_OUTCOME_NOTHING_MATCHED,
)
from src.search_result import (
    Failed,
    FilterMatchedNothing,
    Found,
    NothingMatched,
    SearchDiagnostics,
    SearchHit,
    SearchOutcome,
)

STATUS_FOUND = SEARCH_OUTCOME_FOUND
STATUS_NOTHING_MATCHED = SEARCH_OUTCOME_NOTHING_MATCHED
STATUS_FAILED = SEARCH_OUTCOME_FAILED

STATUS_FILTER_MATCHED_NOTHING = "filter_matched_nothing"

PIECE_SEPARATOR = "#"

ASSISTANT = "assistant"
SERVICE = "service"
Audience = Literal["assistant", "service"]

BRIEF_CUT_MARK = " […]"

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def shorten_to_sentence(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text

    content_limit = max(0, limit - len(BRIEF_CUT_MARK))
    prefix = text[:content_limit]
    sentence_end = max((match.end() for match in _SENTENCE_END.finditer(prefix)), default=0)
    line_end = prefix.rfind("\n") + 1
    boundary = max(sentence_end, line_end)
    if boundary == 0:
        whitespace = max(prefix.rfind(" "), prefix.rfind("\t"))
        boundary = whitespace if whitespace > 0 else content_limit
    shortened = text[:boundary].rstrip()
    return shortened + BRIEF_CUT_MARK


REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")


def current_request_id() -> str:
    return REQUEST_ID.get()


def result_id(document_id: int, ordinal: int) -> str:
    return f"{document_id}{PIECE_SEPARATOR}{ordinal}"


def document_of(identifier: str) -> str:
    document, separator, piece = identifier.rpartition(PIECE_SEPARATOR)
    if separator and document and piece.isdigit():
        return document
    return identifier


def piece_of(identifier: str) -> int | None:
    document, separator, piece = identifier.rpartition(PIECE_SEPARATOR)
    if separator and document.isdigit() and piece.isdigit():
        return int(piece)
    return None


MORE_NOTE = (
    "ask the same question again with 'offset' set to 'next_offset' for the "
    "results after these, or ask a narrower question"
)

EXHAUSTED_NOTE = (
    "there are no results after this position at this deployment's candidate "
    "depth; raise 'candidate_depth' to look deeper, or ask a narrower question"
)


def _next_offset(diagnostics: "SearchDiagnostics | None", hits: Sequence[SearchHit]) -> int:
    start = diagnostics.offset if diagnostics is not None else 0
    return start + len(hits)


def more_section(omitted: Sequence[SearchHit], *, next_offset: int = 0) -> dict[str, Any]:
    return {
        "candidates": len(omitted),
        "documents": len({hit.provenance.document_id for hit in omitted}),
        "next_offset": next_offset,
        "note": MORE_NOTE,
    }


def exhausted_section(*, offset: int, ranked: int) -> dict[str, Any]:
    return {"candidates": 0, "documents": 0, "ranked": ranked, "offset": offset, "note": EXHAUSTED_NOTE}


def _omitted_of(outcome: SearchOutcome) -> tuple[SearchHit, ...]:
    return getattr(outcome, "omitted", ())


def filter_echo(diagnostics: SearchDiagnostics | None) -> dict[str, int] | None:
    if diagnostics is None or diagnostics.documents_considered is None:
        return None
    return {
        "documents_considered": diagnostics.documents_considered,
        "pieces_considered": diagnostics.pieces_considered,
    }


def empty_payload(diagnostics: SearchDiagnostics | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": STATUS_NOTHING_MATCHED, "results": []}
    if diagnostics is not None and diagnostics.offset:
        payload["more"] = exhausted_section(offset=diagnostics.offset, ranked=diagnostics.offset)
    echo = filter_echo(diagnostics)
    if echo is not None:
        payload["filter"] = echo
    return payload


def failure_payload(reason: str, diagnostics: SearchDiagnostics | None = None) -> dict[str, Any]:
    payload = {"status": STATUS_FAILED, "error": reason}
    identifier = current_request_id()
    if identifier:
        payload["request_id"] = identifier
    echo = filter_echo(diagnostics)
    if echo is not None:
        payload["filter"] = echo
    return payload


def passage_payload(hit: SearchHit, rank: int, audience: Audience = SERVICE) -> dict[str, Any]:
    where = hit.provenance
    identifier = result_id(where.document_id, where.ordinal) if where.ordinal is not None else str(hit.chunk_id)
    payload = {
        "id": identifier,
        "text": hit.content,
        "truncated": hit.truncated,
        "full_length": hit.full_length if hit.full_length is not None else len(hit.content),
        "page_start": where.page_start,
        "page_end": where.page_end,
        "heading_path": list(where.heading_path),
        "citation": where.citation(),
        "found_by": list(hit.found_by),
        "rank": rank,
    }
    if audience != ASSISTANT:
        payload["score"] = hit.score
    return payload


def _document_facts(where: Any, audience: Audience) -> dict[str, Any]:
    if audience == ASSISTANT:
        return {
            "id": where.external_id,
            "title": where.title or where.external_id,
            "url": where.link,
            "location": list(where.location),
        }
    return {
        "document": where.external_id,
        "title": where.title or where.external_id,
        "link": where.link,
        "location": list(where.location),
    }


def document_group_payload(
    hits: Sequence[SearchHit], audience: Audience = SERVICE
) -> list[dict[str, Any]]:
    order: list[int] = []
    passages: dict[int, list[dict[str, Any]]] = {}
    facts: dict[int, dict[str, Any]] = {}
    for rank, hit in enumerate(hits, start=1):
        where = hit.provenance
        if where.document_id not in facts:
            order.append(where.document_id)
            passages[where.document_id] = []
            facts[where.document_id] = _document_facts(where, audience)
        passages[where.document_id].append(passage_payload(hit, rank, audience))
    return [{**facts[document_id], "passages": passages[document_id]} for document_id in order]


def grouped_results_payload(
    hits: Sequence[SearchHit],
    omitted: Sequence[SearchHit] = (),
    audience: Audience = SERVICE,
    diagnostics: SearchDiagnostics | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": STATUS_FOUND,
        "results": document_group_payload(hits, audience),
    }
    if omitted:
        payload["more"] = more_section(omitted, next_offset=_next_offset(diagnostics, hits))
    elif diagnostics is not None and diagnostics.offset and not hits:
        # Asked past the end of the ranking: an explicit empty with its reason,
        # never a failure and never a silent deeper scan (AC-299).
        payload["more"] = exhausted_section(offset=diagnostics.offset, ranked=diagnostics.offset)
    echo = filter_echo(diagnostics)
    if echo is not None:
        payload["filter"] = echo
    return payload


def grouped_search_payload(
    outcome: SearchOutcome, audience: Audience = SERVICE
) -> dict[str, Any]:
    if isinstance(outcome, Found):
        return grouped_results_payload(outcome.hits, _omitted_of(outcome), audience, outcome.diagnostics)
    if isinstance(outcome, FilterMatchedNothing):
        payload = empty_payload(outcome.diagnostics)
        payload["status"] = STATUS_FILTER_MATCHED_NOTHING
        payload["clause"] = outcome.clause
        return payload
    if isinstance(outcome, NothingMatched):
        return empty_payload(outcome.diagnostics)
    if isinstance(outcome, Failed):
        return failure_payload(outcome.reason, outcome.diagnostics)
    raise TypeError(f"not a search outcome: {outcome!r}")


def document_view_payload(hits: Sequence[SearchHit]) -> list[dict[str, Any]]:
    order: list[int] = []
    scores: dict[int, list[float]] = {}
    facts: dict[int, dict[str, Any]] = {}
    evidence: dict[int, str] = {}
    for hit in hits:
        where = hit.provenance
        if where.document_id not in facts:
            order.append(where.document_id)
            scores[where.document_id] = []
            evidence[where.document_id] = where.citation()
            facts[where.document_id] = {
                "document": where.external_id,
                "title": where.title or where.external_id,
                "link": where.link,
                "location": list(where.location),
            }
        scores[where.document_id].append(hit.score)
    return [
        {
            **facts[document_id],
            "score": max(scores[document_id]),
            "average_score": sum(scores[document_id]) / len(scores[document_id]),
            "match_count": len(scores[document_id]),
            "evidence": evidence[document_id],
        }
        for document_id in order
    ]


def documents_results_payload(
    hits: Sequence[SearchHit],
    omitted: Sequence[SearchHit] = (),
    diagnostics: SearchDiagnostics | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": STATUS_FOUND, "results": document_view_payload(hits)}
    if omitted:
        payload["more"] = more_section(omitted, next_offset=_next_offset(diagnostics, hits))
    elif diagnostics is not None and diagnostics.offset and not hits:
        payload["more"] = exhausted_section(offset=diagnostics.offset, ranked=diagnostics.offset)
    echo = filter_echo(diagnostics)
    if echo is not None:
        payload["filter"] = echo
    return payload


def documents_search_payload(outcome: SearchOutcome) -> dict[str, Any]:
    if isinstance(outcome, Found):
        return documents_results_payload(outcome.hits, _omitted_of(outcome), outcome.diagnostics)
    if isinstance(outcome, FilterMatchedNothing):
        payload = empty_payload(outcome.diagnostics)
        payload["status"] = STATUS_FILTER_MATCHED_NOTHING
        payload["clause"] = outcome.clause
        return payload
    if isinstance(outcome, NothingMatched):
        return empty_payload(outcome.diagnostics)
    if isinstance(outcome, Failed):
        return failure_payload(outcome.reason, outcome.diagnostics)
    raise TypeError(f"not a search outcome: {outcome!r}")


def document_payload(
    *,
    external_id: str,
    title: str,
    piece_count: int,
    offset: int,
    pieces: Sequence[Mapping[str, Any]],
    outline: Sequence[Mapping[str, Any]] | None = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": STATUS_FOUND,
        "id": external_id,
        "title": title or external_id,
        "piece_count": piece_count,
        "offset": offset,
        "pieces": [dict(piece) for piece in pieces],
    }
    return _with_outline(payload, outline)


def _with_outline(payload: dict[str, Any], outline: "Sequence[Mapping[str, Any]] | None") -> dict[str, Any]:
    if outline is None:
        return payload
    payload["outline"] = [dict(entry) for entry in outline]
    return payload


def not_found_payload(identifier: str) -> dict[str, Any]:
    return {
        "status": STATUS_NOTHING_MATCHED,
        "id": identifier,
        "pieces": [],
    }
