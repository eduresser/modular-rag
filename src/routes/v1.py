"""The conventional interface: the same two operations, with the knobs exposed.

Our own back-end services get what the assistant does not — every retrieval
knob, per request (AC-134) — because they are the ones tuning by measurement and
a deployment is the wrong unit to tune at. What they do **not** get is a second
implementation: both routes call the use cases in ``app.usecases``, so the tool
surface and this one cannot answer the same question differently.

**An out-of-range knob is a 422 naming the limit** (AC-135). It is the caller's
own request that is wrong, and it is the one thing here that is not shaped as a
search outcome: a caller that sent a number this deployment refuses has to see a
refusal, not an empty result set that reads as a thin corpus.

**The three outcomes keep their meaning at the status code too.** Found and
nothing-found are both 200 — the archive answered, and one of the answers is
"not here" (AC-133). A failure is 502: the archive or a provider is what broke,
not the request, and a caller retrying a 502 is doing the right thing while a
caller retrying a 400 is not.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.db import DatabaseUnavailable, connect
from src.filters import FilterError, FilterNode, validate_filter
from src.responses import (
    STATUS_FOUND,
    STATUS_NOTHING_MATCHED,
    document_payload,
    documents_search_payload,
    failure_payload,
    grouped_search_payload,
    not_found_payload,
)
from src.search_result import Failed
from src.usecases.vocabulary import published_vocabulary
from src.store_versions import NoActiveVersion
from src.usecases.fetch import (
    FetchedDocument,
    FetchError,
    FetchFailed,
    NotHeld,
    Outline,
    fetch,
    outline_on,
)
from src.usecases import inventory
from src.usecases.inventory import InventoryError
from src.usecases.inventory import NotHeld as InventoryNotHeld
from src.analytics.latency import DEFAULT_PERCENTILE
from src.usecases.reach import ReachError, ReachUnavailable, corpus_reach, search_latency
from src.usecases.search import KnobOutOfRange, SearchQuery, run_search
from src.usecases.serving import ServingUnavailable, serving
from src.usecases.spend import SpendError, SpendUnavailable, spend
from src.usecases.sync import (
    SyncBusy,
    SyncUnavailable,
    request_sync,
    sync_history,
    sync_status,
)

router = APIRouter(prefix="/v1", tags=["v1"])

CALLER = "v1"

ANSWERED = 200
REFUSED = 422
UPSTREAM_FAILED = 502


class OutlineBody(BaseModel):
    """The one thing an outline takes that does not fit in a query string.

    A filter is a nested shape, and a nested shape in a query string is a
    second encoding for something the archive already publishes one grammar
    for (AC-304). The route stays a GET with a body, which is the lesser of the
    two oddities.
    """

    model_config = ConfigDict(extra="forbid")

    filter: dict[str, Any] | None = None


class SearchBody(BaseModel):
    """One question and the knobs for it, every knob optional (AC-134).

    ``extra="forbid"`` for the same reason the settings namespaces have it: a
    caller that misspells a knob is tuning against a value that was never
    applied, and silence is what makes that measurement look real.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    caller: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1)
    dense_weight: float | None = Field(default=None, ge=0)
    lexical_weight: float | None = Field(default=None, ge=0)
    candidate_depth: int | None = Field(default=None, ge=1)
    index_search_effort: int | None = Field(default=None, ge=1)
    rerank: bool | None = None
    brief: bool | None = None
    view: Literal["passages", "documents"] = "passages"
    filter: dict[str, Any] | None = None

    def query(self, filter_node: FilterNode | None) -> SearchQuery:
        return SearchQuery(
            question=self.question,
            caller=self.caller or CALLER,
            offset=self.offset,
            result_count=self.limit,
            dense_weight=self.dense_weight,
            lexical_weight=self.lexical_weight,
            candidate_depth=self.candidate_depth,
            index_search_effort=self.index_search_effort,
            rerank=self.rerank,
            brief=self.brief,
            filter=filter_node,
        )


def deployment(request: Request):
    """This process's wiring, put on the application at startup."""
    return request.app.state.service


class SyncBody(BaseModel):
    """A full sweep uses the deployment's settings; no request overrides."""

    model_config = ConfigDict(extra="forbid")


@router.post("/sync", status_code=202, responses={401: {}, 409: {}, 503: {}})
def sync_route(
    service: Annotated[Any, Depends(deployment)],
    body: SyncBody | None = None,
) -> Any:
    """Queue a full synchronisation for the worker and return its sync_id.

    Accepts no body or {}. Existing indexed documents retain normal change
    detection. Poll GET /v1/sync/{sync_id} for completion and counters.
    """
    result = request_sync(service.config)
    if isinstance(result, SyncUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    if isinstance(result, SyncBusy):
        payload = failure_payload("A synchronisation is already queued or running.")
        if result.active_sync_id is not None:
            payload["sync_id"] = str(result.active_sync_id)
        return JSONResponse(status_code=409, content=payload)
    return JSONResponse(
        status_code=202, content=result,
        headers={"Location": f"/v1/sync/{result['sync_id']}"},
    )


@router.get("/sync", responses={401: {}, 503: {}})
def sync_history_route(
    service: Annotated[Any, Depends(deployment)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Any:
    """What is running now, what last finished, and what ran before (AC-256, AC-257).

    Registered ahead of ``/sync/{sync_id}``: that path takes a UUID, so a bare
    ``/v1/sync`` would not match it — but the pair reads as one resource and
    keeping them adjacent is what stops the next path from being added between
    them.
    """
    result = sync_history(service.config, limit=limit)
    if isinstance(result, SyncUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    return result.payload()


@router.get("/sync/{sync_id}", responses={401: {}, 404: {}, 503: {}})
def sync_status_route(
    sync_id: UUID,
    service: Annotated[Any, Depends(deployment)],
) -> Any:
    """Read a synchronisation's queued/running/final status and recorded counters."""
    result = sync_status(service.config, sync_id)
    if isinstance(result, SyncUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    if result is None:
        return JSONResponse(status_code=404, content=failure_payload("Synchronisation not found."))
    return result


@router.get("/serving", responses={401: {}, 503: {}})
def serving_route(service: Annotated[Any, Depends(deployment)]) -> Any:
    """Which model, which width and which build are serving (AC-262).

    Not on ``/healthz``: that route must touch nothing (AC-138), and this
    touches the archive.
    """
    result = serving(service.config)
    if isinstance(result, ServingUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    return result.payload()


@router.get("/corpus/reach", responses={401: {}, 422: {}, 503: {}})
def corpus_reach_route(
    service: Annotated[Any, Depends(deployment)],
    since: datetime | None = None,
    until: datetime | None = None,
    limit: Annotated[int | None, Query(ge=1)] = None,
) -> Any:
    """What the archive never returned, and what it can never return (AC-260)."""
    try:
        result = corpus_reach(service.config, since=since, until=until, limit=limit)
    except ReachError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    if isinstance(result, ReachUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    return result.payload()


@router.get("/searches/latency", responses={401: {}, 422: {}, 503: {}})
def search_latency_route(
    service: Annotated[Any, Depends(deployment)],
    since: datetime | None = None,
    until: datetime | None = None,
    percentile: float = DEFAULT_PERCENTILE,
) -> Any:
    """What a search cost in time, against the effort it ran at (AC-261)."""
    try:
        result = search_latency(service.config, since=since, until=until, percentile=percentile)
    except ReachError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    if isinstance(result, ReachUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    return result.payload()


@router.get("/usage", responses={401: {}, 422: {}, 503: {}})
def usage_route(
    service: Annotated[Any, Depends(deployment)],
    by: str = "operation",
    since: datetime | None = None,
    until: datetime | None = None,
) -> Any:
    """What this deployment has spent, grouped, with what it could not price.

    Only one of the eight managed retrieval platforms surveyed answers this
    over its API; the rest send an operator to a vendor console. There is none
    here, so this is not a convenience.
    """
    try:
        result = spend(service.config, by=by, since=since, until=until)
    except SpendError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    if isinstance(result, SpendUnavailable):
        return JSONResponse(status_code=503, content=failure_payload(result.error))
    return result.payload()


@router.post("/search")
def search_route(
    body: SearchBody,
    service: Annotated[Any, Depends(deployment)],
) -> Any:
    """Search the archive with this request's knobs over the deployment's defaults.

    Every answer is the shaped payload itself, at whichever status code fits —
    never FastAPI's ``{"detail": ...}`` envelope. A consumer that has to unwrap
    an error differently from a success is a consumer with two parsers, and the
    second one is written the day the first failure happens in production.

    A malformed filter is refused here, at 422, naming the offending clause
    (AC-284) — the caller's own mistake, at the caller's own status code, the
    same as an out-of-range knob and before the use case ever runs.
    """
    limits = service.config.search
    try:
        filter_node = validate_filter(
            body.filter,
            max_clauses=limits.max_filter_clauses,
            max_depth=limits.max_filter_depth,
            max_pattern_length=limits.max_pattern_length,
        )
    except FilterError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    try:
        outcome = run_search(
            body.query(filter_node),
            config=service.config,
            embedder=service.embedder,
            scorer=service.scorer,
        )
    except KnobOutOfRange as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    shape = documents_search_payload if body.view == "documents" else grouped_search_payload
    payload = shape(outcome)
    if isinstance(outcome, Failed):
        return JSONResponse(status_code=UPSTREAM_FAILED, content=payload)
    return payload


@router.get("/filters")
def filters_route(service: Annotated[Any, Depends(deployment)]) -> Any:
    """The archive's filter vocabulary: every field, its type, its operators,
    and — for a field this service or the reader fixes — the values this
    archive actually holds, counted (AC-287, AC-288). A field whose values
    are produced by the documents instead reports ``"values": "open"``,
    naming no value at all rather than a list nobody could read to the end
    (AC-292). The same table ``/v1/search`` validates a filter against, so a
    filter built from this response is never refused for naming a field or an
    operator this deployment does not have.
    """
    try:
        with connect(service.config.db.url) as conn:
            published = published_vocabulary(conn)
    except DatabaseUnavailable as exc:
        return JSONResponse(status_code=UPSTREAM_FAILED, content=failure_payload(str(exc)))
    return {
        "fields": [
            {
                "name": field.name,
                "type": field.type,
                "operators": list(field.operators),
                **(
                    {
                        "values": [
                            {"value": value.value, "document_count": value.document_count}
                            for value in field.values
                        ]
                    }
                    if field.values is not None
                    else {"values": "open"}
                    if field.values_open
                    else {}
                ),
            }
            for field in published
        ]
    }


@router.get("/documents")
def documents_route(
    service: Annotated[Any, Depends(deployment)],
    limit: Annotated[int | None, Query(ge=1)] = None,
    cursor: str | None = None,
) -> Any:
    """What the archive holds, a page at a time (AC-254).

    Registered ahead of every ``{identifier:path}`` route: a bare
    ``/v1/documents`` would otherwise match one of those with an empty
    identifier, and answer "no such document" to a question about all of them.
    """
    try:
        page = inventory.documents(service.config, limit=limit, cursor=cursor)
    except InventoryError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    except DatabaseUnavailable as exc:
        return JSONResponse(status_code=UPSTREAM_FAILED, content=failure_payload(str(exc)))
    return page.payload()


@router.get("/documents/{identifier:path}/status")
def document_status_route(
    identifier: str,
    service: Annotated[Any, Depends(deployment)],
) -> Any:
    """What the reader concluded about one document, and why (AC-255).

    Registered ahead of ``document_route`` for the reason ``/outline`` already
    is: both paths are ``:path``, so a request ending in ``/status`` would
    otherwise fold the suffix into the identifier and fetch a document nobody
    asked for.
    """
    try:
        outcome = inventory.document_state(service.config, identifier)
    except InventoryError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    except DatabaseUnavailable as exc:
        return JSONResponse(status_code=UPSTREAM_FAILED, content=failure_payload(str(exc)))
    if isinstance(outcome, InventoryNotHeld):
        return not_found_payload(outcome.identifier)
    return outcome.payload()


@router.get("/documents/{identifier:path}/outline")
def outline_route(
    identifier: str,
    service: Annotated[Any, Depends(deployment)],
    depth: Annotated[int | None, Query(ge=1)] = None,
    under: Annotated[str | None, Query()] = None,
    body: OutlineBody | None = None,
) -> Any:
    """A document's outline, and what it was built from (AC-194, AC-195).

    Registered ahead of ``document_route``: both paths are declared
    ``:path`` (a source's own identifier is not ours to constrain), so a
    request ending in ``/outline`` would otherwise match the bare document
    route first, with ``/outline`` folded into the identifier itself. FastAPI
    matches routes in registration order, so this one has to come first.

    ``depth`` cuts it to a level and ``under`` to a branch (AC-302), and an
    outline narrowed either way says so (AC-303). ``under`` is written with the
    same separator the labels use — `Cláusula quarta › Do reajuste` — because
    that is the string a caller read off a search result's heading path, and
    asking it to be re-typed as a list is asking it to be re-typed wrongly.

    Costs no re-read and moves no fingerprint — the columns a piece already
    carries (``heading_path``, ``page_start``) are grouped in place by
    :func:`app.usecases.fetch.outline_on`.
    """
    limits = service.config.search
    try:
        clause = validate_filter(
            body.filter if body else None,
            max_clauses=limits.max_filter_clauses,
            max_depth=limits.max_filter_depth,
            max_pattern_length=limits.max_pattern_length,
        )
    except FilterError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    try:
        with connect(service.config.db.url) as conn:
            outcome = outline_on(
                conn,
                identifier,
                depth=depth,
                under=tuple(segment.strip() for segment in under.split("\u203a")) if under else None,
                clause=clause,
            )
    except DatabaseUnavailable as exc:
        return JSONResponse(status_code=UPSTREAM_FAILED, content=failure_payload(str(exc)))
    except NoActiveVersion as exc:
        return JSONResponse(status_code=UPSTREAM_FAILED, content=failure_payload(str(exc)))
    except psycopg.Error as exc:
        return JSONResponse(
            status_code=UPSTREAM_FAILED,
            content=failure_payload(f"the archive refused the query: {exc}".strip()),
        )
    if isinstance(outcome, NotHeld):
        return {"status": STATUS_NOTHING_MATCHED, "id": outcome.identifier, "basis": None, "entries": []}
    assert isinstance(outcome, Outline)
    return {
        "status": STATUS_FOUND,
        "id": identifier,
        "basis": outcome.basis,
        "entries": [entry.payload() for entry in outcome.entries],
    }


@router.get("/documents/{identifier:path}")
def document_route(
    identifier: str,
    service: Annotated[Any, Depends(deployment)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int | None, Query(ge=1)] = None,
    page_start: Annotated[int | None, Query(ge=1)] = None,
    page_end: Annotated[int | None, Query(ge=1)] = None,
    outline: Annotated[bool, Query()] = True,
) -> Any:
    """One document's pieces in order, windowed (Q-060) or by page range (AC-192).

    The path is declared ``:path`` because a source's own identifier is not
    ours to constrain — a Drive file id is safe, a future source's may contain a
    slash, and a route that truncates one silently fetches a different document.

    ``page_start``/``page_end`` open the window by page instead of by piece
    offset — the same knob :func:`app.usecases.fetch.fetch_on` already carries,
    exposed here because reading a document by its own structure is this
    surface's own capability and not the assistant's (AC-130). A document with
    no pagination refuses with its reason (AC-193), which arrives as a
    :class:`FetchError` like any other malformed request.
    """
    try:
        outcome = fetch(
            identifier,
            config=service.config,
            offset=offset,
            limit=limit,
            page_start=page_start,
            page_end=page_end,
            with_outline=outline,
        )
    except FetchError as exc:
        return JSONResponse(status_code=REFUSED, content=failure_payload(str(exc)))
    if isinstance(outcome, FetchFailed):
        return JSONResponse(
            status_code=UPSTREAM_FAILED, content=failure_payload(outcome.reason)
        )
    if isinstance(outcome, NotHeld):
        return not_found_payload(outcome.identifier)
    return _document(outcome)


def _document(found: FetchedDocument) -> dict[str, Any]:
    return document_payload(
        external_id=found.external_id,
        title=found.title,
        piece_count=found.piece_count,
        offset=found.offset,
        pieces=[piece.payload() for piece in found.pieces],
        outline=[entry.payload() for entry in found.outline] if found.outline_asked else None,
    )
