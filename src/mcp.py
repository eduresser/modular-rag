from __future__ import annotations

import json
import logging
from typing import Any, NotRequired, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent, ToolAnnotations

from src.config import AppConfig
from src.db import connect
from src.filters import FIELDS, validate_filter
from src.responses import (
    ASSISTANT,
    STATUS_FOUND,
    STATUS_NOTHING_MATCHED,
    document_of,
    document_payload,
    failure_payload,
    grouped_search_payload,
    not_found_payload,
)
from src.usecases.fetch import FetchedDocument, FetchFailed, NotHeld, fetch, outline_on
from src.usecases.search import SearchQuery, run_search
from src.usecases.vocabulary import published_vocabulary

logger = logging.getLogger(__name__)

SERVER_NAME = "modular-rag"

CALLER = "mcp"

READ_ONLY = ToolAnnotations(read_only_hint=True)


class _PassagePayload(TypedDict):
    """One passage inside its document's group — ``responses.passage_payload``.

    Carries no ``title``, ``link`` or ``location``: those are the document's
    and live once, on the group above it (AC-182). ``rank`` is its position in
    the ranking the fusion produced, before grouping rearranged anything
    (AC-183).

    It also carries no ``score`` (AC-250) — a float comparable only against the
    other scores in the same answer, which the ranking already expresses. The
    conventional routes keep it, because the callers there are tuning by
    measurement and that is the number they tune against (AC-251).
    """

    id: str
    text: str
    truncated: bool
    full_length: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    citation: str
    found_by: list[str]
    rank: int


class _DocumentGroupPayload(TypedDict):
    """One document, named once, with the passages of it that answered (AC-182)."""

    id: str
    title: str
    url: str | None
    location: list[str]
    passages: list[_PassagePayload]


class _MoreSectionPayload(TypedDict):
    """How much the result count left out, and what to do about it (AC-248, AC-249).

    Two numbers and a sentence. This used to be the candidates themselves, one
    object each — up to ninety of them, because the fused order is the union of
    two legs at the candidate depth. No document-search server surveyed returns
    the candidates it chose not to return; what the field returns is how much
    more there is, and the move that acts on it here is ``offset``, which the
    section carries as ``next_offset`` (AC-298).
    """

    candidates: int
    documents: int
    note: str


class _FilterEchoPayload(TypedDict):
    """How much of the archive a filter left for the search to rank (AC-283)."""

    documents_considered: int
    pieces_considered: int


class _FilterValuePayload(TypedDict):
    """One value this archive holds for a published field, and its document count."""

    value: str
    document_count: int


class _FilterFieldPayload(TypedDict):
    """One entry of the filter vocabulary — ``src.usecases.vocabulary.PublishedField``.

    ``values`` carries a counted list for a field this service or the reader
    fixes (AC-288), the string ``"open"`` for a field whose values grow with
    the corpus (AC-292), and is absent for a field with no notion of
    published values at all — the same three-way shape ``GET /v1/filters``
    answers with, because this tool reads the same use case rather than a
    second query (AC-293).
    """

    name: str
    type: str
    operators: list[str]
    values: NotRequired[list[_FilterValuePayload] | str]


class FiltersToolOutput(TypedDict):
    """What ``filters`` answers with (AC-293)."""

    status: str
    fields: NotRequired[list[_FilterFieldPayload]]
    error: NotRequired[str]


class _OutlineEntryPayload(TypedDict):
    """One section of an outline, and where it begins."""

    ordinal: int
    label: str
    ordinal_end: NotRequired[int]


class OutlineToolOutput(TypedDict):
    """What ``outline`` answers with: a map of a document and none of its text."""

    status: str
    id: NotRequired[str]
    basis: NotRequired[str]
    entries: NotRequired[list[_OutlineEntryPayload]]
    narrowed_by: NotRequired[list[str]]
    entries_total: NotRequired[int]
    error: NotRequired[str]


class SearchToolOutput(TypedDict):
    """What ``search`` answers with, in every one of the outcomes.

    ``results`` is absent on a failure and present-and-empty on nothing
    matched — the same distinction ``responses.py`` draws — so both are
    ``NotRequired`` rather than one being modelled as always there.
    ``filter`` and ``clause`` are present only when this call carried a
    filter (AC-283) — ``clause`` only on the outcome that says the filter
    itself matched nothing (AC-282), never on an unfiltered search.
    """

    status: str
    results: NotRequired[list[_DocumentGroupPayload]]
    more: NotRequired[_MoreSectionPayload]
    filter: NotRequired[_FilterEchoPayload]
    clause: NotRequired[str]
    error: NotRequired[str]


class _DocumentPiecePayload(TypedDict):
    """Where one piece of the window sits — and no text (AC-244).

    The text of the whole window is on ``FetchToolOutput.text``, once. This
    carries what joining destroys: which page each piece covers and the heading
    path it sat under.
    """

    ordinal: int
    page_start: int | None
    page_end: int | None
    heading_path: list[str]


class _OutlineEntryPayload(TypedDict):
    """One section of the whole document, and the piece it begins at (AC-252)."""

    ordinal: int
    label: str


class FetchToolOutput(TypedDict):
    """What ``fetch`` answers with (AC-199).

    ``id``, ``title``, ``text`` and ``link`` are the vendor's own contract;
    ``document``, ``link`` and ``location`` are the same document record a
    grouped answer's document group carries (AC-182, AC-197) — the facts a
    document is described by once, regardless of how it was reached — so a
    caller that fetched a document knows as much about it as one that found
    it. ``pieces`` carries the locations alongside the flat ``text`` rather
    than instead of it — a concatenated string alone loses the page each
    sentence came from (Q-060).
    """

    status: str
    id: NotRequired[str]
    title: NotRequired[str]
    text: NotRequired[str]
    url: NotRequired[str | None]
    location: NotRequired[list[str]]
    piece_count: NotRequired[int]
    offset: NotRequired[int]
    pieces: NotRequired[list[_DocumentPiecePayload]]
    outline: NotRequired[list[_OutlineEntryPayload]]
    error: NotRequired[str]


_FILTER_FIELDS_NOTE = ", ".join(f"{name} ({spec.type})" for name, spec in FIELDS.items())

SEARCH_DESCRIPTION = (
    "Search the firm's document archive. Pass the question as 'query'. "
    "Use it for any question about the contents of "
    "the archive. Returns 'results': one entry per DOCUMENT, each naming the document "
    "once and carrying a 'passages' list of bounded previews in relevance order ('rank' "
    "is the position across all documents). A passage with 'truncated: true' is incomplete; "
    "pass its 'id' to the fetch tool before relying on or quoting the omitted text. Every "
    "passage also has a 'citation' ready to quote. When the search found more than it "
    "returned, 'more' says how many candidates and how many documents were left out, "
    "and 'next_offset' where to continue: ask the same question again with 'offset' "
    "set to that value. How many results a page holds is this deployment's decision "
    "and is not an argument. "
    "'filter' narrows which documents and pieces are searched, evaluated before ranking "
    "rather than after — never widen a question with one, only narrow it. Shape: "
    "{'all'|'any'|'none': [{'field', 'op', 'value'}, ...]}, nestable, e.g. "
    "{'all': [{'field': 'title', 'op': 'glob', 'value': '*aditivo*'}, "
    "{'field': 'page', 'op': 'between', 'value': [30, 50]}]}. "
    f"Fields, each with its type: {_FILTER_FIELDS_NOTE}. Operators depend on the field's "
    "type (text: is/is_not/in/not_in/exists; text_searchable: also contains/starts_with/"
    "glob/regex; enum/enum_open: is/is_not/in/not_in; number/date: is/gt/gte/lt/lte/"
    "between/exists; text_list: has_any/has_all/has_none/path_starts_with/exists; "
    "range: between/gte/lte). A malformed filter comes back as 'status': 'failed', naming "
    "the offending clause; a filter that is valid but selects no document comes back as "
    "'status': 'filter_matched_nothing', naming the responsible clause."
)

FETCH_DESCRIPTION = (
    "Retrieve a document from the firm's archive by the 'id' of a search result. An id "
    "naming a passage ('123#7') returns that passage and its neighbours; an id naming a "
    "document ('123') starts at its beginning. 'offset' moves the window and 'limit' "
    "sets how many pieces it holds — a limit above this deployment's maximum is served "
    "short and says so under 'reduced', never refused. Returns 'piece_count' for the "
    "whole document, 'offset' for this window, and 'outline' listing every section with "
    "the piece it begins at — use it to jump straight to a clause instead of reading "
    "from the start, or pass 'outline': false to leave it out."
)

OUTLINE_DESCRIPTION = (
    "Where a document's sections begin, without any of its text. Takes the 'id' of a "
    "document (or of a passage — the document holding it is used). Returns 'basis' saying "
    "what the outline was built from — the document's own headings, or its pages when the "
    "reader gave none — and 'entries', each naming the piece it begins at, so a fetch can "
    "open that section directly. An entry with 'ordinal_end' stands for a run of pieces "
    "rather than one. 'depth' returns only the top levels of a deep outline and 'under' "
    "only the branch below a heading path, written the way a heading path is printed "
    "('Cláusula quarta › Do reajuste'); an outline narrowed either way says so in "
    "'narrowed_by' and gives the whole outline's size in 'entries_total'. Use it before "
    "fetching a long document, to fetch the part that answers the question instead of "
    "reading from the start. 'filter' narrows it by the same vocabulary 'search' takes — "
    "a clause over 'heading_path' answers 'which sections mention this' without reading "
    "any of them."
)

FILTERS_DESCRIPTION = (
    "List the filter vocabulary 'search' accepts in its 'filter' argument: every field, "
    "its type and the operators that type takes. A field whose values are fixed by this "
    "service or by the reader also carries the values this archive currently holds, each "
    "with the number of documents behind it; a field whose values are produced by the "
    "documents instead reports 'values': 'open', naming no value at all rather than a list "
    "nobody could read to the end. Call this before guessing a value for 'search' — it "
    "takes no arguments and never changes what a search would find."
)


def _answered(payload: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
    )


def build_server(service: Any, *, name: str = SERVER_NAME) -> MCPServer:
    server = MCPServer(name=name)

    @server.tool(name="search", description=SEARCH_DESCRIPTION, structured_output=True, annotations=READ_ONLY)
    def search(
        query: str, offset: int = 0, filter: dict[str, Any] | None = None  # noqa: A002
    ) -> SearchToolOutput:
        try:
            limits = service.config.search
            filter_node = validate_filter(
                filter,
                max_clauses=limits.max_filter_clauses,
                max_depth=limits.max_filter_depth,
                max_pattern_length=limits.max_pattern_length,
            )
            outcome = run_search(
                SearchQuery(
                    question=query,
                    caller=CALLER,
                    offset=offset,
                    result_count=service.config.mcp.defaults.search.limit,
                    brief=True,
                    filter=filter_node,
                ),
                config=service.config,
                embedder=service.embedder,
                scorer=service.scorer,
            )
        except Exception as exc:
            return _failed(exc, "search")
        return _answered(grouped_search_payload(outcome, ASSISTANT))

    @server.tool(name="fetch", description=FETCH_DESCRIPTION, structured_output=True, annotations=READ_ONLY)
    def fetch_document(
        id: str,  # noqa: A002 — the protocol names it
        offset: int = 0,
        limit: int | None = None,
        outline: bool = True,
    ) -> FetchToolOutput:
        reduced_to: int | None = None
        try:
            ceiling = service.config.mcp.limits.fetch.limit
            if limit is not None and limit > ceiling:
                reduced_to = ceiling
                limit = ceiling
            outcome = fetch(
                id,
                config=service.config,
                offset=offset,
                limit=limit,
                with_outline=outline,
                audience=ASSISTANT,
            )
            link, location = (
                _document_facts(service.config, outcome.external_id)
                if isinstance(outcome, FetchedDocument)
                else (None, ())
            )
        except Exception as exc:
            return _failed(exc, "fetch")
        if isinstance(outcome, FetchFailed):
            return _answered(failure_payload(outcome.reason))
        if isinstance(outcome, NotHeld):
            return _answered(_absent(outcome.identifier))
        answer = _document(outcome, link=link, location=location)
        if reduced_to is not None:
            answer["reduced"] = {
                "limit": reduced_to,
                "setting": "mcp.limits.fetch.limit",
                "note": (
                    "the window asked for is above this deployment's maximum; these are the "
                    f"first {reduced_to} pieces from 'offset' — ask again with a later "
                    "'offset' for the rest"
                ),
            }
        return _answered(answer)

    @server.tool(name="filters", description=FILTERS_DESCRIPTION, structured_output=True, annotations=READ_ONLY)
    def filters() -> FiltersToolOutput:
        try:
            with connect(service.config.db.url) as conn:
                published = published_vocabulary(conn)
        except Exception as exc:
            return _failed(exc, "filters")
        return _answered({"status": STATUS_FOUND, "fields": [_field_payload(field) for field in published]})

    @server.tool(name="outline", description=OUTLINE_DESCRIPTION, structured_output=True, annotations=READ_ONLY)
    def outline(
        id: str,  # noqa: A002 — as the fetch tool names it
        depth: int | None = None,
        under: str | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002 — the same word `search` uses
    ) -> OutlineToolOutput:
        try:
            limits = service.config.search
            clause = validate_filter(
                filter,
                max_clauses=limits.max_filter_clauses,
                max_depth=limits.max_filter_depth,
                max_pattern_length=limits.max_pattern_length,
            )
            with connect(service.config.db.url) as conn:
                outcome = outline_on(
                    conn,
                    id,
                    depth=depth,
                    under=tuple(segment.strip() for segment in under.split("\u203a")) if under else None,
                    clause=clause,
                )
        except Exception as exc:
            return _failed(exc, "outline")
        if isinstance(outcome, NotHeld):
            return _answered({"status": STATUS_NOTHING_MATCHED, "id": outcome.identifier, "entries": []})
        return _answered({"status": STATUS_FOUND, "id": document_of(id), **outcome.payload()})

    return server


def _failed(exc: Exception, tool: str) -> CallToolResult:
    logger.exception("the %s tool failed", tool)
    return _answered(
        failure_payload(f"the {tool} could not be completed: {type(exc).__name__}: {exc}")
    )


def _absent(identifier: str) -> dict[str, Any]:
    return {**not_found_payload(identifier), "piece_count": 0, "offset": 0}


def _field_payload(field: Any) -> dict[str, Any]:
    return {
        "name": field.name,
        "type": field.type,
        "operators": list(field.operators),
        **(
            {"values": [{"value": v.value, "document_count": v.document_count} for v in field.values]}
            if field.values is not None
            else {"values": "open"}
            if field.values_open
            else {}
        ),
    }


def _document_facts(config: AppConfig, external_id: str) -> tuple[str | None, tuple[str, ...]]:
    with connect(config.db.url) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT link, location FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    if row is None:
        return None, ()
    return row[0], tuple(row[1] or ())


def _document(found: FetchedDocument, *, link: str | None, location: tuple[str, ...]) -> dict[str, Any]:
    payload = document_payload(
        external_id=found.external_id,
        title=found.title,
        piece_count=found.piece_count,
        offset=found.offset,
        pieces=[piece.location() for piece in found.pieces],
        outline=[entry.payload() for entry in found.outline] if found.outline_asked else None,
    )
    return {
        **payload,
        "text": "\n\n".join(piece.text for piece in found.pieces),
        "url": link,
        "location": list(location),
    }
