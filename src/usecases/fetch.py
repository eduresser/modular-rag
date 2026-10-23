from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import psycopg

from src.config import AppConfig
from src.db import DatabaseUnavailable, connect
from src.filters import FilterNode, compile_filter
from src.responses import ASSISTANT, SERVICE, Audience, document_of, piece_of
from src.store import DOCUMENT_STATE_GONE, ModelStamp
from src.store_versions import NoActiveVersion, active_version


class FetchError(ValueError):
    ...


def ceiling_for(config: AppConfig, audience: Audience) -> int:
    if audience == ASSISTANT:
        return config.mcp.limits.fetch.limit
    return config.api.limits.fetch.limit


def span_for(config: AppConfig, *, audience: Audience, named_piece: int | None) -> int:
    ceiling = ceiling_for(config, audience)
    surface = config.mcp if audience == ASSISTANT else config.api
    if named_piece is None:
        return min(surface.defaults.fetch.limit, ceiling)
    return min(surface.defaults.fetch.passage_window, ceiling)


@dataclass(frozen=True, slots=True)
class DocumentPiece:

    ordinal: int
    text: str
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
        }

    def location(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
        }


@dataclass(frozen=True, slots=True)
class FetchedDocument:

    external_id: str
    title: str
    piece_count: int
    offset: int
    pieces: tuple[DocumentPiece, ...]
    outline: tuple["OutlineEntry", ...] = ()
    outline_asked: bool = True


@dataclass(frozen=True, slots=True)
class NotHeld:

    identifier: str


@dataclass(frozen=True, slots=True)
class FetchFailed:

    reason: str


FetchOutcome = FetchedDocument | NotHeld | FetchFailed


def window(
    config: AppConfig, *, offset: int, limit: int | None, audience: Audience = SERVICE
) -> tuple[int, int]:
    ceiling = ceiling_for(config, audience)
    setting = "mcp.limits.fetch.limit" if audience == ASSISTANT else "api.limits.fetch.limit"
    if offset < 0:
        raise FetchError(f"a window cannot start before the first piece, got offset {offset}")
    if limit is None:
        return offset, ceiling
    if limit < 1:
        raise FetchError(f"a window of {limit} pieces returns nothing; ask for at least one")
    if limit > ceiling:
        raise FetchError(
            f"a window of {limit} pieces is above this deployment's maximum of {ceiling} "
            f"({setting}); the request is refused rather than reduced, "
            "because a silently shortened document reads as a complete one"
        )
    return offset, limit


def page_range(page_start: int | None, page_end: int | None) -> tuple[int, int]:
    if page_start is None or page_end is None:
        raise FetchError("a page range needs both its first and its last page")
    if page_start < 1:
        raise FetchError(f"a page range cannot start before page 1, got {page_start}")
    if page_end < page_start:
        raise FetchError(f"page range {page_start}..{page_end} runs backwards")
    return page_start, page_end


def _resolve_document(cursor: psycopg.Cursor, document_key: str) -> tuple[object, ...] | None:
    if document_key.isdigit():
        cursor.execute(
            "SELECT id, title, state, external_id FROM documents WHERE id = %s",
            (int(document_key),),
        )
        row = cursor.fetchone()
        if row is not None:
            return row
    cursor.execute(
        "SELECT id, title, state, external_id FROM documents WHERE external_id = %s",
        (document_key,),
    )
    return cursor.fetchone()


def fetch_on(
    conn: psycopg.Connection,
    identifier: str,
    *,
    config: AppConfig,
    offset: int = 0,
    limit: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    audience: Audience = SERVICE,
    stamp: ModelStamp | None = None,
    with_outline: bool = True,
) -> FetchOutcome:
    if not identifier.strip():
        raise FetchError("a fetch needs an identifier; there is no document called nothing")
    document_key = document_of(identifier)
    named_ordinal = piece_of(identifier)
    by_page = page_start is not None or page_end is not None
    if by_page:
        first_page, last_page = page_range(page_start, page_end)
    else:
        start, count = window(
            config,
            offset=offset + (named_ordinal or 0),
            limit=limit if limit is not None else span_for(
                config, audience=audience, named_piece=named_ordinal
            ),
            audience=audience,
        )
    scope = stamp or active_version(conn)
    with conn.cursor() as cursor:
        row = _resolve_document(cursor, document_key)
        if row is None or row[2] == DOCUMENT_STATE_GONE:
            return NotHeld(identifier=document_key)
        document_id, title, _, external_id = int(row[0]), row[1], row[2], row[3]
        cursor.execute(
            """
            SELECT count(*) FROM chunks
             WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
            """,
            (document_id, scope.model, scope.dimension),
        )
        (piece_count,) = cursor.fetchone()
        if by_page:
            cursor.execute(
                """
                SELECT count(*) FROM chunks
                 WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                   AND (page_start IS NOT NULL OR page_end IS NOT NULL)
                """,
                (document_id, scope.model, scope.dimension),
            )
            (paginated_pieces,) = cursor.fetchone()
            if paginated_pieces == 0:
                raise FetchError(
                    f"document {external_id!r} carries no pagination — its reader returned no "
                    "page for any of its pieces; ask for it by piece offset instead"
                )
            cursor.execute(
                """
                  SELECT ordinal, content, page_start, page_end, heading_path
                    FROM chunks
                   WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                     AND page_start IS NOT NULL AND page_end IS NOT NULL
                     AND page_start <= %s AND page_end >= %s
                ORDER BY ordinal
                """,
                (document_id, scope.model, scope.dimension, last_page, first_page),
            )
        else:
            cursor.execute(
                """
                  SELECT ordinal, content, page_start, page_end, heading_path
                    FROM chunks
                   WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                ORDER BY ordinal
                  OFFSET %s LIMIT %s
                """,
                (document_id, scope.model, scope.dimension, start, count),
            )
        pieces = tuple(
            DocumentPiece(
                ordinal=int(piece[0]),
                text=piece[1],
                page_start=piece[2],
                page_end=piece[3],
                heading_path=tuple(piece[4] or ()),
            )
            for piece in cursor.fetchall()
        )
    outline = outline_on(conn, document_key, stamp=scope) if with_outline else None
    return FetchedDocument(
        external_id=external_id,
        title=title,
        piece_count=int(piece_count),
        offset=pieces[0].ordinal if by_page and pieces else (0 if by_page else start),
        pieces=pieces,
        outline=outline.entries if isinstance(outline, Outline) else (),
        outline_asked=with_outline,
    )


@dataclass(frozen=True, slots=True)
class OutlineEntry:

    ordinal: int
    label: str
    ordinal_end: int | None = None

    def payload(self) -> dict[str, object]:
        shape: dict[str, object] = {"ordinal": self.ordinal, "label": self.label}
        if self.ordinal_end is not None:
            shape["ordinal_end"] = self.ordinal_end
        return shape


@dataclass(frozen=True, slots=True)
class Outline:

    basis: str
    entries: tuple[OutlineEntry, ...]
    narrowed_by: tuple[str, ...] = ()
    entries_total: int | None = None

    def payload(self) -> dict[str, object]:
        shape: dict[str, object] = {
            "basis": self.basis,
            "entries": [entry.payload() for entry in self.entries],
        }
        if self.narrowed_by:
            shape["narrowed_by"] = list(self.narrowed_by)
            shape["entries_total"] = self.entries_total
        return shape


OutlineOutcome = Outline | NotHeld


def _runs_of(positions: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    for ordinal, value in positions:
        if runs and value - ordinal == runs[-1][3] - runs[-1][1] and ordinal == runs[-1][1] + 1:
            first_ordinal, _, first_value, _ = runs[-1]
            runs[-1] = (first_ordinal, ordinal, first_value, value)
            continue
        runs.append((ordinal, ordinal, value, value))
    return runs


def _collapsed(runs: list[tuple[int, int, int, int]], noun: str) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []
    for first_ordinal, last_ordinal, first_value, last_value in runs:
        if first_ordinal == last_ordinal:
            entries.append(OutlineEntry(ordinal=first_ordinal, label=f"{noun} {first_value}"))
            continue
        entries.append(
            OutlineEntry(
                ordinal=first_ordinal,
                label=f"{noun}s {first_value}\u2013{last_value}",
                ordinal_end=last_ordinal,
            )
        )
    return entries


def _outline_from(
    pieces: list[tuple[int, int | None, int | None, list[str] | None]],
) -> Outline:
    headings: list[OutlineEntry] = []
    last_heading: tuple[str, ...] | None = None
    for ordinal, _, _, heading_path in pieces:
        path = tuple(heading_path or ())
        if path and path != last_heading:
            headings.append(OutlineEntry(ordinal=ordinal, label=" \u203a ".join(path)))
            last_heading = path
    if headings:
        return Outline(basis="headings", entries=tuple(headings))

    pages: list[tuple[int, int]] = []
    last_page: int | None = None
    for ordinal, page_start, _, _ in pieces:
        if page_start is not None and page_start != last_page:
            pages.append((ordinal, page_start))
            last_page = page_start
    if pages:
        return Outline(basis="pages", entries=tuple(_collapsed(_runs_of(pages), "page")))

    return Outline(
        basis="pieces",
        entries=tuple(
            _collapsed(_runs_of([(ordinal, ordinal) for ordinal, *_ in pieces]), "piece")
        ),
    )


def _narrowed(
    outline: Outline,
    paths: list[tuple[str, ...]],
    *,
    depth: int | None,
    under: tuple[str, ...] | None,
) -> Outline:
    if depth is None and under is None:
        return outline
    if outline.basis != "headings":
        return outline
    total = len(outline.entries)
    kept: list[OutlineEntry] = []
    narrowed_by: list[str] = []
    if depth is not None:
        narrowed_by.append(f"depth={depth}")
    if under is not None:
        narrowed_by.append("under=" + " \u203a ".join(under))
    for entry, path in zip(outline.entries, paths):
        if depth is not None and len(path) > depth:
            continue
        if under is not None and path[: len(under)] != under:
            continue
        kept.append(entry)
    return Outline(
        basis=outline.basis,
        entries=tuple(kept),
        narrowed_by=tuple(narrowed_by),
        entries_total=total,
    )


def outline_on(
    conn: psycopg.Connection,
    identifier: str,
    *,
    stamp: ModelStamp | None = None,
    depth: int | None = None,
    under: "tuple[str, ...] | None" = None,
    clause: "FilterNode | None" = None,
) -> OutlineOutcome:
    document_key = document_of(identifier)
    scope = stamp or active_version(conn)
    with conn.cursor() as cursor:
        row = _resolve_document(cursor, document_key)
        if row is None or row[2] == DOCUMENT_STATE_GONE:
            return NotHeld(identifier=document_key)
        document_id = int(row[0])
        compiled = compile_filter(clause)
        narrowing, parameters = compiled if compiled else ("", [])
        cursor.execute(
            f"""
              SELECT ordinal, page_start, page_end, heading_path
                FROM chunks AS c
               WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                 {"AND " + narrowing if narrowing else ""}
            ORDER BY ordinal
            """,
            (document_id, scope.model, scope.dimension, *parameters),
        )
        pieces = cursor.fetchall()
        if clause is not None:
            cursor.execute(
                """
                  SELECT count(*) FROM chunks
                   WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                """,
                (document_id, scope.model, scope.dimension),
            )
            whole = int(cursor.fetchone()[0])
    outline = _outline_from(pieces)
    paths: list[tuple[str, ...]] = []
    seen: tuple[str, ...] | None = None
    for _, _, _, heading_path in pieces:
        path = tuple(heading_path or ())
        if path and path != seen:
            paths.append(path)
            seen = path
    narrowed = _narrowed(outline, paths, depth=depth, under=under)
    if clause is None:
        return narrowed
    return Outline(
        basis=narrowed.basis,
        entries=narrowed.entries,
        narrowed_by=(*narrowed.narrowed_by, "filter"),
        entries_total=narrowed.entries_total if narrowed.entries_total is not None else whole,
    )


def fetch(
    identifier: str,
    *,
    config: AppConfig,
    offset: int = 0,
    limit: int | None = None,
    with_outline: bool = True,
    page_start: int | None = None,
    page_end: int | None = None,
    audience: Audience = SERVICE,
    connection: psycopg.Connection | None = None,
) -> FetchOutcome:
    def read(conn: psycopg.Connection) -> FetchOutcome:
        return fetch_on(
            conn,
            identifier,
            config=config,
            offset=offset,
            limit=limit,
            with_outline=with_outline,
            page_start=page_start,
            page_end=page_end,
            audience=audience,
        )

    if connection is not None:
        return _guarded(lambda: read(connection))
    try:
        with connect(config.db.url) as conn:
            return _guarded(lambda: read(conn))
    except DatabaseUnavailable as exc:
        return FetchFailed(reason=str(exc))


def _guarded(read: Callable[[], FetchOutcome]) -> FetchOutcome:
    try:
        return read()
    except NoActiveVersion as exc:
        return FetchFailed(reason=str(exc))
    except psycopg.Error as exc:
        return FetchFailed(reason=f"the archive refused the query: {exc}".strip())
