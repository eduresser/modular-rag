from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

from src.config import AppConfig
from src.db import connect

MAX_PAGE = 200

DEFAULT_PAGE = 50


class InventoryError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class DocumentRow:

    id: int
    external_id: str
    title: str
    state: str
    link: str | None
    location: tuple[str, ...]
    piece_count: int
    indexed_at: datetime | None
    last_seen_at: datetime | None

    def payload(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "document": self.external_id,
            "title": self.title or self.external_id,
            "state": self.state,
            "link": self.link,
            "location": list(self.location),
            "piece_count": self.piece_count,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }


@dataclass(frozen=True, slots=True)
class DocumentPage:

    documents: tuple[DocumentRow, ...]
    total_count: int
    next_cursor: str | None

    def payload(self) -> dict[str, object]:
        return {
            "documents": [document.payload() for document in self.documents],
            "total_count": self.total_count,
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class DocumentState:

    external_id: str
    state: str
    read_state: str | None
    reader: str | None
    reader_version: str | None
    page_count: int | None
    byte_size: int | None
    failure_reason: str | None
    read_at: datetime | None
    parsed_uri: str | None

    def payload(self) -> dict[str, object]:
        return {
            "document": self.external_id,
            "state": self.state,
            "read": {
                "state": self.read_state,
                "reader": self.reader,
                "reader_version": self.reader_version,
                "page_count": self.page_count,
                "byte_size": self.byte_size,
                "failure_reason": self.failure_reason,
                "read_at": self.read_at.isoformat() if self.read_at else None,
            },
            "parsed_uri": self.parsed_uri,
        }


@dataclass(frozen=True, slots=True)
class NotHeld:

    identifier: str


def page_size(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_PAGE
    if limit < 1:
        raise InventoryError(f"a page of {limit} documents returns nothing; ask for at least one")
    if limit > MAX_PAGE:
        raise InventoryError(
            f"a page of {limit} documents is above this service's maximum of {MAX_PAGE}; "
            "the request is refused rather than reduced, because a silently shortened "
            "page reads as the end of the archive"
        )
    return limit


def _after(cursor: str | None) -> int:
    if cursor is None or not cursor.strip():
        return 0
    try:
        return int(cursor)
    except ValueError as exc:
        raise InventoryError(
            f"cursor {cursor!r} is not one this listing issued; pass back the "
            "`next_cursor` of the previous page, or omit it to start from the beginning"
        ) from exc


def list_on(
    conn: psycopg.Connection, *, limit: int | None = None, cursor: str | None = None
) -> DocumentPage:
    size = page_size(limit)
    after = _after(cursor)
    with conn.cursor() as db:
        db.execute("SELECT count(*) FROM documents")
        (total,) = db.fetchone()
        db.execute(
            """
              SELECT d.id, d.external_id, d.title, d.state, d.link, d.location,
                     (SELECT count(*) FROM chunks c WHERE c.document_id = d.id),
                     i.indexed_at, i.last_seen_at
                FROM documents d
           LEFT JOIN document_ingest i ON i.document_id = d.id
               WHERE d.id > %s
            ORDER BY d.id
               LIMIT %s
            """,
            (after, size + 1),
        )
        rows = db.fetchall()
    more = len(rows) > size
    documents = tuple(
        DocumentRow(
            id=int(row[0]),
            external_id=row[1],
            title=row[2],
            state=row[3],
            link=row[4],
            location=tuple(row[5] or ()),
            piece_count=int(row[6]),
            indexed_at=row[7],
            last_seen_at=row[8],
        )
        for row in rows[:size]
    )
    return DocumentPage(
        documents=documents,
        total_count=int(total),
        next_cursor=str(documents[-1].id) if more and documents else None,
    )


def state_on(conn: psycopg.Connection, identifier: str) -> DocumentState | NotHeld:
    if not identifier.strip():
        raise InventoryError("a document state needs an identifier; there is no document called nothing")
    with conn.cursor() as db:
        db.execute(
            """
              SELECT d.external_id, d.state, d.parsed_uri,
                     r.state, r.reader, r.reader_version, r.page_count,
                     r.byte_size, r.failure_reason, r.read_at
                FROM documents d
           LEFT JOIN document_reads r ON r.document_id = d.id
               WHERE d.external_id = %s OR d.id::text = %s
            """,
            (identifier, identifier),
        )
        row = db.fetchone()
    if row is None:
        return NotHeld(identifier=identifier)
    return DocumentState(
        external_id=row[0],
        state=row[1],
        parsed_uri=row[2],
        read_state=row[3],
        reader=row[4],
        reader_version=row[5],
        page_count=row[6],
        byte_size=row[7],
        failure_reason=row[8],
        read_at=row[9],
    )


def documents(
    config: AppConfig, *, limit: int | None = None, cursor: str | None = None
) -> DocumentPage:
    with connect(config.db.url) as conn:
        return list_on(conn, limit=limit, cursor=cursor)


def document_state(config: AppConfig, identifier: str) -> DocumentState | NotHeld:
    with connect(config.db.url) as conn:
        return state_on(conn, identifier)
