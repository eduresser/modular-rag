from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from src.db import halfvec_literal

DOCUMENT_STATE_INDEXED = "indexed"
DOCUMENT_STATE_GONE = "gone"


@dataclass(frozen=True, slots=True)
class ModelStamp:
    model: str
    dimension: int


@dataclass(frozen=True, slots=True)
class Piece:
    ordinal: int
    content: str
    embedding: Sequence[float]
    page_start: int | None = None
    page_end: int | None = None
    heading_path: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StoreError(Exception):
    pass


def upsert_document(
    conn: psycopg.Connection,
    *,
    external_id: str,
    title: str | None = None,
    parsed_uri: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    media_type: str | None = None,
    link: str | None = None,
    location: Sequence[str] | None = None,
    modified_at: datetime | None = None,
    size_bytes: int | None = None,
) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                       external_id, title, parsed_uri, metadata,
                       media_type, link, location, modified_at, size_bytes)
                 VALUES (%(external_id)s,
                         COALESCE(%(title)s, ''),
                         %(parsed_uri)s,
                         COALESCE(%(metadata)s, '{}'::jsonb),
                         %(media_type)s,
                         %(link)s,
                         COALESCE(%(location)s, '{}'::text[]),
                         %(modified_at)s,
                         %(size_bytes)s)
            ON CONFLICT (external_id) DO UPDATE
                    SET title       = COALESCE(%(title)s, documents.title),
                        parsed_uri  = COALESCE(%(parsed_uri)s, documents.parsed_uri),
                        metadata    = COALESCE(%(metadata)s, documents.metadata),
                        media_type  = COALESCE(%(media_type)s, documents.media_type),
                        link        = COALESCE(%(link)s, documents.link),
                        location    = COALESCE(%(location)s, documents.location),
                        modified_at = COALESCE(%(modified_at)s, documents.modified_at),
                        size_bytes  = COALESCE(%(size_bytes)s, documents.size_bytes),
                        updated_at  = now()
              RETURNING id
            """,
            {
                "external_id": external_id,
                "title": title,
                "parsed_uri": parsed_uri,
                "metadata": Jsonb(dict(metadata)) if metadata is not None else None,
                "media_type": media_type,
                "link": link,
                "location": list(location) if location is not None else None,
                "modified_at": modified_at,
                "size_bytes": size_bytes,
            },
        )
        (document_id,) = cursor.fetchone()
    return int(document_id)


def upsert_pieces(
    conn: psycopg.Connection,
    *,
    document_id: int,
    stamp: ModelStamp,
    pieces: Sequence[Piece],
) -> int:
    _reject_malformed(stamp, pieces)
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM chunks
                      WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s
                """,
                (document_id, stamp.model, stamp.dimension),
            )
            for piece in pieces:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id, ordinal, content, page_start, page_end, heading_path,
                        embed_model, embed_dimension, embedding, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::halfvec, %s)
                    """,
                    (
                        document_id,
                        piece.ordinal,
                        piece.content,
                        piece.page_start,
                        piece.page_end,
                        list(piece.heading_path),
                        stamp.model,
                        stamp.dimension,
                        halfvec_literal(piece.embedding),
                        Jsonb(dict(piece.metadata)),
                    ),
                )
    return len(pieces)


def _reject_malformed(stamp: ModelStamp, pieces: Sequence[Piece]) -> None:
    seen: set[int] = set()
    for piece in pieces:
        if len(piece.embedding) != stamp.dimension:
            raise StoreError(
                f"piece {piece.ordinal} carries a {len(piece.embedding)}-dimension vector "
                f"but is stamped for {stamp.model} at {stamp.dimension} dimensions"
            )
        if piece.ordinal in seen:
            raise StoreError(f"two pieces claim position {piece.ordinal} in the same document")
        if piece.page_start is None and piece.page_end is None and not piece.heading_path:
            raise StoreError(
                f"piece {piece.ordinal} has no page range and no heading path, so nothing that "
                "returns it could be cited (principle P-002)"
            )
        seen.add(piece.ordinal)


def mark_document_indexed(conn: psycopg.Connection, *, external_id: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
               SET state = %s, updated_at = now()
             WHERE external_id = %s
         RETURNING id
            """,
            (DOCUMENT_STATE_INDEXED, external_id),
        )
        return cursor.fetchone() is not None


def mark_document_gone(conn: psycopg.Connection, *, external_id: str) -> bool:
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                   SET state = %s, updated_at = now()
                 WHERE external_id = %s
             RETURNING id
                """,
                (DOCUMENT_STATE_GONE, external_id),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            cursor.execute("DELETE FROM chunks WHERE document_id = %s", (row[0],))
    return True
