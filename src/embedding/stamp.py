from __future__ import annotations

from collections.abc import Sequence

import psycopg

from src.config import AppConfig
from src.models import Chunk
from src.store import ModelStamp, Piece


class StampError(Exception):
    ...


def stamp_from_config(config: AppConfig) -> ModelStamp:
    return ModelStamp(model=config.embed.model, dimension=config.embed.dimension)


def attach(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], stamp: ModelStamp) -> list[Piece]:
    if len(chunks) != len(vectors):
        raise StampError(
            f"{len(chunks)} piece(s) and {len(vectors)} vector(s) cannot be paired; "
            "pairing them by position would give every piece after the gap the "
            "meaning of its neighbour, and no row would look wrong"
        )
    for position, vector in enumerate(vectors):
        if len(vector) != stamp.dimension:
            raise StampError(
                f"vector {position} is {len(vector)} wide and the stamp says "
                f"{stamp.model} at {stamp.dimension}"
            )
    return [chunk.to_piece(vector) for chunk, vector in zip(chunks, vectors)]


def stamps_present(conn: psycopg.Connection) -> tuple[ModelStamp, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT embed_model, embed_dimension, count(*)
              FROM chunks
             GROUP BY embed_model, embed_dimension
             ORDER BY count(*) DESC, embed_model
            """
        )
        rows = cursor.fetchall()
    return tuple(ModelStamp(model=row[0], dimension=int(row[1])) for row in rows)


def stamp_of(conn: psycopg.Connection, chunk_id: int) -> ModelStamp | None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT embed_model, embed_dimension FROM chunks WHERE id = %s", (chunk_id,)
        )
        row = cursor.fetchone()
    return None if row is None else ModelStamp(model=row[0], dimension=int(row[1]))


def documents_missing(conn: psycopg.Connection, stamp: ModelStamp) -> tuple[int, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT document_id
              FROM chunks
             WHERE document_id NOT IN (
                       SELECT document_id
                         FROM chunks
                        WHERE embed_model = %s AND embed_dimension = %s
                   )
             ORDER BY document_id
            """,
            (stamp.model, stamp.dimension),
        )
        rows = cursor.fetchall()
    return tuple(int(row[0]) for row in rows)
