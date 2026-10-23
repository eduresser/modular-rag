from __future__ import annotations

import psycopg

from src.store import ModelStamp


class NoActiveVersion(Exception):
    pass


def active_version(conn: psycopg.Connection) -> ModelStamp:
    with conn.cursor() as cursor:
        cursor.execute("SELECT embed_model, embed_dimension FROM active_model_version WHERE only_row")
        row = cursor.fetchone()
    if row is None:
        raise NoActiveVersion(
            "the archive has no active embedding model; activate one before searching, "
            "otherwise a query would be answered from an unknown mixture of model generations"
        )
    return ModelStamp(model=row[0], dimension=int(row[1]))


def current_version(conn: psycopg.Connection) -> ModelStamp | None:
    try:
        return active_version(conn)
    except NoActiveVersion:
        return None


def activate_version(conn: psycopg.Connection, stamp: ModelStamp) -> ModelStamp | None:
    previous = current_version(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO active_model_version (only_row, embed_model, embed_dimension)
                 VALUES (true, %s, %s)
            ON CONFLICT (only_row) DO UPDATE
                    SET embed_model     = EXCLUDED.embed_model,
                        embed_dimension = EXCLUDED.embed_dimension,
                        activated_at    = now()
            """,
            (stamp.model, stamp.dimension),
        )
    return previous


def discard_other_generations(conn: psycopg.Connection, keep: ModelStamp) -> dict[ModelStamp, int]:
    held = pieces_under(conn, keep)
    if held == 0:
        raise ValueError(
            f"nothing is stored under {keep.model!r} at {keep.dimension}, so discarding every "
            "other generation would leave an archive with no pieces at all"
        )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM chunks
                  WHERE NOT (embed_model = %s AND embed_dimension = %s)
              RETURNING embed_model, embed_dimension
            """,
            (keep.model, keep.dimension),
        )
        gone: dict[ModelStamp, int] = {}
        for model, dimension in cursor.fetchall():
            stamp = ModelStamp(model=model, dimension=int(dimension))
            gone[stamp] = gone.get(stamp, 0) + 1
    return gone


def pieces_under(conn: psycopg.Connection, stamp: ModelStamp) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM chunks WHERE embed_model = %s AND embed_dimension = %s",
            (stamp.model, stamp.dimension),
        )
        (count,) = cursor.fetchone()
    return int(count)
