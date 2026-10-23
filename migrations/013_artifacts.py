from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS parsed_artifacts (
        uri         text        PRIMARY KEY,
        payload     bytea       NOT NULL,
        media_type  text        NOT NULL,
        byte_size   bigint      NOT NULL,
        written_at  timestamptz NOT NULL DEFAULT now()
    )
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)
