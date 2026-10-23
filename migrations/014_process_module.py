from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE document_ingest
        ADD COLUMN IF NOT EXISTS process_module text
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)
