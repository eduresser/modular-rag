from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE sync_runs
        ADD COLUMN IF NOT EXISTS documents_retired integer NOT NULL DEFAULT 0
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)
