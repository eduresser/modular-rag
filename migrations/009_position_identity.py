from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE sync_positions
        ADD COLUMN IF NOT EXISTS issued_by text,
        ADD COLUMN IF NOT EXISTS scope     text
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
