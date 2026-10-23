from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS media_type  text,
        ADD COLUMN IF NOT EXISTS link        text,
        ADD COLUMN IF NOT EXISTS location    text[] NOT NULL DEFAULT '{}',
        ADD COLUMN IF NOT EXISTS modified_at timestamptz,
        ADD COLUMN IF NOT EXISTS size_bytes  bigint
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
