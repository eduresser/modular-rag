from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    ALTER TABLE sync_runs
        ADD COLUMN IF NOT EXISTS window_start timestamptz,
        ADD COLUMN IF NOT EXISTS window_end   timestamptz
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = 'sync_runs_window_end_only_on_success'
        ) THEN
            ALTER TABLE sync_runs
                ADD CONSTRAINT sync_runs_window_end_only_on_success CHECK (
                    window_end IS NULL OR status = 'completed'
                );
        END IF;
    END $$;
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
