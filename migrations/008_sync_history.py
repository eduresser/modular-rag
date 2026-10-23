from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS sync_run_lock (
        only_row   boolean PRIMARY KEY DEFAULT true,
        run_id     bigint  NOT NULL REFERENCES sync_runs (id) ON DELETE CASCADE,
        started_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT sync_run_lock_is_a_single_row CHECK (only_row)
    )
    """,
    """
    CREATE OR REPLACE FUNCTION sync_runs_forbid_rewrite() RETURNS trigger AS $$
    BEGIN
        IF OLD.status <> 'running' THEN
            RAISE EXCEPTION
                'sync run % already finished as %; a finished run''s record is never rewritten (AC-023)',
                OLD.id, OLD.status;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS sync_runs_finished_is_final ON sync_runs",
    """
    CREATE TRIGGER sync_runs_finished_is_final
        AFTER UPDATE ON sync_runs
        FOR EACH ROW
        EXECUTE FUNCTION sync_runs_forbid_rewrite()
    """,
    """
    ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS first_seen_run_id bigint REFERENCES sync_runs (id) ON DELETE SET NULL,
        ADD COLUMN IF NOT EXISTS last_seen_run_id  bigint REFERENCES sync_runs (id) ON DELETE SET NULL
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
