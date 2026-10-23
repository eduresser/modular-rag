import psycopg


STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sync_requests (
        id          uuid PRIMARY KEY,
        status      text NOT NULL DEFAULT 'queued',
        run_id      bigint REFERENCES sync_runs (id),
        created_at  timestamptz NOT NULL DEFAULT clock_timestamp(),
        started_at  timestamptz,
        finished_at timestamptz,
        CONSTRAINT sync_requests_status_known CHECK (
            status IN ('queued', 'running', 'completed', 'failed', 'halted')
        ),
        CONSTRAINT sync_requests_finish_matches_status CHECK (
            (status IN ('queued', 'running')) = (finished_at IS NULL)
        )
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS sync_requests_one_active
        ON sync_requests ((true)) WHERE status IN ('queued', 'running')
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)
