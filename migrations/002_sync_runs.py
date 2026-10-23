from __future__ import annotations

import psycopg

RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_HALTED = "halted"
RUN_STATUS_FAILED = "failed"

# Closed set: only `completed` lets reconciliation delete anything (P-003)
RUN_STATUSES: tuple[str, ...] = (
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_HALTED,
    RUN_STATUS_FAILED,
)

RUN_KIND_SWEEP = "sweep"
RUN_KIND_INCREMENTAL = "incremental"

RUN_KINDS: tuple[str, ...] = (RUN_KIND_SWEEP, RUN_KIND_INCREMENTAL)

_STATUS_LIST = ", ".join(f"'{status}'" for status in RUN_STATUSES)
_KIND_LIST = ", ".join(f"'{kind}'" for kind in RUN_KINDS)

STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS sync_runs (
        id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        client            text        NOT NULL,
        kind              text        NOT NULL,
        status            text        NOT NULL DEFAULT '{RUN_STATUS_RUNNING}',
        started_at        timestamptz NOT NULL DEFAULT now(),
        finished_at       timestamptz,
        documents_seen    integer     NOT NULL DEFAULT 0,
        documents_added   integer     NOT NULL DEFAULT 0,
        documents_updated integer     NOT NULL DEFAULT 0,
        documents_skipped integer     NOT NULL DEFAULT 0,
        documents_removed integer     NOT NULL DEFAULT 0,
        documents_errored integer     NOT NULL DEFAULT 0,
        access_lost       integer     NOT NULL DEFAULT 0,
        documents_unproven integer    NOT NULL DEFAULT 0,
        sweep_completed   boolean     NOT NULL DEFAULT false,
        feed_position     text,
        detail            text,
        CONSTRAINT sync_runs_status_known CHECK (status IN ({_STATUS_LIST})),
        CONSTRAINT sync_runs_kind_known   CHECK (kind IN ({_KIND_LIST})),
        CONSTRAINT sync_runs_client_named CHECK (btrim(client) <> ''),
        CONSTRAINT sync_runs_finish_matches_status CHECK (
            (status = '{RUN_STATUS_RUNNING}' AND finished_at IS NULL)
            OR (status <> '{RUN_STATUS_RUNNING}' AND finished_at IS NOT NULL)
        ),
        CONSTRAINT sync_runs_sweep_completion_is_final CHECK (
            NOT sweep_completed OR status <> '{RUN_STATUS_RUNNING}'
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS sync_runs_started_idx ON sync_runs (started_at DESC)",
    "CREATE INDEX IF NOT EXISTS sync_runs_status_idx ON sync_runs (status, started_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS sync_positions (
        only_row   boolean     PRIMARY KEY DEFAULT true,
        position   text        NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now(),
        run_id     bigint      REFERENCES sync_runs (id) ON DELETE SET NULL,
        CONSTRAINT sync_positions_is_a_single_row CHECK (only_row)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_observations (
        run_id      bigint NOT NULL REFERENCES sync_runs (id) ON DELETE CASCADE,
        external_id text   NOT NULL,
        PRIMARY KEY (run_id, external_id)
    )
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
