from __future__ import annotations

import psycopg

READ_STATE_READ = "read"
READ_STATE_FAILED = "failed"

READ_STATES: tuple[str, ...] = (READ_STATE_READ, READ_STATE_FAILED)

_STATE_LIST = ", ".join(f"'{state}'" for state in READ_STATES)

STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS document_reads (
        document_id    bigint      PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
        state          text        NOT NULL,
        reader         text        NOT NULL,
        reader_version text        NOT NULL,
        page_count     integer,
        byte_size      bigint,
        failure_reason text,
        read_at        timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT document_reads_state_known CHECK (state IN ({_STATE_LIST})),
        CONSTRAINT document_reads_version_pinned
            CHECK (btrim(reader) <> '' AND btrim(reader_version) <> ''),
        CONSTRAINT document_reads_failure_has_reason
            CHECK (state <> '{READ_STATE_FAILED}'
                   OR (failure_reason IS NOT NULL AND btrim(failure_reason) <> '')),
        CONSTRAINT document_reads_pages_positive CHECK (page_count IS NULL OR page_count >= 0),
        CONSTRAINT document_reads_bytes_positive CHECK (byte_size IS NULL OR byte_size >= 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS document_reads_failed_idx
        ON document_reads (read_at)
     WHERE state = 'failed'
    """,
    "CREATE INDEX IF NOT EXISTS document_reads_version_idx ON document_reads (reader_version)",
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
