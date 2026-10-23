from __future__ import annotations

import psycopg

CLAIM_STATE_CLAIMED = "claimed"
CLAIM_STATE_DONE = "done"
CLAIM_STATE_FAILED = "failed"

CLAIM_STATES: tuple[str, ...] = (CLAIM_STATE_CLAIMED, CLAIM_STATE_DONE, CLAIM_STATE_FAILED)

ACTION_TOUCH = "touch"
ACTION_REEMBED = "reembed"
ACTION_RECUT = "recut"
ACTION_READ = "read"

ACTIONS: tuple[str, ...] = (ACTION_TOUCH, ACTION_REEMBED, ACTION_RECUT, ACTION_READ)

FAILURE_PERMANENT = "permanent"
FAILURE_TRANSIENT = "transient"

FAILURE_KINDS: tuple[str, ...] = (FAILURE_PERMANENT, FAILURE_TRANSIENT)

_CLAIM_STATE_LIST = ", ".join(f"'{state}'" for state in CLAIM_STATES)
_ACTION_LIST = ", ".join(f"'{action}'" for action in ACTIONS)
_FAILURE_KIND_LIST = ", ".join(f"'{kind}'" for kind in FAILURE_KINDS)

STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS document_ingest (
        document_id       bigint      PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
        content_digest    text        NOT NULL,
        fingerprint       text        NOT NULL,
        read_fingerprint  text        NOT NULL,
        cut_fingerprint   text        NOT NULL,
        embed_fingerprint text        NOT NULL,
        reader_version    text        NOT NULL,
        cut_version       text        NOT NULL,
        revision_marker   text,
        piece_count       integer     NOT NULL DEFAULT 0,
        indexed_at        timestamptz NOT NULL DEFAULT now(),
        last_seen_at      timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT document_ingest_digest_present CHECK (btrim(content_digest) <> ''),
        CONSTRAINT document_ingest_fingerprint_present CHECK (
            btrim(fingerprint) <> ''
            AND btrim(read_fingerprint) <> ''
            AND btrim(cut_fingerprint) <> ''
            AND btrim(embed_fingerprint) <> ''
        ),
        CONSTRAINT document_ingest_versions_present CHECK (
            btrim(reader_version) <> '' AND btrim(cut_version) <> ''
        ),
        CONSTRAINT document_ingest_pieces_positive CHECK (piece_count >= 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS document_ingest_fingerprint_idx
        ON document_ingest (fingerprint)
    """,
    f"""
    CREATE TABLE IF NOT EXISTS ingest_claims (
        id             bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        external_id    text        NOT NULL,
        content_digest text        NOT NULL,
        fingerprint    text        NOT NULL,
        document_id    bigint      REFERENCES documents (id) ON DELETE SET NULL,
        run_id         bigint      REFERENCES sync_runs (id) ON DELETE SET NULL,
        app_version    text        NOT NULL,
        state          text        NOT NULL DEFAULT '{CLAIM_STATE_CLAIMED}',
        action         text,
        deliveries     integer     NOT NULL DEFAULT 1,
        failure_kind   text,
        failure_reason text,
        claimed_at     timestamptz NOT NULL DEFAULT now(),
        settled_at     timestamptz,
        CONSTRAINT ingest_claims_one_per_unit UNIQUE (external_id, content_digest, fingerprint),
        CONSTRAINT ingest_claims_identified CHECK (
            btrim(external_id) <> '' AND btrim(content_digest) <> '' AND btrim(fingerprint) <> ''
        ),
        CONSTRAINT ingest_claims_state_known CHECK (state IN ({_CLAIM_STATE_LIST})),
        CONSTRAINT ingest_claims_action_known CHECK (action IS NULL OR action IN ({_ACTION_LIST})),
        CONSTRAINT ingest_claims_version_pinned CHECK (btrim(app_version) <> ''),
        CONSTRAINT ingest_claims_deliveries_counted CHECK (deliveries >= 1),
        CONSTRAINT ingest_claims_settlement_matches_state CHECK (
            (state = '{CLAIM_STATE_CLAIMED}' AND settled_at IS NULL)
            OR (state <> '{CLAIM_STATE_CLAIMED}' AND settled_at IS NOT NULL)
        ),
        CONSTRAINT ingest_claims_failure_is_actionable CHECK (
            state <> '{CLAIM_STATE_FAILED}'
            OR (failure_reason IS NOT NULL AND btrim(failure_reason) <> ''
                AND failure_kind IN ({_FAILURE_KIND_LIST}))
        ),
        CONSTRAINT ingest_claims_failure_kind_known CHECK (
            failure_kind IS NULL OR failure_kind IN ({_FAILURE_KIND_LIST})
        ),
        CONSTRAINT ingest_claims_done_names_its_action CHECK (
            state <> '{CLAIM_STATE_DONE}' OR action IS NOT NULL
        )
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS ingest_claims_in_flight_idx
        ON ingest_claims (app_version, claimed_at)
     WHERE state = '{CLAIM_STATE_CLAIMED}'
    """,
    "CREATE INDEX IF NOT EXISTS ingest_claims_document_idx ON ingest_claims (external_id)",
    """
    CREATE TABLE IF NOT EXISTS indexing_generations (
        fingerprint   text        PRIMARY KEY,
        settings      jsonb       NOT NULL,
        first_seen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT indexing_generations_fingerprint_present CHECK (btrim(fingerprint) <> '')
    )
    """,
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
