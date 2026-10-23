from __future__ import annotations

import psycopg

from src.search_events import SEARCH_OUTCOMES

_OUTCOME_LIST = ", ".join(f"'{outcome}'" for outcome in SEARCH_OUTCOMES)

STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS search_events (
        id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        client              text        NOT NULL,
        caller              text        NOT NULL,
        question_digest     text        NOT NULL,
        question_text       text,
        dense_weight        numeric     NOT NULL,
        lexical_weight      numeric     NOT NULL,
        candidate_depth     integer     NOT NULL,
        result_count        integer     NOT NULL,
        index_search_effort integer     NOT NULL,
        document_ids        bigint[]    NOT NULL DEFAULT '{{}}',
        chunk_ids           bigint[]    NOT NULL DEFAULT '{{}}',
        rerank_model        text,
        duration_ms         numeric     NOT NULL,
        outcome             text        NOT NULL,
        occurred_at         timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT search_events_client_named   CHECK (btrim(client) <> ''),
        CONSTRAINT search_events_caller_named   CHECK (btrim(caller) <> ''),
        CONSTRAINT search_events_digest_present CHECK (btrim(question_digest) <> ''),
        CONSTRAINT search_events_outcome_known  CHECK (outcome IN ({_OUTCOME_LIST})),
        CONSTRAINT search_events_duration_positive CHECK (duration_ms >= 0),
        CONSTRAINT search_events_effort_positive   CHECK (index_search_effort >= 1),
        CONSTRAINT search_events_depth_covers_count CHECK (candidate_depth >= result_count)
    )
    """,
    "CREATE INDEX IF NOT EXISTS search_events_occurred_idx ON search_events (occurred_at)",
    "CREATE INDEX IF NOT EXISTS search_events_digest_idx ON search_events (question_digest)",
    "CREATE INDEX IF NOT EXISTS search_events_effort_idx ON search_events (index_search_effort)",
    "CREATE INDEX IF NOT EXISTS search_events_documents_idx ON search_events USING gin (document_ids)",
    """
    CREATE TABLE IF NOT EXISTS search_daily (
        day                 date        NOT NULL,
        client              text        NOT NULL,
        outcome             text        NOT NULL,
        searches            bigint      NOT NULL,
        documents_returned  bigint      NOT NULL,
        duration_p50_ms     numeric     NOT NULL,
        duration_p95_ms     numeric     NOT NULL,
        rolled_up_at        timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (day, client, outcome)
    )
    """,
    "CREATE INDEX IF NOT EXISTS search_daily_day_idx ON search_daily (day)",
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
