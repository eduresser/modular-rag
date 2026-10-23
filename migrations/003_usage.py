from __future__ import annotations

import psycopg

from src.prices import BILLABLE_OPERATIONS
from src.usage import USAGE_OUTCOMES, USAGE_REASONS


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS usage_events (
        id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        client      text        NOT NULL,
        operation   text        NOT NULL,
        reason      text        NOT NULL,
        provider    text        NOT NULL,
        model       text        NOT NULL,
        unit        text        NOT NULL,
        units       numeric     NOT NULL,
        unit_price  numeric,
        currency    text        NOT NULL DEFAULT 'USD',
        cost        numeric     GENERATED ALWAYS AS (units * unit_price) STORED,
        duration_ms numeric     NOT NULL,
        outcome     text        NOT NULL,
        document_id bigint      REFERENCES documents (id) ON DELETE SET NULL,
        occurred_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT usage_events_client_named   CHECK (btrim(client) <> ''),
        CONSTRAINT usage_events_operation_known CHECK (operation IN ({_sql_list(BILLABLE_OPERATIONS)})),
        CONSTRAINT usage_events_reason_known    CHECK (reason IN ({_sql_list(USAGE_REASONS)})),
        CONSTRAINT usage_events_outcome_known   CHECK (outcome IN ({_sql_list(USAGE_OUTCOMES)})),
        CONSTRAINT usage_events_units_positive  CHECK (units >= 0),
        CONSTRAINT usage_events_price_positive  CHECK (unit_price >= 0),
        CONSTRAINT usage_events_duration_positive CHECK (duration_ms >= 0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS usage_events_occurred_idx ON usage_events (occurred_at)",
    "CREATE INDEX IF NOT EXISTS usage_events_reason_idx ON usage_events (reason, occurred_at)",
    "CREATE INDEX IF NOT EXISTS usage_events_provider_idx ON usage_events (provider, occurred_at)",
    """
    CREATE INDEX IF NOT EXISTS usage_events_document_idx
        ON usage_events (document_id)
     WHERE document_id IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_daily (
        day             date        NOT NULL,
        client          text        NOT NULL,
        operation       text        NOT NULL,
        reason          text        NOT NULL,
        provider        text        NOT NULL,
        currency        text        NOT NULL,
        events          bigint      NOT NULL,
        units           numeric     NOT NULL,
        cost            numeric     NOT NULL,
        errors          bigint      NOT NULL,
        unpriced_events bigint      NOT NULL,
        unpriced_units  numeric     NOT NULL,
        duration_p50_ms numeric     NOT NULL,
        duration_p95_ms numeric     NOT NULL,
        rolled_up_at    timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (day, client, operation, reason, provider, currency)
    )
    """,
    "CREATE INDEX IF NOT EXISTS usage_daily_day_idx ON usage_daily (day)",
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)  # type: ignore[arg-type]
