from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import psycopg

from src.usage import OUTCOME_FAILED

MEDIAN = 0.5
HIGH_PERCENTILE = 0.95

_DAY = "(occurred_at AT TIME ZONE 'UTC')::date"


@dataclass(frozen=True, slots=True)
class UsageDay:

    day: date
    client: str
    operation: str
    reason: str
    provider: str
    currency: str
    events: int
    units: Decimal
    cost: Decimal
    errors: int
    duration_p50_ms: Decimal
    duration_p95_ms: Decimal
    unpriced_events: int = 0
    unpriced_units: Decimal = Decimal("0")

    @property
    def complete(self) -> bool:
        return self.unpriced_events == 0


@dataclass(frozen=True, slots=True)
class SearchDay:

    day: date
    client: str
    outcome: str
    searches: int
    documents_returned: int
    duration_p50_ms: Decimal
    duration_p95_ms: Decimal


@dataclass(frozen=True, slots=True)
class RollupOutcome:

    days_rolled_up: tuple[date, ...]
    usage_aggregates: int
    search_aggregates: int
    usage_rows_pruned: int
    search_rows_pruned: int
    oldest_retained_day: date

    @property
    def pruned_anything(self) -> bool:
        return bool(self.usage_rows_pruned or self.search_rows_pruned)


def days_with_raw_rows(conn: psycopg.Connection) -> tuple[date, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_DAY} AS day FROM usage_events
             UNION
            SELECT {_DAY} AS day FROM search_events
             ORDER BY day
            """
        )
        rows = cursor.fetchall()
    return tuple(row[0] for row in rows)


def roll_up_usage(conn: psycopg.Connection, days: tuple[date, ...]) -> int:
    if not days:
        return 0
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO usage_daily (
                day, client, operation, reason, provider, currency,
                events, units, cost, errors, unpriced_events, unpriced_units,
                duration_p50_ms, duration_p95_ms, rolled_up_at
            )
            SELECT {_DAY},
                   client,
                   operation,
                   reason,
                   provider,
                   currency,
                   count(*),
                   coalesce(sum(units), 0),
                   coalesce(sum(cost), 0),
                   count(*) FILTER (WHERE outcome = %(failed)s),
                   count(*) FILTER (WHERE unit_price IS NULL),
                   coalesce(sum(units) FILTER (WHERE unit_price IS NULL), 0),
                   (percentile_cont(%(median)s)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric,
                   (percentile_cont(%(high)s)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric,
                   now()
              FROM usage_events
             WHERE {_DAY} = ANY (%(days)s)
             GROUP BY {_DAY}, client, operation, reason, provider, currency
                ON CONFLICT (day, client, operation, reason, provider, currency) DO UPDATE
               SET events          = EXCLUDED.events,
                   units           = EXCLUDED.units,
                   cost            = EXCLUDED.cost,
                   errors          = EXCLUDED.errors,
                   unpriced_events = EXCLUDED.unpriced_events,
                   unpriced_units  = EXCLUDED.unpriced_units,
                   duration_p50_ms = EXCLUDED.duration_p50_ms,
                   duration_p95_ms = EXCLUDED.duration_p95_ms,
                   rolled_up_at    = EXCLUDED.rolled_up_at
            """,
            {
                "days": list(days),
                "failed": OUTCOME_FAILED,
                "median": MEDIAN,
                "high": HIGH_PERCENTILE,
            },
        )
        written = cursor.rowcount
    return int(written)


def roll_up_searches(conn: psycopg.Connection, days: tuple[date, ...]) -> int:
    if not days:
        return 0
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO search_daily (
                day, client, outcome, searches, documents_returned,
                duration_p50_ms, duration_p95_ms, rolled_up_at
            )
            SELECT {_DAY},
                   client,
                   outcome,
                   count(*),
                   coalesce(sum(cardinality(document_ids)), 0),
                   (percentile_cont(%(median)s)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric,
                   (percentile_cont(%(high)s)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric,
                   now()
              FROM search_events
             WHERE {_DAY} = ANY (%(days)s)
             GROUP BY {_DAY}, client, outcome
                ON CONFLICT (day, client, outcome) DO UPDATE
               SET searches           = EXCLUDED.searches,
                   documents_returned = EXCLUDED.documents_returned,
                   duration_p50_ms    = EXCLUDED.duration_p50_ms,
                   duration_p95_ms    = EXCLUDED.duration_p95_ms,
                   rolled_up_at       = EXCLUDED.rolled_up_at
            """,
            {"days": list(days), "median": MEDIAN, "high": HIGH_PERCENTILE},
        )
        written = cursor.rowcount
    return int(written)


def prune_raw(conn: psycopg.Connection, oldest_retained_day: date) -> tuple[int, int]:
    with conn.cursor() as cursor:
        cursor.execute(f"DELETE FROM usage_events WHERE {_DAY} < %s", (oldest_retained_day,))
        usage_pruned = cursor.rowcount
        cursor.execute(f"DELETE FROM search_events WHERE {_DAY} < %s", (oldest_retained_day,))
        search_pruned = cursor.rowcount
    return int(usage_pruned), int(search_pruned)


def run_rollup(
    conn: psycopg.Connection,
    *,
    retention_days: int,
    now: datetime | None = None,
    days: tuple[date, ...] | None = None,
) -> RollupOutcome:
    if retention_days < 1:
        raise ValueError(
            f"raw retention has to be at least one day, got {retention_days}; "
            "a retention of zero would prune the day still being written"
        )
    moment = now or datetime.now(timezone.utc)
    oldest_retained_day = (moment - timedelta(days=retention_days)).astimezone(timezone.utc).date()
    with conn.transaction():
        target = days if days is not None else days_with_raw_rows(conn)
        usage_aggregates = roll_up_usage(conn, target)
        search_aggregates = roll_up_searches(conn, target)
        usage_pruned, search_pruned = prune_raw(conn, oldest_retained_day)
    return RollupOutcome(
        days_rolled_up=tuple(target),
        usage_aggregates=usage_aggregates,
        search_aggregates=search_aggregates,
        usage_rows_pruned=usage_pruned,
        search_rows_pruned=search_pruned,
        oldest_retained_day=oldest_retained_day,
    )


def usage_aggregate(conn: psycopg.Connection, day: date) -> tuple[UsageDay, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT day, client, operation, reason, provider, currency, events, units, cost,
                   errors, duration_p50_ms, duration_p95_ms, unpriced_events, unpriced_units
              FROM usage_daily
             WHERE day = %s
             ORDER BY cost DESC, operation, reason
            """,
            (day,),
        )
        rows = cursor.fetchall()
    return tuple(
        UsageDay(
            day=row[0],
            client=row[1],
            operation=row[2],
            reason=row[3],
            provider=row[4],
            currency=row[5],
            events=int(row[6]),
            units=row[7],
            cost=row[8],
            errors=int(row[9]),
            duration_p50_ms=row[10],
            duration_p95_ms=row[11],
            unpriced_events=int(row[12]),
            unpriced_units=row[13],
        )
        for row in rows
    )


def search_aggregate(conn: psycopg.Connection, day: date) -> tuple[SearchDay, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT day, client, outcome, searches, documents_returned,
                   duration_p50_ms, duration_p95_ms
              FROM search_daily
             WHERE day = %s
             ORDER BY searches DESC, outcome
            """,
            (day,),
        )
        rows = cursor.fetchall()
    return tuple(
        SearchDay(
            day=row[0],
            client=row[1],
            outcome=row[2],
            searches=int(row[3]),
            documents_returned=int(row[4]),
            duration_p50_ms=row[5],
            duration_p95_ms=row[6],
        )
        for row in rows
    )
