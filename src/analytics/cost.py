from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import psycopg

from src.prices import DEFAULT_CURRENCY
from src.usage import OUTCOME_FAILED, REASON_WATERMARK_MOVED


@dataclass(frozen=True, slots=True)
class CostFigure:

    cost: Decimal
    currency: str
    units: Decimal
    unpriced_units: Decimal
    unpriced_events: int

    @property
    def complete(self) -> bool:
        return self.unpriced_events == 0


@dataclass(frozen=True, slots=True)
class CostSlice:

    label: str
    currency: str
    events: int
    units: Decimal
    cost: Decimal
    failures: int
    unpriced_events: int = 0
    unpriced_units: Decimal = Decimal("0")

    @property
    def complete(self) -> bool:
        return self.unpriced_events == 0


def _slices(conn: psycopg.Connection, column: str, since: datetime, until: datetime) -> tuple[CostSlice, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {column} AS label,
                   currency,
                   count(*)                                     AS events,
                   coalesce(sum(units), 0)                      AS units,
                   coalesce(sum(cost), 0)                       AS cost,
                   count(*) FILTER (WHERE outcome = %(failed)s) AS failures,
                   count(*) FILTER (WHERE unit_price IS NULL)   AS unpriced_events,
                   coalesce(sum(units) FILTER (WHERE unit_price IS NULL), 0) AS unpriced_units
              FROM usage_events
             WHERE occurred_at >= %(since)s AND occurred_at < %(until)s
             GROUP BY label, currency
             ORDER BY cost DESC, label
            """,
            {"since": since, "until": until, "failed": OUTCOME_FAILED},
        )
        rows = cursor.fetchall()
    return tuple(
        CostSlice(
            label=row[0],
            currency=row[1],
            events=int(row[2]),
            units=row[3],
            cost=row[4],
            failures=int(row[5]),
            unpriced_events=int(row[6]),
            unpriced_units=row[7],
        )
        for row in rows
    )


def cost_by_reason(conn: psycopg.Connection, *, since: datetime, until: datetime) -> tuple[CostSlice, ...]:
    return _slices(conn, "reason", since, until)


def cost_by_provider(conn: psycopg.Connection, *, since: datetime, until: datetime) -> tuple[CostSlice, ...]:
    return _slices(conn, "provider", since, until)


def cost_by_operation(conn: psycopg.Connection, *, since: datetime, until: datetime) -> tuple[CostSlice, ...]:
    return _slices(conn, "operation", since, until)


def _figure(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
    currency: str,
    reason: str | None = None,
) -> CostFigure:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT coalesce(sum(cost), 0),
                   coalesce(sum(units), 0),
                   coalesce(sum(units) FILTER (WHERE unit_price IS NULL), 0),
                   count(*) FILTER (WHERE unit_price IS NULL)
              FROM usage_events
             WHERE occurred_at >= %(since)s AND occurred_at < %(until)s
               AND currency = %(currency)s
               -- Cast explicit: with no value on either side of the comparison
               -- PostgreSQL cannot infer the parameter's type at all.
               AND (%(reason)s::text IS NULL OR reason = %(reason)s::text)
            """,
            {"since": since, "until": until, "currency": currency, "reason": reason},
        )
        cost, units, unpriced_units, unpriced_events = cursor.fetchone()
    return CostFigure(
        cost=cost,
        currency=currency,
        units=units,
        unpriced_units=unpriced_units,
        unpriced_events=int(unpriced_events),
    )


def total_cost(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
    currency: str = DEFAULT_CURRENCY,
) -> CostFigure:
    return _figure(conn, since=since, until=until, currency=currency)


def cost_of_unchanged_content(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
    currency: str = DEFAULT_CURRENCY,
) -> CostFigure:
    return _figure(
        conn, since=since, until=until, currency=currency, reason=REASON_WATERMARK_MOVED
    )


def unchanged_content_breakdown(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
) -> tuple[CostSlice, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT operation,
                   currency,
                   count(*)                                     AS events,
                   coalesce(sum(units), 0)                      AS units,
                   coalesce(sum(cost), 0)                       AS cost,
                   count(*) FILTER (WHERE outcome = %(failed)s) AS failures,
                   count(*) FILTER (WHERE unit_price IS NULL)   AS unpriced_events,
                   coalesce(sum(units) FILTER (WHERE unit_price IS NULL), 0) AS unpriced_units
              FROM usage_events
             WHERE occurred_at >= %(since)s AND occurred_at < %(until)s
               AND reason = %(reason)s
             GROUP BY operation, currency
             ORDER BY cost DESC, operation
            """,
            {
                "since": since,
                "until": until,
                "reason": REASON_WATERMARK_MOVED,
                "failed": OUTCOME_FAILED,
            },
        )
        rows = cursor.fetchall()
    return tuple(
        CostSlice(
            label=row[0],
            currency=row[1],
            events=int(row[2]),
            units=row[3],
            cost=row[4],
            failures=int(row[5]),
            unpriced_events=int(row[6]),
            unpriced_units=row[7],
        )
        for row in rows
    )
