from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import psycopg

from src.prices import BILLABLE_OPERATIONS, DEFAULT_CURRENCY, Price, PriceBook

REASON_NEW_DOCUMENT = "new_document"
REASON_CONTENT_CHANGED = "content_changed"
REASON_CONFIGURATION_CHANGED = "configuration_changed"
REASON_WATERMARK_MOVED = "watermark_moved"
REASON_OPERATOR_REQUESTED = "operator_requested"
REASON_QUESTION_ASKED = "question_asked"

USAGE_REASONS: tuple[str, ...] = (
    REASON_NEW_DOCUMENT,
    REASON_CONTENT_CHANGED,
    REASON_CONFIGURATION_CHANGED,
    REASON_WATERMARK_MOVED,
    REASON_OPERATOR_REQUESTED,
    REASON_QUESTION_ASKED,
)

REPROCESSING_REASONS: tuple[str, ...] = (
    REASON_CONTENT_CHANGED,
    REASON_CONFIGURATION_CHANGED,
    REASON_WATERMARK_MOVED,
    REASON_OPERATOR_REQUESTED,
)

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
USAGE_OUTCOMES: tuple[str, ...] = (OUTCOME_SUCCEEDED, OUTCOME_FAILED)


class UsageError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class UsageEvent:
    operation: str
    reason: str
    provider: str
    model: str
    unit: str
    units: Decimal
    duration_ms: Decimal
    unit_price: Decimal | None = None
    outcome: str = OUTCOME_SUCCEEDED
    currency: str = DEFAULT_CURRENCY
    document_id: int | None = None
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.operation not in BILLABLE_OPERATIONS:
            raise UsageError(
                f"operation {self.operation!r} is not billable; the billable operations are "
                f"{', '.join(BILLABLE_OPERATIONS)}"
            )
        if self.reason not in USAGE_REASONS:
            raise UsageError(
                f"reason {self.reason!r} is not one of the recorded reasons "
                f"({', '.join(USAGE_REASONS)}); the reason is the column that makes this "
                "table worth having (principle P-010), so it cannot be improvised"
            )
        if self.outcome not in USAGE_OUTCOMES:
            raise UsageError(
                f"outcome {self.outcome!r} is neither {OUTCOME_SUCCEEDED!r} nor {OUTCOME_FAILED!r}"
            )
        if not self.unit:
            raise UsageError("a usage row has to say what a unit is, or its units mean nothing")
        if self.units < 0:
            raise UsageError(f"units consumed cannot be negative, got {self.units}")
        if self.unit_price is not None and self.unit_price < 0:
            raise UsageError(f"a unit price cannot be negative, got {self.unit_price}")
        if self.duration_ms < 0:
            raise UsageError(f"a duration cannot be negative, got {self.duration_ms}")

    @property
    def priced(self) -> bool:
        return self.unit_price is not None

    @property
    def cost(self) -> Decimal | None:
        if self.unit_price is None:
            return None
        return self.units * self.unit_price


def priced(
    book: PriceBook,
    *,
    operation: str,
    reason: str,
    provider: str,
    model: str,
    unit: str,
    units: Decimal | int | float | str,
    duration_ms: Decimal | int | float | str,
    outcome: str = OUTCOME_SUCCEEDED,
    document_id: int | None = None,
    occurred_at: datetime | None = None,
) -> UsageEvent:
    price: Price | None = book.price_or_none(
        operation=operation, provider=provider, model=model, unit=unit
    )
    return UsageEvent(
        operation=operation,
        reason=reason,
        provider=provider,
        model=model,
        unit=unit,
        units=Decimal(str(units)),
        unit_price=price.amount if price is not None else None,
        currency=price.currency if price is not None else DEFAULT_CURRENCY,
        duration_ms=Decimal(str(duration_ms)),
        outcome=outcome,
        document_id=document_id,
        occurred_at=occurred_at,
    )


@dataclass(frozen=True, slots=True)
class UsageRecord:
    @property
    def priced(self) -> bool:
        return self.unit_price is not None

    id: int
    client: str
    operation: str
    reason: str
    provider: str
    model: str
    unit: str
    units: Decimal
    unit_price: Decimal | None
    currency: str
    cost: Decimal | None
    duration_ms: Decimal
    outcome: str
    document_id: int | None
    occurred_at: datetime


_COLUMNS = """
    id, client, operation, reason, provider, model, unit, units, unit_price,
    currency, cost, duration_ms, outcome, document_id, occurred_at
"""


def record_usage(conn: psycopg.Connection, event: UsageEvent, *, client: str) -> int:
    if not client.strip():
        raise UsageError(
            "a usage row has to name the client whose deployment produced it (AC-168); "
            "it is what lets one view across deployments group by client"
        )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO usage_events (
                client, operation, reason, provider, model, unit, units, unit_price,
                currency, duration_ms, outcome, document_id, occurred_at
            ) VALUES (
                %(client)s, %(operation)s, %(reason)s, %(provider)s, %(model)s, %(unit)s,
                %(units)s, %(unit_price)s, %(currency)s, %(duration_ms)s, %(outcome)s,
                %(document_id)s, COALESCE(%(occurred_at)s, now())
            ) RETURNING id
            """,
            {
                "client": client,
                "operation": event.operation,
                "reason": event.reason,
                "provider": event.provider,
                "model": event.model,
                "unit": event.unit,
                "units": event.units,
                "unit_price": event.unit_price,
                "currency": event.currency,
                "duration_ms": event.duration_ms,
                "outcome": event.outcome,
                "document_id": event.document_id,
                "occurred_at": event.occurred_at,
            },
        )
        (usage_id,) = cursor.fetchone()
    return int(usage_id)


def read_usage(conn: psycopg.Connection, usage_id: int) -> UsageRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {_COLUMNS} FROM usage_events WHERE id = %s", (usage_id,))
        row = cursor.fetchone()
    return _record(row) if row is not None else None


def usage_between(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
) -> tuple[UsageRecord, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_COLUMNS}
              FROM usage_events
             WHERE occurred_at >= %s AND occurred_at < %s
             ORDER BY occurred_at, id
            """,
            (since, until),
        )
        rows = cursor.fetchall()
    return tuple(_record(row) for row in rows)


def _record(row: tuple) -> UsageRecord:
    return UsageRecord(
        id=int(row[0]),
        client=row[1],
        operation=row[2],
        reason=row[3],
        provider=row[4],
        model=row[5],
        unit=row[6],
        units=row[7],
        unit_price=row[8],
        currency=row[9],
        cost=row[10],
        duration_ms=row[11],
        outcome=row[12],
        document_id=int(row[13]) if row[13] is not None else None,
        occurred_at=row[14],
    )
