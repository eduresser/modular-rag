from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import psycopg

from src.prices import DEFAULT_CURRENCY
from src.processing.artifact import READ_STATE_FAILED
from src.processing.to_chunks import NO_LOCATOR_REASON
from src.store import DOCUMENT_STATE_INDEXED


@dataclass(frozen=True, slots=True)
class UnusedDocument:

    document_id: int
    external_id: str
    title: str
    cost: Decimal
    currency: str
    operations: int
    first_seen_at: datetime
    unpriced_units: Decimal = Decimal("0")

    @property
    def complete(self) -> bool:
        return self.unpriced_units == 0


def never_returned(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
    currency: str = DEFAULT_CURRENCY,
    limit: int | None = None,
) -> tuple[UnusedDocument, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.id,
                   d.external_id,
                   d.title,
                   coalesce(sum(u.cost), 0) AS cost,
                   count(u.id)              AS operations,
                   d.first_seen_at,
                   coalesce(sum(u.units) FILTER (WHERE u.unit_price IS NULL), 0) AS unpriced_units
              FROM documents d
              LEFT JOIN usage_events u
                     ON u.document_id = d.id
                    AND u.currency = %(currency)s
             WHERE d.state = %(indexed)s
               AND NOT EXISTS (
                       SELECT 1
                         FROM search_events s
                        WHERE s.occurred_at >= %(since)s
                          AND s.occurred_at <  %(until)s
                          AND d.id = ANY (s.document_ids)
                   )
             GROUP BY d.id, d.external_id, d.title, d.first_seen_at
             ORDER BY cost DESC, d.id
             LIMIT %(limit)s
            """,
            {
                "since": since,
                "until": until,
                "currency": currency,
                "indexed": DOCUMENT_STATE_INDEXED,
                "limit": limit,
            },
        )
        rows = cursor.fetchall()
    return tuple(
        UnusedDocument(
            document_id=int(row[0]),
            external_id=row[1],
            title=row[2],
            cost=row[3],
            currency=currency,
            operations=int(row[4]),
            first_seen_at=row[5],
            unpriced_units=row[6],
        )
        for row in rows
    )


def returned_document_ids(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
) -> tuple[int, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT unnest(document_ids) AS document_id
              FROM search_events
             WHERE occurred_at >= %s AND occurred_at < %s
             ORDER BY document_id
            """,
            (since, until),
        )
        rows = cursor.fetchall()
    return tuple(int(row[0]) for row in rows)


@dataclass(frozen=True, slots=True)
class RefusedForLocator:

    document_id: int
    external_id: str
    title: str
    failure_reason: str
    read_at: datetime


def refused_for_want_of_locator(conn: psycopg.Connection) -> tuple[RefusedForLocator, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.id, d.external_id, d.title, r.failure_reason, r.read_at
              FROM document_reads r
              JOIN documents d ON d.id = r.document_id
             WHERE r.state = %(failed)s
               AND r.failure_reason LIKE %(marker)s
             ORDER BY r.read_at DESC, d.id
            """,
            {"failed": READ_STATE_FAILED, "marker": f"%{NO_LOCATOR_REASON}%"},
        )
        rows = cursor.fetchall()
    return tuple(
        RefusedForLocator(
            document_id=int(row[0]),
            external_id=row[1],
            title=row[2],
            failure_reason=row[3],
            read_at=row[4],
        )
        for row in rows
    )
