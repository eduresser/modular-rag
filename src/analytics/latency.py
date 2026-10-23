from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import psycopg

DEFAULT_PERCENTILE = 0.95


@dataclass(frozen=True, slots=True)
class LatencyGroup:

    index_search_effort: int
    searches: int
    median_ms: Decimal
    high_percentile_ms: Decimal
    slowest_ms: Decimal


def latency_by_index_effort(
    conn: psycopg.Connection,
    *,
    since: datetime,
    until: datetime,
    percentile: float = DEFAULT_PERCENTILE,
) -> tuple[LatencyGroup, ...]:
    if not 0 < percentile < 1:
        raise ValueError(f"a percentile is a fraction between 0 and 1, got {percentile}")
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT index_search_effort,
                   count(*) AS searches,
                   (percentile_cont(0.5)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric AS median_ms,
                   (percentile_cont(%(percentile)s)
                        WITHIN GROUP (ORDER BY duration_ms::double precision))::numeric AS high_ms,
                   max(duration_ms) AS slowest_ms
              FROM search_events
             WHERE occurred_at >= %(since)s AND occurred_at < %(until)s
             GROUP BY index_search_effort
             ORDER BY index_search_effort
            """,
            {"since": since, "until": until, "percentile": percentile},
        )
        rows = cursor.fetchall()
    return tuple(
        LatencyGroup(
            index_search_effort=int(row[0]),
            searches=int(row[1]),
            median_ms=row[2],
            high_percentile_ms=row[3],
            slowest_ms=row[4],
        )
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class LatencyStep:

    below: LatencyGroup
    above: LatencyGroup
    factor: Decimal

    @property
    def effort_floor(self) -> int:
        return self.above.index_search_effort


def largest_step(groups: tuple[LatencyGroup, ...]) -> LatencyStep | None:
    steps: list[LatencyStep] = []
    for below, above in zip(groups, groups[1:]):
        if above.high_percentile_ms <= 0:
            continue
        steps.append(
            LatencyStep(
                below=below,
                above=above,
                factor=below.high_percentile_ms / above.high_percentile_ms,
            )
        )
    if not steps:
        return None
    return max(steps, key=lambda step: step.factor)
