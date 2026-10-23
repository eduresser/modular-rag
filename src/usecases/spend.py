from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from src.analytics.cost import (
    CostFigure,
    CostSlice,
    cost_by_operation,
    cost_by_provider,
    cost_by_reason,
    total_cost,
)
from src.config import AppConfig
from src.db import connect

Grouping = Literal["operation", "provider", "reason"]
GROUPINGS: dict[str, object] = {
    "operation": cost_by_operation,
    "provider": cost_by_provider,
    "reason": cost_by_reason,
}


class SpendError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class SpendUnavailable:
    error: str = "The cost history is unavailable; try again later."


@dataclass(frozen=True, slots=True)
class Spend:

    since: datetime
    until: datetime
    by: str
    slices: tuple[CostSlice, ...]
    total: CostFigure

    def payload(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "by": self.by,
            "slices": [
                {
                    "key": line.label,
                    "events": line.events,
                    "units": str(line.units),
                    "cost": str(line.cost),
                    "currency": line.currency,
                    "failures": line.failures,
                    "unpriced_units": str(line.unpriced_units),
                    "unpriced_events": line.unpriced_events,
                    "complete": line.complete,
                }
                for line in self.slices
            ],
            "total": {
                "cost": str(self.total.cost),
                "currency": self.total.currency,
                "units": str(self.total.units),
                "unpriced_units": str(self.total.unpriced_units),
                "unpriced_events": self.total.unpriced_events,
                "complete": self.total.complete,
            },
        }


def window(
    config: AppConfig, *, since: datetime | None, until: datetime | None
) -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    retention = timedelta(days=config.usage.raw_retention_days)
    resolved_until = until or now
    resolved_since = since or (resolved_until - retention)
    if resolved_since >= resolved_until:
        raise SpendError(
            f"a window has to start before it ends, got since={resolved_since.isoformat()} "
            f"and until={resolved_until.isoformat()}"
        )
    if now - resolved_since > retention:
        raise SpendError(
            f"a window reaching back past this deployment's raw retention of "
            f"{config.usage.raw_retention_days} days (usage.raw_retention_days) cannot be "
            "answered at this resolution; the rows behind it are rolled up into daily "
            "totals, and answering from those without saying so would change what the "
            "figure means"
        )
    return resolved_since, resolved_until


def grouping(by: str) -> Grouping:
    if by not in GROUPINGS:
        known = ", ".join(sorted(GROUPINGS))
        raise SpendError(f"cost cannot be grouped by {by!r}; it can be grouped by {known}")
    return by  # type: ignore[return-value]


def spend(
    config: AppConfig,
    *,
    by: str = "operation",
    since: datetime | None = None,
    until: datetime | None = None,
) -> Spend | SpendUnavailable:
    key = grouping(by)
    start, end = window(config, since=since, until=until)
    try:
        with connect(config.db.url) as conn:
            slices = GROUPINGS[key](conn, since=start, until=end)  # type: ignore[operator]
            total = total_cost(conn, since=start, until=end)
    except Exception:
        return SpendUnavailable()
    return Spend(since=start, until=end, by=key, slices=tuple(slices), total=total)
