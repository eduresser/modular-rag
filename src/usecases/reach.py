from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.analytics.corpus import never_returned, refused_for_want_of_locator
from src.analytics.latency import DEFAULT_PERCENTILE, largest_step, latency_by_index_effort
from src.config import AppConfig
from src.db import connect

MAX_LISTED = 100


class ReachError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class ReachUnavailable:
    error: str = "The corpus reach report is unavailable; try again later."


@dataclass(frozen=True, slots=True)
class Reach:

    since: datetime
    until: datetime
    never_returned: tuple
    uncitable: tuple
    listed_limit: int

    def payload(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            # Indexed, paid for, and never shown to anybody in this window.
            # Not necessarily a defect: it may be about something nobody asked.
            "never_returned": {
                "count": len(self.never_returned),
                "truncated": len(self.never_returned) >= self.listed_limit,
                "documents": [
                    {
                        "id": str(document.document_id),
                        "document": document.external_id,
                        "title": document.title or document.external_id,
                        "cost": str(document.cost),
                        "currency": document.currency,
                        "operations": document.operations,
                        "first_seen_at": document.first_seen_at.isoformat(),
                        "complete": document.complete,
                    }
                    for document in self.never_returned
                ],
            },
            "uncitable": {
                "count": len(self.uncitable),
                "documents": [
                    {
                        "id": str(document.document_id),
                        "document": document.external_id,
                        "title": document.title or document.external_id,
                        "failure_reason": document.failure_reason,
                        "read_at": document.read_at.isoformat(),
                    }
                    for document in self.uncitable
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class Latency:

    since: datetime
    until: datetime
    percentile: float
    groups: tuple
    step: object | None

    def payload(self) -> dict[str, object]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "percentile": self.percentile,
            "by_index_search_effort": [
                {
                    "index_search_effort": group.index_search_effort,
                    "searches": group.searches,
                    "median_ms": str(group.median_ms),
                    "high_percentile_ms": str(group.high_percentile_ms),
                    "slowest_ms": str(group.slowest_ms),
                }
                for group in self.groups
            ],
            "largest_step": None
            if self.step is None
            else {
                "below_effort": self.step.below.index_search_effort,
                "above_effort": self.step.above.index_search_effort,
                "factor": str(self.step.factor),
                "effort_floor": self.step.effort_floor,
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
        raise ReachError(
            f"a window has to start before it ends, got since={resolved_since.isoformat()} "
            f"and until={resolved_until.isoformat()}"
        )
    if now - resolved_since > retention:
        raise ReachError(
            f"a window reaching back past this deployment's raw retention of "
            f"{config.usage.raw_retention_days} days (usage.raw_retention_days) cannot be "
            "answered at this resolution"
        )
    return resolved_since, resolved_until


def corpus_reach(
    config: AppConfig,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
) -> Reach | ReachUnavailable:
    start, end = window(config, since=since, until=until)
    listed = MAX_LISTED if limit is None else limit
    if listed < 1 or listed > MAX_LISTED:
        raise ReachError(
            f"a report of {listed} documents is outside the 1..{MAX_LISTED} this route names; "
            "the count above the list is what answers 'how many', so the list itself stays bounded"
        )
    try:
        with connect(config.db.url) as conn:
            unused = never_returned(conn, since=start, until=end, limit=listed)
            uncitable = refused_for_want_of_locator(conn)
    except Exception:
        return ReachUnavailable()
    return Reach(
        since=start,
        until=end,
        never_returned=unused,
        uncitable=uncitable[:listed],
        listed_limit=listed,
    )


def search_latency(
    config: AppConfig,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    percentile: float = DEFAULT_PERCENTILE,
) -> Latency | ReachUnavailable:
    start, end = window(config, since=since, until=until)
    if not 0 < percentile < 1:
        raise ReachError(f"a percentile is a fraction between 0 and 1, got {percentile}")
    try:
        with connect(config.db.url) as conn:
            groups = latency_by_index_effort(conn, since=start, until=end, percentile=percentile)
    except Exception:
        return ReachUnavailable()
    return Latency(
        since=start,
        until=end,
        percentile=percentile,
        groups=groups,
        step=largest_step(groups),
    )
