from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import psycopg

from src.sources.base import (
    ActivityEvent,
    DeletionEvent,
    DocumentUnavailable,
    MoveEvent,
    PermissionChangeEvent,
    Source,
)
from src.sync_history import DocumentHistory, document_history, list_runs
from src.sync_ledger import RunRecord
from src.sync_reconcile import absent_since, indexed_external_ids

# The verdicts a real reconciliation could give a candidate — see
# src.sync_reconcile.reconcile, whose per-candidate order this mirrors.
VERDICT_GONE = "gone"
VERDICT_MOVED = "moved"
VERDICT_PERMISSION_CHANGED = "permission_changed"
VERDICT_UNPROVEN = "unproven"
VERDICT_STILL_PRESENT = "still_present"


class ActivityHistoryLike(Protocol):
    def events_since(
        self, moment: datetime, *, until: datetime | None = None
    ) -> Iterable[ActivityEvent]: ...


@dataclass(frozen=True, slots=True)
class MissingDocumentVerdict:
    external_id: str
    verdict: str
    detail: str

    @property
    def would_be_removed(self) -> bool:
        return self.verdict == VERDICT_GONE


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    external_id: str
    indexed: bool
    present_at_source: bool
    trashed: bool
    access_denied: bool
    activity: tuple[ActivityEvent, ...]
    history: DocumentHistory | None


def activity_in_window(
    history: ActivityHistoryLike,
    *,
    since: datetime,
    until: datetime | None = None,
) -> tuple[ActivityEvent, ...]:
    return tuple(history.events_since(since, until=until))


def diagnose_absences(
    conn: psycopg.Connection,
    source: Source,
    *,
    observed: Iterable[str],
    activity: Mapping[str, Sequence[ActivityEvent]] | None = None,
) -> tuple[MissingDocumentVerdict, ...]:
    candidates = absent_since(conn, observed)
    grouped = activity or {}
    results: list[MissingDocumentVerdict] = []
    for external_id in candidates:
        events = grouped.get(external_id, ())
        if any(isinstance(event, DeletionEvent) for event in events):
            results.append(
                MissingDocumentVerdict(
                    external_id=external_id,
                    verdict=VERDICT_GONE,
                    detail="the activity history recorded a deletion event naming it",
                )
            )
        elif any(isinstance(event, MoveEvent) for event in events):
            results.append(
                MissingDocumentVerdict(
                    external_id=external_id,
                    verdict=VERDICT_MOVED,
                    detail="the activity history recorded a move out of the watched tree — not a deletion (AC-013)",
                )
            )
        elif any(isinstance(event, PermissionChangeEvent) for event in events):
            results.append(
                MissingDocumentVerdict(
                    external_id=external_id,
                    verdict=VERDICT_PERMISSION_CHANGED,
                    detail="the activity history recorded a permission change — not a deletion (AC-013)",
                )
            )
        else:
            try:
                source.probe(external_id)
            except DocumentUnavailable as exc:
                if exc.trashed:
                    results.append(
                        MissingDocumentVerdict(
                            external_id=external_id,
                            verdict=VERDICT_GONE,
                            detail=f"the source still answers for it, but marks it trashed ({exc.reason})",
                        )
                    )
                else:
                    results.append(
                        MissingDocumentVerdict(
                            external_id=external_id,
                            verdict=VERDICT_UNPROVEN,
                            detail=(
                                f"the source answered not found ({exc.reason}), with no deletion "
                                "event either; absence is never proof (AC-012), so a real run "
                                "would leave it indexed"
                            ),
                        )
                    )
            else:
                results.append(
                    MissingDocumentVerdict(
                        external_id=external_id,
                        verdict=VERDICT_STILL_PRESENT,
                        detail="the source handed it over — the disappearance was never real",
                    )
                )
    return tuple(results)


def document_evidence(
    conn: psycopg.Connection,
    source: Source,
    external_id: str,
    *,
    activity: Sequence[ActivityEvent] = (),
) -> DocumentEvidence:
    indexed = external_id in indexed_external_ids(conn)
    try:
        source.probe(external_id)
    except DocumentUnavailable as exc:
        present_at_source = False
        trashed = exc.trashed
        access_denied = exc.denied
    else:
        present_at_source = True
        trashed = False
        access_denied = False
    return DocumentEvidence(
        external_id=external_id,
        indexed=indexed,
        present_at_source=present_at_source,
        trashed=trashed,
        access_denied=access_denied,
        activity=tuple(activity),
        history=document_history(conn, external_id),
    )


def stored_runs(conn: psycopg.Connection, limit: int = 20) -> tuple[RunRecord, ...]:
    return list_runs(conn, limit=limit)
