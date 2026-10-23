from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import psycopg

from src.sync_ledger import RUN_KIND_INCREMENTAL, RUN_KIND_SWEEP, last_completed_sweep, stored_position

REASON_NO_BASELINE = "no_baseline_sweep"
REASON_NO_POSITION = "no_position_recorded"
REASON_POSITION_HOLDS = "position_still_describes_this_corpus"


@dataclass(frozen=True, slots=True)
class Decision:
    kind: str
    reason: str
    detail: str

    @property
    def is_delta(self) -> bool:
        return self.kind == RUN_KIND_INCREMENTAL


def sweep_because(reason: str, detail: str) -> Decision:
    return Decision(kind=RUN_KIND_SWEEP, reason=reason, detail=detail)


class Resume(Protocol):
    @property
    def origin(self) -> tuple[str, str]:
        ...

    def objection(self, position: str, issued_by: str, scope: str) -> Decision | None:
        ...


def decide_next_run(conn: psycopg.Connection, resume: Resume) -> Decision:
    if last_completed_sweep(conn) is None:
        return sweep_because(
            REASON_NO_BASELINE,
            "no sweep has completed, so there is no enumeration for a delta to follow (AC-021)",
        )
    stored = stored_position(conn)
    if stored is None:
        return sweep_because(
            REASON_NO_POSITION,
            "a baseline exists but no feed position was recorded, so a delta has no mark to "
            "resume from",
        )
    if not stored.issued_by or not stored.scope:
        return sweep_because(
            REASON_NO_POSITION,
            "the stored position does not say which credential took it or which corpus it "
            "describes, so nothing about it can be checked; the sweep records both",
        )
    objection = resume.objection(stored.position, stored.issued_by, stored.scope)
    if objection is not None:
        return objection
    return Decision(
        kind=RUN_KIND_INCREMENTAL,
        reason=REASON_POSITION_HOLDS,
        detail=f"position {stored.position!r} still describes this corpus, and the source "
        "raised no objection to resuming from it",
    )
