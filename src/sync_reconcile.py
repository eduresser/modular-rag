from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import psycopg

from src.sources.base import DocumentUnavailable, Source
from src.sources.base import ActivityEvent, DeletionEvent, MoveEvent, PermissionChangeEvent
from src.store import DOCUMENT_STATE_INDEXED, mark_document_gone
from src.sync_ledger import RunRecord


@dataclass(frozen=True, slots=True)
class Reconciliation:
    removed: tuple[str, ...] = ()
    moved: tuple[str, ...] = ()
    permission_changed: tuple[str, ...] = ()
    unproven: tuple[str, ...] = ()
    still_present: tuple[str, ...] = ()
    carried_out: tuple[tuple[str, str], ...] = ()
    considered: int = 0
    refusal: str | None = None
    detail: str = ""

    @property
    def removed_anything(self) -> bool:
        return bool(self.removed)

    @property
    def vetoed(self) -> tuple[str, ...]:
        carried = tuple(document for document, _ in self.carried_out)
        return tuple(sorted(self.moved + self.permission_changed + carried))


def stored_parents(
    conn: psycopg.Connection, external_ids: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    wanted = sorted({identifier for identifier in external_ids if identifier})
    if not wanted:
        return {}
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT external_id, metadata -> 'parents'
              FROM documents
             WHERE external_id = ANY(%s)
            """,
            (wanted,),
        )
        rows = cursor.fetchall()
    return {
        external_id: tuple(str(parent) for parent in parents)
        for external_id, parents in rows
        if isinstance(parents, list) and parents
    }


def _carried_out_by(
    activity: Mapping[str, Sequence[ActivityEvent]], parents: Sequence[str]
) -> str | None:
    for folder in parents:
        for event in activity.get(folder, ()):
            if isinstance(event, (MoveEvent, PermissionChangeEvent)):
                return folder
    return None


def indexed_count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM documents WHERE state = %s", (DOCUMENT_STATE_INDEXED,)
        )
        (count,) = cursor.fetchone()
    return int(count)


def indexed_external_ids(conn: psycopg.Connection) -> frozenset[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT external_id FROM documents WHERE state = %s", (DOCUMENT_STATE_INDEXED,)
        )
        return frozenset(row[0] for row in cursor.fetchall())


def absent_since(conn: psycopg.Connection, observed: Iterable[str]) -> tuple[str, ...]:
    seen = frozenset(observed)
    return tuple(sorted(indexed_external_ids(conn) - seen))


def reconcile(
    conn: psycopg.Connection,
    source: Source,
    *,
    run: RunRecord,
    reported_removals: Sequence[str] = (),
    observed: Iterable[str] | None = None,
    activity: Mapping[str, Sequence[ActivityEvent]] | None = None,
) -> Reconciliation:
    candidates = {identifier for identifier in reported_removals if identifier}
    if observed is not None:
        if not run.may_remove:
            return Reconciliation(
                considered=0,
                refusal="sweep_did_not_complete",
                detail=(
                    f"run {run.id} is a {run.kind} that finished as {run.status!r} with "
                    f"sweep_completed={run.sweep_completed}, so documents it did not see were "
                    "never observed as absent — they simply never arrived (principle P-003)"
                ),
            )
        candidates.update(absent_since(conn, observed))

    if not candidates:
        return Reconciliation(considered=0, detail="nothing was reported removed and nothing was missing")

    grouped = activity or {}
    ordered = sorted(candidates)

    gone: list[str] = []
    moved: list[str] = []
    permission_changed: list[str] = []
    unproven: list[str] = []
    still_present: list[str] = []
    carried_out: list[tuple[str, str]] = []
    ancestry = stored_parents(conn, ordered)

    for external_id in ordered:
        events = grouped.get(external_id, ())
        if any(isinstance(event, DeletionEvent) for event in events):
            gone.append(external_id)
        elif any(isinstance(event, MoveEvent) for event in events):
            moved.append(external_id)
        elif any(isinstance(event, PermissionChangeEvent) for event in events):
            permission_changed.append(external_id)
        else:
            try:
                source.probe(external_id)
            except DocumentUnavailable as exc:
                if exc.trashed:
                    gone.append(external_id)
                else:
                    carrier = _carried_out_by(grouped, ancestry.get(external_id, ()))
                    if carrier is not None:
                        carried_out.append((external_id, carrier))
                    else:
                        # Not-found is the absence of proof, never proof of
                        # absence (principle P-005): left indexed, unproven.
                        unproven.append(external_id)
            else:
                still_present.append(external_id)

    with conn.transaction():
        removed = [
            external_id for external_id in gone if mark_document_gone(conn, external_id=external_id)
        ]

    return Reconciliation(
        removed=tuple(removed),
        moved=tuple(moved),
        permission_changed=tuple(permission_changed),
        unproven=tuple(unproven),
        still_present=tuple(still_present),
        carried_out=tuple(carried_out),
        considered=len(ordered),
        detail=_report(
            removed=removed,
            moved=moved,
            permission_changed=permission_changed,
            unproven=unproven,
            still_present=still_present,
            carried_out=carried_out,
        ),
    )


def _report(
    *,
    carried_out: Sequence[tuple[str, str]] = (),
    removed: Sequence[str],
    moved: Sequence[str],
    permission_changed: Sequence[str],
    unproven: Sequence[str],
    still_present: Sequence[str],
) -> str:
    parts = [f"{len(removed)} confirmed gone"]
    if moved:
        parts.append(f"{len(moved)} left indexed — moved out of the watched tree, not deleted")
    if permission_changed:
        parts.append(f"{len(permission_changed)} left indexed — a permission change, not a deletion")
    if carried_out:
        folders = sorted({folder for _, folder in carried_out})
        parts.append(
            f"{len(carried_out)} left indexed — carried out by "
            f"{len(folders)} folder(s) above them ({', '.join(folders)}), not deleted"
        )
    if unproven:
        parts.append(f"{len(unproven)} disappearances unproven and left indexed")
    if still_present:
        parts.append(f"{len(still_present)} still present despite being reported removed")
    return "; ".join(parts)
