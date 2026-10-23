from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

import psycopg

from migrations import load as load_migration

_migration = load_migration("002_sync_runs")

RUN_STATUS_RUNNING: str = _migration.RUN_STATUS_RUNNING
RUN_STATUS_COMPLETED: str = _migration.RUN_STATUS_COMPLETED
RUN_STATUS_HALTED: str = _migration.RUN_STATUS_HALTED
RUN_STATUS_FAILED: str = _migration.RUN_STATUS_FAILED
RUN_STATUSES: tuple[str, ...] = _migration.RUN_STATUSES

RUN_KIND_SWEEP: str = _migration.RUN_KIND_SWEEP
RUN_KIND_INCREMENTAL: str = _migration.RUN_KIND_INCREMENTAL
RUN_KINDS: tuple[str, ...] = _migration.RUN_KINDS


class LedgerError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RunCounters:
    documents_seen: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_removed: int = 0
    documents_errored: int = 0
    access_lost: int = 0
    #: Disappearances the source explained neither way — left indexed, never
    #: removed, and counted so the gap is visible (AC-019, principle P-004).
    documents_unproven: int = 0
    documents_retired: int = 0


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: int
    client: str
    kind: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    counters: RunCounters
    sweep_completed: bool
    feed_position: str | None
    window_start: datetime | None
    window_end: datetime | None
    detail: str | None

    @property
    def finished(self) -> bool:
        return self.status != RUN_STATUS_RUNNING

    @property
    def may_remove(self) -> bool:
        return self.kind == RUN_KIND_SWEEP and self.status == RUN_STATUS_COMPLETED and self.sweep_completed


_COLUMNS = """
    id, client, kind, status, started_at, finished_at,
    documents_seen, documents_added, documents_updated, documents_skipped,
    documents_removed, documents_errored, access_lost, documents_unproven,
    documents_retired,
    sweep_completed, feed_position, window_start, window_end, detail
"""


def start_run(conn: psycopg.Connection, *, client: str, kind: str) -> int:
    if kind not in RUN_KINDS:
        raise LedgerError(f"run kind {kind!r} is not one of {', '.join(RUN_KINDS)}")
    if not client.strip():
        raise LedgerError("a sync run has to name the client whose deployment produced it")
    window_start = next_window_start(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO sync_runs (client, kind, window_start) VALUES (%s, %s, %s) RETURNING id",
            (client, kind, window_start),
        )
        (run_id,) = cursor.fetchone()
    return int(run_id)


def require_baseline_sweep(conn: psycopg.Connection) -> None:
    if last_completed_sweep(conn) is None:
        raise LedgerError(
            "a delta run cannot start before a baseline sweep has completed; "
            "run a full sweep first (AC-021)"
        )


def start_delta_run(conn: psycopg.Connection, *, client: str) -> int:
    require_baseline_sweep(conn)
    return start_run(conn, client=client, kind=RUN_KIND_INCREMENTAL)


def record_observed(conn: psycopg.Connection, run_id: int, external_ids: Iterable[str]) -> int:
    identifiers = sorted({identifier for identifier in external_ids if identifier})
    if not identifiers:
        return 0
    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO sync_observations (run_id, external_id) VALUES (%s, %s)
            ON CONFLICT (run_id, external_id) DO NOTHING
            """,
            [(run_id, identifier) for identifier in identifiers],
        )
    return len(identifiers)


def observed(conn: psycopg.Connection, run_id: int) -> frozenset[str]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT external_id FROM sync_observations WHERE run_id = %s", (run_id,))
        return frozenset(row[0] for row in cursor.fetchall())


def commit_page(
    conn: psycopg.Connection,
    *,
    run_id: int,
    external_ids: Sequence[str],
    next_position: str,
    issued_by: str | None = None,
    scope: str | None = None,
) -> None:
    if not next_position:
        raise LedgerError(
            "a page cannot be committed without the position that follows it; without one "
            "the next run would start over from the beginning (AC-011)"
        )
    with conn.transaction():
        record_observed(conn, run_id, external_ids)
        advance_position(
            conn, position=next_position, run_id=run_id, issued_by=issued_by, scope=scope
        )


def advance_position(
    conn: psycopg.Connection,
    *,
    position: str,
    run_id: int | None = None,
    issued_by: str | None = None,
    scope: str | None = None,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO sync_positions (only_row, position, run_id, issued_by, scope)
                 VALUES (true, %(position)s, %(run_id)s, %(issued_by)s, %(scope)s)
            ON CONFLICT (only_row) DO UPDATE
                    SET position = EXCLUDED.position,
                        run_id = EXCLUDED.run_id,
                        issued_by = COALESCE(EXCLUDED.issued_by, sync_positions.issued_by),
                        scope = COALESCE(EXCLUDED.scope, sync_positions.scope),
                        updated_at = now()
            """,
            {"position": position, "run_id": run_id, "issued_by": issued_by, "scope": scope},
        )


def current_position(conn: psycopg.Connection) -> str | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT position FROM sync_positions WHERE only_row")
        row = cursor.fetchone()
    return row[0] if row is not None else None


@dataclass(frozen=True, slots=True)
class StoredPosition:
    position: str
    issued_by: str | None
    scope: str | None


def stored_position(conn: psycopg.Connection) -> StoredPosition | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT position, issued_by, scope FROM sync_positions WHERE only_row")
        row = cursor.fetchone()
    if row is None:
        return None
    return StoredPosition(position=row[0], issued_by=row[1], scope=row[2])


def finish_run(
    conn: psycopg.Connection,
    run_id: int,
    *,
    status: str,
    counters: RunCounters | None = None,
    sweep_completed: bool = False,
    detail: str | None = None,
    window_end: datetime | None = None,
) -> None:
    if status not in RUN_STATUSES or status == RUN_STATUS_RUNNING:
        raise LedgerError(
            f"a run cannot finish with status {status!r}; finishing statuses are "
            f"{', '.join(s for s in RUN_STATUSES if s != RUN_STATUS_RUNNING)}"
        )
    if sweep_completed and status != RUN_STATUS_COMPLETED:
        raise LedgerError(
            f"a run finishing as {status!r} cannot claim its sweep completed; a sweep that "
            "did not finish never observed the missing documents as absent (principle P-003)"
        )
    if window_end is not None and status != RUN_STATUS_COMPLETED:
        raise LedgerError(
            f"a run finishing as {status!r} cannot record a window end; only a completed "
            "run's coverage is trusted as the edge the next run resumes from (AC-022)"
        )
    tallies = counters or RunCounters()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE sync_runs
               SET status = %(status)s,
                   finished_at = now(),
                   documents_seen = %(seen)s,
                   documents_added = %(added)s,
                   documents_updated = %(updated)s,
                   documents_skipped = %(skipped)s,
                   documents_removed = %(removed)s,
                   documents_errored = %(errored)s,
                   access_lost = %(access_lost)s,
                   documents_unproven = %(unproven)s,
                   documents_retired = %(retired)s,
                   sweep_completed = %(sweep_completed)s,
                   feed_position = (SELECT position FROM sync_positions WHERE only_row),
                   window_end = %(window_end)s,
                   detail = %(detail)s
             WHERE id = %(run_id)s
            """,
            {
                "status": status,
                "seen": tallies.documents_seen,
                "added": tallies.documents_added,
                "updated": tallies.documents_updated,
                "skipped": tallies.documents_skipped,
                "removed": tallies.documents_removed,
                "errored": tallies.documents_errored,
                "access_lost": tallies.access_lost,
                "unproven": tallies.documents_unproven,
                "retired": tallies.documents_retired,
                "sweep_completed": sweep_completed,
                "window_end": window_end,
                "detail": detail,
                "run_id": run_id,
            },
        )
        if cursor.rowcount == 0:
            raise LedgerError(f"no sync run with id {run_id}")


def read_run(conn: psycopg.Connection, run_id: int) -> RunRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {_COLUMNS} FROM sync_runs WHERE id = %s", (run_id,))
        row = cursor.fetchone()
    return _record(row) if row is not None else None


def last_completed_sweep(conn: psycopg.Connection) -> RunRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_COLUMNS}
              FROM sync_runs
             WHERE kind = %s AND status = %s AND sweep_completed
             ORDER BY started_at DESC
             LIMIT 1
            """,
            (RUN_KIND_SWEEP, RUN_STATUS_COMPLETED),
        )
        row = cursor.fetchone()
    return _record(row) if row is not None else None


def next_window_start(conn: psycopg.Connection) -> datetime | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT window_end FROM sync_runs
             WHERE status = %s AND window_end IS NOT NULL
             ORDER BY window_end DESC
             LIMIT 1
            """,
            (RUN_STATUS_COMPLETED,),
        )
        row = cursor.fetchone()
    return row[0] if row is not None else None


def recent_runs(conn: psycopg.Connection, limit: int = 20) -> tuple[RunRecord, ...]:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {_COLUMNS} FROM sync_runs ORDER BY started_at DESC, id DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
    return tuple(_record(row) for row in rows)


def _record(row: tuple) -> RunRecord:
    return RunRecord(
        id=int(row[0]),
        client=row[1],
        kind=row[2],
        status=row[3],
        started_at=row[4],
        finished_at=row[5],
        counters=RunCounters(
            documents_seen=int(row[6]),
            documents_added=int(row[7]),
            documents_updated=int(row[8]),
            documents_skipped=int(row[9]),
            documents_removed=int(row[10]),
            documents_errored=int(row[11]),
            access_lost=int(row[12]),
            documents_unproven=int(row[13]),
            documents_retired=int(row[14]),
        ),
        sweep_completed=bool(row[15]),
        feed_position=row[16],
        window_start=row[17],
        window_end=row[18],
        detail=row[19],
    )


def as_completing_sweep(conn: psycopg.Connection, run_id: int) -> RunRecord:
    record = read_run(conn, run_id)
    if record is None:
        raise LedgerError(f"no sync run with id {run_id}")
    if record.kind != RUN_KIND_SWEEP:
        raise LedgerError(
            f"run {run_id} is a {record.kind}, and only a sweep ever observes absence "
            "(principle P-003)"
        )
    if record.status != RUN_STATUS_RUNNING:
        raise LedgerError(
            f"run {run_id} already finished as {record.status!r}; this is for the sweep that "
            "is still open and has just finished enumerating"
        )
    return replace(record, status=RUN_STATUS_COMPLETED, sweep_completed=True)
