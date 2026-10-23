from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID, uuid4

import psycopg

from src.sync_ledger import read_run

logger = logging.getLogger(__name__)

EXECUTION_LOCK = 0x70617265746F


@dataclass(frozen=True)
class SyncRequest:
    id: UUID
    status: str
    run_id: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class SyncBusy:
    active_sync_id: UUID | None = None


def active_request(conn: psycopg.Connection) -> SyncRequest | None:
    row = conn.execute(
        "SELECT id, status, run_id, created_at, started_at, finished_at "
        "FROM sync_requests WHERE status IN ('queued', 'running')"
    ).fetchone()
    return SyncRequest(*row) if row else None


def read_request(conn: psycopg.Connection, sync_id: UUID) -> SyncRequest | None:
    row = conn.execute(
        "SELECT id, status, run_id, created_at, started_at, finished_at "
        "FROM sync_requests WHERE id = %s", (sync_id,),
    ).fetchone()
    return SyncRequest(*row) if row else None


def enqueue(conn: psycopg.Connection) -> SyncRequest | SyncBusy:
    acquired = conn.execute(
        "SELECT pg_try_advisory_xact_lock(%s)", (EXECUTION_LOCK,)
    ).fetchone()[0]
    active = active_request(conn)
    if active is not None or not acquired:
        return SyncBusy(active.id if active else None)
    # The unique index is a second guard, even for a writer not taking the
    # advisory lock. A conflict is an ordinary busy result, never a 500.
    row = conn.execute(
        "INSERT INTO sync_requests (id) VALUES (%s) ON CONFLICT DO NOTHING "
        "RETURNING id, status, run_id, created_at, started_at, finished_at",
        (uuid4(),),
    ).fetchone()
    if row is None:
        active = active_request(conn)
        return SyncBusy(active.id if active else None)
    return SyncRequest(*row)


@contextmanager
def execution_lock(conn: psycopg.Connection):
    acquired = conn.execute(
        "SELECT pg_try_advisory_lock(%s)", (EXECUTION_LOCK,)
    ).fetchone()[0]
    try:
        yield acquired
    finally:
        if acquired:
            try:
                conn.execute("SELECT pg_advisory_unlock(%s)", (EXECUTION_LOCK,))
            except psycopg.Error as exc:
                logger.error("sync ownership release failed: error_type=%s", type(exc).__name__)
                _release_after_failure(conn)


def _release_after_failure(conn: psycopg.Connection) -> None:
    try:
        conn.rollback()
        conn.execute("SELECT pg_advisory_unlock(%s)", (EXECUTION_LOCK,))
    except psycopg.Error as exc:
        logger.error("sync ownership rollback-and-release failed: error_type=%s", type(exc).__name__)
        conn.close()


def mark_running(conn: psycopg.Connection, sync_id: UUID) -> None:
    conn.execute(
        "UPDATE sync_requests SET status = 'running', "
        "started_at = COALESCE(started_at, clock_timestamp()) WHERE id = %s",
        (sync_id,),
    )


def attach_run(conn: psycopg.Connection, sync_id: UUID, run_id: int) -> None:
    conn.execute("UPDATE sync_requests SET run_id = %s WHERE id = %s", (run_id, sync_id))


def finish_request(conn: psycopg.Connection, sync_id: UUID, status: str) -> None:
    conn.execute(
        "UPDATE sync_requests SET status = %s, finished_at = clock_timestamp() "
        "WHERE id = %s AND status IN ('queued', 'running')",
        (status, sync_id),
    )


def request_payload(conn: psycopg.Connection, request: SyncRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "sync_id": str(request.id),
        "status": request.status,
        "created_at": request.created_at.isoformat(),
    }
    if request.started_at is not None:
        payload["started_at"] = request.started_at.isoformat()
    if request.finished_at is not None:
        payload["finished_at"] = request.finished_at.isoformat()
    if request.run_id is not None:
        payload["run_id"] = request.run_id
        record = read_run(conn, request.run_id)
        if record is not None:
            payload["counters"] = asdict(record.counters)
    if request.status == "failed":
        payload["error"] = "Synchronisation failed; check the worker logs."
    return payload
