from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.sync_ledger import RunRecord, finish_run, read_run, recent_runs, start_run


class HistoryError(Exception):
    pass


def list_runs(conn: psycopg.Connection, limit: int = 20) -> tuple[RunRecord, ...]:
    return recent_runs(conn, limit=limit)


def locked_run(conn: psycopg.Connection) -> RunRecord | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT run_id FROM sync_run_lock")
        row = cursor.fetchone()
    if row is None:
        return None
    return read_run(conn, int(row[0]))


def _refusal(in_progress: RunRecord | None) -> str:
    if in_progress is None:
        return "a synchronisation is already running; it must finish before another can start"
    return (
        f"a synchronisation is already running (run {in_progress.id}, started "
        f"{in_progress.started_at.isoformat()}); it must finish before another can start"
    )


def start_exclusive_run(conn: psycopg.Connection, *, client: str, kind: str) -> int:
    in_progress = locked_run(conn)
    if in_progress is not None:
        raise HistoryError(_refusal(in_progress))
    try:
        with conn.transaction():
            run_id = start_run(conn, client=client, kind=kind)
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO sync_run_lock (run_id) VALUES (%s)", (run_id,))
            return run_id
    except psycopg.errors.UniqueViolation as exc:
        raise HistoryError(_refusal(locked_run(conn))) from exc


def finish_exclusive_run(conn: psycopg.Connection, run_id: int, **kwargs: object) -> None:
    finish_run(conn, run_id, **kwargs)  # type: ignore[arg-type]
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM sync_run_lock WHERE run_id = %s", (run_id,))


@dataclass(frozen=True, slots=True)
class DocumentHistory:
    first_seen_run_id: int | None
    last_seen_run_id: int | None


def mark_document_seen(conn: psycopg.Connection, *, external_id: str, run_id: int) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
               SET last_seen_run_id = %(run_id)s,
                   first_seen_run_id = COALESCE(first_seen_run_id, %(run_id)s)
             WHERE external_id = %(external_id)s
            """,
            {"run_id": run_id, "external_id": external_id},
        )
        if cursor.rowcount == 0:
            raise HistoryError(f"no document with external id {external_id!r} to record as seen")


def document_history(conn: psycopg.Connection, external_id: str) -> DocumentHistory | None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT first_seen_run_id, last_seen_run_id FROM documents WHERE external_id = %s",
            (external_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return DocumentHistory(first_seen_run_id=row[0], last_seen_run_id=row[1])
