import logging
from dataclasses import asdict, dataclass
from uuid import UUID

from src.config import AppConfig
from src.db import connect
from src.sync_history import list_runs, locked_run
from src.sync_queue import SyncBusy, enqueue, read_request, request_payload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncUnavailable:
    error: str = "Synchronisation is unavailable; try again later."


def request_sync(config: AppConfig) -> dict | SyncBusy | SyncUnavailable:
    try:
        with connect(config.db.url) as conn:
            request = enqueue(conn)
            result = request if isinstance(request, SyncBusy) else request_payload(conn, request)
        return result
    except Exception as exc:
        logger.error("sync admission failed: error_type=%s", type(exc).__name__)
        return SyncUnavailable()


def sync_status(config: AppConfig, sync_id: UUID) -> dict | None | SyncUnavailable:
    try:
        with connect(config.db.url) as conn:
            request = read_request(conn, sync_id)
            return request_payload(conn, request) if request else None
    except Exception as exc:
        logger.error("sync status failed: sync_id=%s error_type=%s", sync_id, type(exc).__name__)
        return SyncUnavailable()


@dataclass(frozen=True)
class SyncHistory:

    current: dict | None
    last: dict | None
    runs: tuple[dict, ...]

    def payload(self) -> dict:
        return {
            "current": self.current,
            "last": self.last,
            "runs": list(self.runs),
        }


def _run_payload(record) -> dict:
    return {
        "run_id": record.id,
        "client": record.client,
        "kind": record.kind,
        "status": record.status,
        "started_at": record.started_at.isoformat(),
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
        "counters": asdict(record.counters),
    }


def sync_history(config: AppConfig, *, limit: int = 20) -> SyncHistory | SyncUnavailable:
    try:
        with connect(config.db.url) as conn:
            running = locked_run(conn)
            records = list_runs(conn, limit=limit)
    except Exception as exc:
        logger.error("sync history failed: error_type=%s", type(exc).__name__)
        return SyncUnavailable()
    current = _run_payload(running) if running is not None else None
    finished = next(
        (record for record in records if record.finished_at is not None),
        None,
    )
    return SyncHistory(
        current=current,
        last=_run_payload(finished) if finished is not None else None,
        runs=tuple(_run_payload(record) for record in records),
    )
