from __future__ import annotations

import logging
import time
from dataclasses import replace
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg

from src.ingest import IngestRun, Pipeline, ingest_documents
from src.ingest_idempotency import (
    APP_VERSION,
    CLAIM_STATE_CLAIMED,
    FAILURE_PERMANENT,
    Claim,
    fail,
    in_flight,
    settle,
)
from src.reprocess import ACTION_TOUCH
from src.schedule import next_fire
from src.sources.base import (
    ActivityEvent,
    DocumentUnavailable,
    SourceDocument,
    group_by_document,
)
from src.sync_decision import decide_next_run
from src.sync_queue import (
    active_request,
    attach_run,
    execution_lock,
    finish_request,
    mark_running,
)
from src.reprocess import Retirement, retire_foreign_pieces
from src.sync_reconcile import Reconciliation, reconcile
from src.sync_ledger import (
    RUN_KIND_INCREMENTAL,
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_HALTED,
    LedgerError,
    RunCounters,
    commit_page,
    current_position,
    finish_run,
    observed,
    read_run,
    RunRecord,
    as_completing_sweep,
    record_observed,
    start_run,
)
from src.usage import REASON_NEW_DOCUMENT

logger = logging.getLogger(__name__)

PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class WorkKey:
    external_id: str
    content_digest: str
    fingerprint: str

    def __str__(self) -> str:
        return f"{self.external_id}:{self.content_digest}:{self.fingerprint}"


def work_key(*, external_id: str, content_digest: str, fingerprint: str) -> str:
    return str(WorkKey(external_id, content_digest, fingerprint))


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    run_id: int
    runs: tuple[IngestRun, ...]
    counters: RunCounters
    completed: bool
    halted: bool = False

    @property
    def errored(self) -> int:
        return self.counters.documents_errored

    @property
    def indexed(self) -> int:
        return self.counters.documents_added + self.counters.documents_updated


def sweep(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    *,
    page_size: int = PAGE_SIZE,
    app_version: str = APP_VERSION,
    should_stop: Callable[[], bool] = lambda: False,
    on_started: Callable[[int], None] = lambda run_id: None,
) -> SweepOutcome:
    run_id = start_run(conn, client=pipeline.client, kind=RUN_KIND_SWEEP)
    on_started(run_id)
    pipeline.checkpoint()
    position = pipeline.source.start_position()
    runs: list[IngestRun] = []
    counters = RunCounters()
    try:
        for page in _pages(pipeline.source.sweep(), page_size):
            if should_stop():
                finish_run(
                    conn,
                    run_id,
                    status=RUN_STATUS_HALTED,
                    counters=counters,
                    detail="stopped: the process was asked to shut down before the sweep finished",
                )
                pipeline.checkpoint()
                return SweepOutcome(
                    run_id=run_id, runs=tuple(runs), counters=counters, completed=False, halted=True
                )
            result = ingest_documents(
                conn, pipeline, page, run_id=run_id, app_version=app_version
            )
            runs.append(result)
            counters = _accumulate(counters, result)
            issued_by, scope = pipeline.resume.origin if pipeline.resume else (None, None)
            commit_page(
                conn,
                run_id=run_id,
                external_ids=[document.external_id for document in page],
                next_position=position,
                issued_by=issued_by,
                scope=scope,
            )
            pipeline.checkpoint()
    except Exception as error:  # noqa: BLE001 — the enumeration, not one document
        finish_run(
            conn,
            run_id,
            status=RUN_STATUS_FAILED,
            counters=counters,
            detail=f"the enumeration stopped: {error}",
        )
        pipeline.checkpoint()
        return SweepOutcome(run_id=run_id, runs=tuple(runs), counters=counters, completed=False)
    closing = as_completing_sweep(conn, run_id)
    outcome = _reconcile_before_closing(
        conn, pipeline, run_id, closing, observed_ids=observed(conn, run_id)
    )
    retirement = _retire_before_closing(conn, pipeline, run_id, closing)
    finish_run(
        conn,
        run_id,
        status=RUN_STATUS_COMPLETED,
        counters=_with_reconciliation(counters, outcome, retirement),
        sweep_completed=True,
    )
    pipeline.checkpoint()
    return SweepOutcome(run_id=run_id, runs=tuple(runs), counters=counters, completed=True)


def _with_reconciliation(
    counters: RunCounters, outcome: Reconciliation, retirement: Retirement | None = None
) -> RunCounters:
    return replace(
        counters,
        documents_removed=len(outcome.removed),
        documents_unproven=len(outcome.unproven),
        access_lost=len(outcome.vetoed),
        documents_retired=len(retirement.retired) if retirement else 0,
    )


def _reconcile_before_closing(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    run_id: int,
    record: RunRecord,
    *,
    observed_ids: Iterable[str] | None = None,
    reported_removals: Sequence[str] = (),
) -> Reconciliation:
    grouped: Mapping[str, Sequence[ActivityEvent]] = {}
    if pipeline.activity is not None:
        grouped = group_by_document(pipeline.activity(record.window_start))
    outcome = reconcile(
        conn,
        pipeline.source,
        run=record,
        reported_removals=reported_removals,
        observed=observed_ids,
        activity=grouped,
    )
    if outcome.removed or outcome.unproven or outcome.vetoed:
        logger.info("reconciliation for run %d: %s", run_id, outcome.detail)
    return outcome


def _retire_before_closing(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    run_id: int,
    record: RunRecord,
) -> Retirement:
    outcome = retire_foreign_pieces(
        conn, module=pipeline.config.process.type, may_remove=record.may_remove
    )
    if outcome.retired or outcome.spared or outcome.refusal:
        level = logger.warning if outcome.refusal else logger.info
        level("module retirement for run %d: %s", run_id, outcome.detail)
    return outcome


def resume(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    *,
    app_version: str = APP_VERSION,
    run_id: int | None = None,
) -> IngestRun:
    claims = in_flight(conn, app_version=app_version, run_id=run_id)
    if not claims:
        return IngestRun(stamp=pipeline.stamp)
    documents = tuple(_probe_all(pipeline, claims))
    run = (
        ingest_documents(conn, pipeline, documents, run_id=run_id, app_version=app_version)
        if documents
        else IngestRun(stamp=pipeline.stamp)
    )
    _settle_resumed(conn, claims, run)
    return run


def _settle_resumed(
    conn: psycopg.Connection, claims: Sequence[Claim], run: IngestRun
) -> None:
    outcomes = {result.external_id: result for result in run.results}
    for claim in claims:
        if claim.state != CLAIM_STATE_CLAIMED:
            continue
        result = outcomes.get(claim.external_id)
        if result is None:
            continue
        if result.failure is not None:
            fail(
                conn,
                claim.id,
                reason=result.failure.reason,
                kind=result.failure.kind,
                document_id=result.document_id or claim.document_id,
            )
        else:
            settle(
                conn,
                claim.id,
                action=result.action,
                document_id=result.document_id or claim.document_id,
            )


def unfinished(
    conn: psycopg.Connection,
    *,
    app_version: str = APP_VERSION,
    run_id: int | None = None,
) -> tuple[Claim, ...]:
    return in_flight(conn, app_version=app_version, run_id=run_id)


def incremental(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    *,
    position: str | None = None,
    app_version: str = APP_VERSION,
    should_stop: Callable[[], bool] = lambda: False,
) -> SweepOutcome:
    run_id = start_run(conn, client=pipeline.client, kind=RUN_KIND_INCREMENTAL)
    runs: list[IngestRun] = []
    counters = RunCounters()
    cursor = position if position is not None else current_position(conn)
    removals: list[str] = []
    try:
        while True:
            if should_stop():
                finish_run(
                    conn,
                    run_id,
                    status=RUN_STATUS_HALTED,
                    counters=counters,
                    detail="stopped: the process was asked to shut down before the change feed was exhausted",
                )
                pipeline.checkpoint()
                return SweepOutcome(
                    run_id=run_id, runs=tuple(runs), counters=counters, completed=False, halted=True
                )
            page = pipeline.source.changes_since(cursor)
            if page.documents:
                result = ingest_documents(
                    conn, pipeline, page.documents, run_id=run_id, app_version=app_version
                )
                runs.append(result)
                counters = _accumulate(counters, result)
            removals.extend(page.removed)
            record_observed(
                conn, run_id, (document.external_id for document in page.documents)
            )
            commit_page(
                conn,
                run_id=run_id,
                external_ids=[document.external_id for document in page.documents],
                next_position=page.next_position,
            )
            pipeline.checkpoint()
            cursor = page.next_position
            if page.exhausted:
                break
    except Exception as error:  # noqa: BLE001 — the feed, not one document
        finish_run(
            conn,
            run_id,
            status=RUN_STATUS_FAILED,
            counters=counters,
            detail=f"the change feed stopped: {error}",
        )
        pipeline.checkpoint()
        return SweepOutcome(run_id=run_id, runs=tuple(runs), counters=counters, completed=False)
    record = read_run(conn, run_id)
    outcome = _reconcile_before_closing(
        conn, pipeline, run_id, record, reported_removals=removals
    )
    # An incremental run is never a completed sweep, so this reports what it
    # found and removes nothing — see `retire_foreign_pieces`.
    retirement = _retire_before_closing(conn, pipeline, run_id, record)
    finish_run(
        conn,
        run_id,
        status=RUN_STATUS_COMPLETED,
        counters=_with_reconciliation(counters, outcome, retirement),
    )
    pipeline.checkpoint()
    return SweepOutcome(run_id=run_id, runs=tuple(runs), counters=counters, completed=True)


SCHEDULE_FULL_SWEEP = "full_sweep"
SCHEDULE_INCREMENTAL = "incremental"

POLL_INTERVAL_SECONDS = 5.0


def run_schedule(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    *,
    should_stop: Callable[[], bool],
    app_version: str = APP_VERSION,
    page_size: int = PAGE_SIZE,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleeper: Callable[[float], None] = time.sleep,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    with execution_lock(conn) as acquired:
        if not acquired:
            logger.info("another worker owns synchronisation; leaving work with it")
            return
        if active_request(conn) is None:
            resume(conn, pipeline, app_version=app_version)
        pipeline.checkpoint()

    expressions = {
        SCHEDULE_FULL_SWEEP: pipeline.config.schedule.full_sweep,
        SCHEDULE_INCREMENTAL: pipeline.config.schedule.incremental,
    }
    active = {name: expression for name, expression in expressions.items() if expression}
    fires = {name: next_fire(expression, now=clock()) for name, expression in active.items()}
    _report_schedules(active, fires)

    while not should_stop():
        if run_requested_sync(
            conn, pipeline, app_version=app_version, page_size=page_size,
            should_stop=should_stop,
        ):
            now = clock()
            for name in active:
                if fires[name] <= now:
                    logger.info("skipping the scheduled %r run: requested sync covered its time", name)
                    fires[name] = next_fire(active[name], now=now)
            continue
        now = clock()
        due = sorted((name for name in active if fires[name] <= now), key=fires.get)
        for name in due:
            _fire(
                conn,
                pipeline,
                name,
                app_version=app_version,
                page_size=page_size,
                should_stop=should_stop,
            )
            fires[name] = next_fire(active[name], now=clock())
        if not due:
            sleeper(poll_interval)


def run_requested_sync(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    *,
    app_version: str = APP_VERSION,
    page_size: int = PAGE_SIZE,
    should_stop: Callable[[], bool] = lambda: False,
) -> bool:
    if active_request(conn) is None:
        return False
    with execution_lock(conn) as acquired:
        if not acquired:
            return False
        request = active_request(conn)
        if request is None:
            return False
        record = read_run(conn, request.run_id) if request.run_id is not None else None
        if record is not None and record.status in (RUN_STATUS_COMPLETED, RUN_STATUS_FAILED):
            finish_request(conn, request.id, record.status)
            pipeline.checkpoint()
            return True
        mark_running(conn, request.id)
        pipeline.checkpoint()
        logger.info("starting requested sync: sync_id=%s", request.id)
        run_id = request.run_id

        def started(identifier: int) -> None:
            nonlocal run_id
            run_id = identifier
            attach_run(conn, request.id, identifier)

        try:
            if record is not None:
                resume(conn, pipeline, app_version=app_version, run_id=record.id)
                if not record.finished:
                    finish_run(
                        conn, record.id, status=RUN_STATUS_HALTED, counters=record.counters,
                        detail="interrupted attempt; the durable request is being resumed",
                    )
                pipeline.checkpoint()
            outcome = sweep(
                conn, pipeline, app_version=app_version, page_size=page_size,
                should_stop=should_stop, on_started=started,
            )
            if outcome.halted:
                conn.execute("UPDATE sync_requests SET status = 'queued' WHERE id = %s", (request.id,))
            else:
                finish_request(
                    conn, request.id,
                    RUN_STATUS_COMPLETED if outcome.completed else RUN_STATUS_FAILED,
                )
        except Exception as exc:
            logger.error(
                "requested sync failed: sync_id=%s run_id=%s error_type=%s",
                request.id, run_id, type(exc).__name__,
            )
            if conn.closed or conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
                raise
            current = read_run(conn, run_id) if run_id is not None else None
            if current is not None and not current.finished:
                finish_run(conn, run_id, status=RUN_STATUS_FAILED, detail="requested sync failed; check worker logs")
            finish_request(conn, request.id, RUN_STATUS_FAILED)
        pipeline.checkpoint()
        logger.info("requested sync settled: sync_id=%s run_id=%s", request.id, run_id)
        return True


def _report_schedules(
    active: "dict[str, str]", fires: "dict[str, datetime]"
) -> None:
    if not active:
        logger.info(
            "no schedule is configured, so this worker runs nothing on its own; "
            "set schedule.full_sweep or schedule.incremental to change that"
        )
        return
    for name in sorted(active):
        logger.info(
            "schedule %r is %r, next run at %s",
            name,
            active[name],
            fires[name].isoformat(),
        )


def _fire(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    name: str,
    *,
    app_version: str,
    page_size: int,
    should_stop: Callable[[], bool],
) -> SweepOutcome | None:
    with execution_lock(conn) as acquired:
        if not acquired or active_request(conn) is not None:
            logger.info("skipping the scheduled %r run: another synchronisation is queued or running", name)
            return None
        return _fire_owned(
            conn, pipeline, name, app_version=app_version,
            page_size=page_size, should_stop=should_stop,
        )


def _fire_owned(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    name: str,
    *,
    app_version: str,
    page_size: int,
    should_stop: Callable[[], bool],
) -> SweepOutcome | None:
    waiting = in_flight(conn, app_version=app_version)
    if waiting:
        logger.info(
            "skipping the scheduled %r run: %d claim(s) for %s are still in flight, "
            "so a previous run has not settled yet",
            name,
            len(waiting),
            app_version,
        )
        return None
    if name == SCHEDULE_FULL_SWEEP:
        return sweep(conn, pipeline, page_size=page_size, app_version=app_version, should_stop=should_stop)
    if name == SCHEDULE_INCREMENTAL:
        if pipeline.resume is not None:
            decision = decide_next_run(conn, pipeline.resume)
            if not decision.is_delta:
                logger.info(
                    "the scheduled %r run is sweeping instead: %s (%s)",
                    name,
                    decision.detail,
                    decision.reason,
                )
                return sweep(
                    conn, pipeline, page_size=page_size, app_version=app_version,
                    should_stop=should_stop,
                )
        return incremental(conn, pipeline, app_version=app_version, should_stop=should_stop)
    raise ValueError(f"no such schedule: {name!r}")


def _pages(documents: Iterable[SourceDocument], size: int) -> Iterator[Sequence[SourceDocument]]:
    if size < 1:
        raise ValueError(f"a page has to hold at least one document, got {size}")
    page: list[SourceDocument] = []
    for document in documents:
        page.append(document)
        if len(page) == size:
            yield tuple(page)
            page = []
    if page:
        yield tuple(page)


def _probe_all(pipeline: Pipeline, claims: Sequence[Claim]) -> Iterator[SourceDocument]:
    for claim in claims:
        try:
            yield pipeline.source.probe(claim.external_id)
        except DocumentUnavailable:
            continue


def _accumulate(counters: RunCounters, result: IngestRun) -> RunCounters:
    added = sum(1 for item in result.indexed if item.reason == REASON_NEW_DOCUMENT)
    return RunCounters(
        documents_seen=counters.documents_seen + len(result.results),
        documents_added=counters.documents_added + added,
        documents_updated=counters.documents_updated + len(result.indexed) - added,
        documents_skipped=(
            counters.documents_skipped
            + sum(1 for item in result.results if item.action == ACTION_TOUCH and item.indexed)
            + len(result.duplicates)
        ),
        documents_removed=counters.documents_removed,
        documents_errored=counters.documents_errored + result.errored,
        access_lost=counters.access_lost,
    )
