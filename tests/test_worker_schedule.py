
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.ingest_idempotency import APP_VERSION, claim
from src.sync_decision import sweep_because
from src.sync_ledger import (
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    advance_position,
    finish_run,
    start_run,
)
from src.store import DOCUMENT_STATE_GONE, DOCUMENT_STATE_INDEXED
from src.sync_ledger import read_run
from src.worker import SCHEDULE_FULL_SWEEP, SCHEDULE_INCREMENTAL, _fire, incremental, run_schedule, sweep
from tests.conftest import contract_payload, ingest_world, pdf_file

CLAUSES = ("Cláusula primeira: do objeto.", "Cláusula segunda: do prazo.")


def _world(monkeypatch, names=("doc-1",), **kwargs):
    return ingest_world(
        monkeypatch,
        files=[pdf_file(name, content=f"%PDF {name}".encode()) for name in names],
        outputs={name: contract_payload(clauses=CLAUSES) for name in names},
        **kwargs,
    )


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class _Clock:

    def __init__(self, start: datetime) -> None:
        self.current = start
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.current += timedelta(seconds=seconds)


def test_a_deployment_naming_no_schedule_idles_rather_than_busy_looping(monkeypatch, conn):
    world = _world(monkeypatch)
    clock = _Clock(_utc(2026, 9, 10, 3, 0, 0))
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    run_schedule(conn, world.pipeline, should_stop=should_stop, clock=clock.now, sleeper=clock.sleep)

    assert clock.slept
    assert world.reads == 0


def test_resume_takes_up_unfinished_claims_before_any_schedule_is_checked(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))
    claim(
        conn,
        external_id="doc-1",
        content_digest="a-digest-from-before-the-restart",
        fingerprint="a-fingerprint-from-before-the-restart",
        app_version=APP_VERSION,
    )

    run_schedule(conn, world.pipeline, should_stop=lambda: True, clock=lambda: _utc(2026, 9, 10, 3, 0, 0))

    assert world.reads == 1


def test_a_due_full_sweep_schedule_calls_sweep(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",), RAG_SCHEDULE_FULL_SWEEP="* * * * *")
    clock = _Clock(_utc(2026, 9, 10, 3, 0, 0))

    run_schedule(
        conn,
        world.pipeline,
        should_stop=lambda: world.reads >= 1,
        clock=clock.now,
        sleeper=clock.sleep,
        poll_interval=90.0,
    )

    assert world.reads == 1
    with conn.cursor() as cursor:
        cursor.execute("SELECT kind FROM sync_runs")
        assert [row[0] for row in cursor.fetchall()] == ["sweep"]


def test_a_due_incremental_schedule_calls_incremental(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",), RAG_SCHEDULE_INCREMENTAL="* * * * *")
    advance_position(conn, position=world.pipeline.source.start_position())
    world.drive.push_change("doc-1")
    clock = _Clock(_utc(2026, 9, 10, 3, 0, 0))

    run_schedule(
        conn,
        world.pipeline,
        should_stop=lambda: world.reads >= 1,
        clock=clock.now,
        sleeper=clock.sleep,
        poll_interval=90.0,
    )

    assert world.reads == 1
    with conn.cursor() as cursor:
        cursor.execute("SELECT kind FROM sync_runs")
        assert [row[0] for row in cursor.fetchall()] == ["incremental"]


def test_fire_skips_and_logs_when_claims_are_in_flight(monkeypatch, conn, caplog):
    world = _world(monkeypatch, names=("doc-1",))
    claim(conn, external_id="doc-1", content_digest="d", fingerprint="f", app_version=APP_VERSION)

    with caplog.at_level(logging.INFO, logger="src.worker"):
        result = _fire(
            conn,
            world.pipeline,
            SCHEDULE_FULL_SWEEP,
            app_version=APP_VERSION,
            page_size=50,
            should_stop=lambda: False,
        )

    assert result is None
    assert world.reads == 0
    assert any(
        "full_sweep" in record.getMessage() and "in flight" in record.getMessage()
        for record in caplog.records
    )


def test_fire_runs_the_named_schedule_when_nothing_is_in_flight(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))

    outcome = _fire(
        conn,
        world.pipeline,
        SCHEDULE_FULL_SWEEP,
        app_version=APP_VERSION,
        page_size=50,
        should_stop=lambda: False,
    )

    assert outcome.completed is True
    assert world.reads == 1


def test_fire_rejects_a_schedule_name_it_does_not_know(monkeypatch, conn):
    world = _world(monkeypatch)

    with pytest.raises(ValueError):
        _fire(
            conn,
            world.pipeline,
            "nonsense",
            app_version=APP_VERSION,
            page_size=50,
            should_stop=lambda: False,
        )


def test_sweep_halts_between_pages_and_the_run_record_says_so(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    outcome = sweep(conn, world.pipeline, page_size=1, should_stop=should_stop)

    assert outcome.completed is False
    assert outcome.halted is True
    assert world.reads == 1
    with conn.cursor() as cursor:
        cursor.execute("SELECT status, sweep_completed FROM sync_runs WHERE id = %s", (outcome.run_id,))
        assert cursor.fetchone() == ("halted", False)


def test_incremental_halts_before_the_feed_is_exhausted_and_the_run_record_says_so(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))
    world.drive.push_change("doc-1")
    advance_position(conn, position=world.pipeline.source.start_position())

    outcome = incremental(conn, world.pipeline, should_stop=lambda: True)

    assert outcome.completed is False
    assert outcome.halted is True
    assert world.reads == 0
    with conn.cursor() as cursor:
        cursor.execute("SELECT status FROM sync_runs WHERE id = %s", (outcome.run_id,))
        assert cursor.fetchone() == ("halted",)


class _Resume:

    def __init__(self, objection=None) -> None:
        self._objection = objection

    @property
    def origin(self) -> tuple[str, str]:
        return ("rag@example.iam.gserviceaccount.com", "root-folder")

    def objection(self, position: str, issued_by: str, scope: str):
        return self._objection


def test_a_sweep_stamps_whose_position_it_took(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))
    pipeline = replace(world.pipeline, resume=_Resume())

    sweep(conn, pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)

    with conn.cursor() as cursor:
        cursor.execute("SELECT issued_by, scope FROM sync_positions WHERE only_row")
        assert cursor.fetchone() == ("rag@example.iam.gserviceaccount.com", "root-folder")


def test_an_incremental_schedule_sweeps_when_the_source_objects(monkeypatch, conn, caplog):
    world = _world(monkeypatch, names=("doc-1",))
    refused = sweep_because("position_ahead_of_the_source", "the mark is past the source's own")
    pipeline = replace(world.pipeline, resume=_Resume(objection=refused))
    advance_position(
        conn,
        position=world.pipeline.source.start_position(),
        issued_by="rag@example.iam.gserviceaccount.com",
        scope="root-folder",
    )
    run_id = start_run(conn, client=pipeline.client, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    with caplog.at_level(logging.INFO):
        outcome = _fire(
            conn,
            pipeline,
            SCHEDULE_INCREMENTAL,
            app_version=APP_VERSION,
            page_size=10,
            should_stop=lambda: False,
        )

    assert outcome is not None
    with conn.cursor() as cursor:
        cursor.execute("SELECT kind FROM sync_runs WHERE id = %s", (outcome.run_id,))
        assert cursor.fetchone()[0] == "sweep"
    assert "sweeping instead" in caplog.text
    assert "position_ahead_of_the_source" in caplog.text


def test_a_pipeline_whose_source_offers_no_opinion_still_takes_the_delta(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))
    assert world.pipeline.resume is None
    advance_position(conn, position=world.pipeline.source.start_position())
    world.drive.push_change("doc-1")

    outcome = _fire(
        conn,
        world.pipeline,
        SCHEDULE_INCREMENTAL,
        app_version=APP_VERSION,
        page_size=10,
        should_stop=lambda: False,
    )

    with conn.cursor() as cursor:
        cursor.execute("SELECT kind FROM sync_runs WHERE id = %s", (outcome.run_id,))
        assert cursor.fetchone()[0] == "incremental"


def test_a_completed_sweep_removes_a_document_the_source_says_is_trashed(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    sweep(conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)
    assert _state(conn, "doc-2") == DOCUMENT_STATE_INDEXED

    world.drive.files["doc-2"] = replace(world.drive.files["doc-2"], trashed=True)

    outcome = sweep(
        conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False
    )

    assert _state(conn, "doc-2") == DOCUMENT_STATE_GONE
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED
    record = read_run(conn, outcome.run_id)
    assert record.counters.documents_removed == 1
    assert record.counters.documents_unproven == 0


def test_a_sweep_leaves_an_unexplained_disappearance_indexed_and_counted(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    sweep(conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)

    world.drive.files.pop("doc-2")

    outcome = sweep(
        conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False
    )

    assert _state(conn, "doc-2") == DOCUMENT_STATE_INDEXED
    record = read_run(conn, outcome.run_id)
    assert record.counters.documents_removed == 0
    assert record.counters.documents_unproven == 1


def test_a_sweep_that_did_not_finish_removes_nothing(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    sweep(conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)
    world.drive.files.pop("doc-2")

    outcome = sweep(
        conn, world.pipeline, page_size=1, app_version=APP_VERSION, should_stop=lambda: True
    )

    assert outcome.halted is True
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED
    assert _state(conn, "doc-2") == DOCUMENT_STATE_INDEXED


def test_an_incremental_run_reconciles_only_what_the_feed_named(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    sweep(conn, world.pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)

    world.drive.files["doc-1"] = replace(world.drive.files["doc-1"], trashed=True)
    world.drive.push_change("doc-1", removed=True)

    outcome = incremental(conn, world.pipeline, app_version=APP_VERSION)

    assert _state(conn, "doc-1") == DOCUMENT_STATE_GONE
    assert _state(conn, "doc-2") == DOCUMENT_STATE_INDEXED
    assert read_run(conn, outcome.run_id).counters.documents_removed == 1


def test_the_activity_history_is_asked_for_the_window_the_run_covered(monkeypatch, conn):
    asked: list = []

    world = _world(monkeypatch, names=("doc-1",))
    pipeline = replace(world.pipeline, activity=lambda since: asked.append(since) or ())
    sweep(conn, pipeline, page_size=10, app_version=APP_VERSION, should_stop=lambda: False)

    assert asked, "the reconciliation never consulted the activity history"
    assert asked[0] is None


def _state(conn, external_id: str) -> str:
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    return row[0] if row else "absent"
