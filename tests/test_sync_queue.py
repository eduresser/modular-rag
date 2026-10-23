
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest

from src.sync_ledger import RUN_KIND_SWEEP, RUN_STATUS_FAILED, finish_run, start_run
from src.sync_queue import (
    SyncBusy,
    active_request,
    attach_run,
    enqueue,
    execution_lock,
    finish_request,
    mark_running,
    read_request,
)
from src.worker import run_requested_sync, run_schedule
from tests.conftest import FakeEmbeddings, contract_payload, ingest_world, pdf_file


def test_admission_does_not_commit_the_callers_transaction(sync_database):
    with psycopg.connect(sync_database) as owner, psycopg.connect(sync_database) as observer:
        request = enqueue(owner)
        assert read_request(observer, request.id) is None
        owner.rollback()
        assert read_request(owner, request.id) is None
        assert active_request(observer) is None
        assert not isinstance(enqueue(observer), SyncBusy)


def test_the_schema_refuses_a_second_active_request_and_an_unknown_state(conn):
    first = enqueue(conn)
    with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
        conn.execute("INSERT INTO sync_requests (id) VALUES (%s)", (uuid4(),))
    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        conn.execute("UPDATE sync_requests SET status = 'misspelled' WHERE id = %s", (first.id,))
    assert active_request(conn).id == first.id
    finish_request(conn, first.id, "completed")
    second = enqueue(conn)
    assert second.id != first.id
    assert read_request(conn, first.id).status == "completed"


@pytest.mark.parametrize("ending", ["normal", "exception", "disconnect", "aborted"])
def test_execution_ownership_survives_commits_but_not_its_owner(sync_database, ending):
    with psycopg.connect(sync_database) as owner, psycopg.connect(sync_database) as other:
        try:
            with execution_lock(owner) as acquired:
                assert acquired
                owner.commit()
                assert isinstance(enqueue(other), SyncBusy)
                with execution_lock(other) as competing:
                    assert not competing
                if ending == "exception":
                    raise RuntimeError("simulated worker failure")
                if ending == "disconnect":
                    owner.close()
                if ending == "aborted":
                    owner.execute("SELECT 1 / 0")
        except (RuntimeError, psycopg.errors.DivisionByZero):
            assert ending in ("exception", "aborted")
        with execution_lock(other) as released:
            assert released
        assert not isinstance(enqueue(other), SyncBusy)


def test_finished_dispatch_is_not_rewritten_and_is_never_executed_again(conn, monkeypatch):
    request = enqueue(conn)
    finish_request(conn, request.id, "failed")
    first = read_request(conn, request.id)
    finish_request(conn, request.id, "completed")
    assert read_request(conn, request.id) == first
    world = ingest_world(monkeypatch, files=[pdf_file("doc-1")])
    assert run_requested_sync(conn, world.pipeline) is False
    assert not world.drive.requests


def test_a_failed_attempt_is_acknowledged_after_restart_without_another_source_call(conn, monkeypatch):
    request = enqueue(conn)
    mark_running(conn, request.id)
    run_id = start_run(conn, client="example", kind=RUN_KIND_SWEEP)
    attach_run(conn, request.id, run_id)
    finish_run(conn, run_id, status=RUN_STATUS_FAILED, detail="internal-source-error")
    world = ingest_world(monkeypatch, files=[pdf_file("doc-1")])
    assert run_requested_sync(conn, world.pipeline) is True
    assert read_request(conn, request.id).status == "failed"
    assert not world.drive.requests
    assert conn.execute("SELECT count(*) FROM sync_runs").fetchone()[0] == 1


def test_graceful_shutdown_requeues_the_same_request_and_reuses_committed_documents(sync_database, monkeypatch):
    world = ingest_world(
        monkeypatch, files=[pdf_file("doc-1"), pdf_file("doc-2")],
        outputs={name: contract_payload() for name in ("doc-1", "doc-2")},
    )
    with psycopg.connect(sync_database) as conn:
        request = enqueue(conn)
        pipeline = replace(world.pipeline, checkpoint=conn.commit)
        assert run_requested_sync(conn, pipeline, page_size=1, should_stop=lambda: world.reads >= 1)
        stopped = read_request(conn, request.id)
        assert stopped.status == "queued" and stopped.finished_at is None
        assert world.reads == 1
    with psycopg.connect(sync_database) as conn:
        assert run_requested_sync(conn, replace(world.pipeline, checkpoint=conn.commit))
        resumed = read_request(conn, request.id)
        assert resumed.status == "completed" and resumed.run_id != stopped.run_id
        assert world.reads == 2


def test_worker_startup_does_not_resume_another_workers_live_claims(sync_database, monkeypatch):
    world = ingest_world(monkeypatch, files=[pdf_file("doc-1")])
    with psycopg.connect(sync_database) as owner, psycopg.connect(sync_database) as contender:
        with execution_lock(owner):
            run_schedule(contender, world.pipeline, should_stop=lambda: False)
        assert not world.drive.requests


def test_an_uncommittable_request_is_a_generic_failure_and_leaves_no_job(api_world, sync_database):
    with psycopg.connect(sync_database) as setup:
        setup.execute("""
            CREATE FUNCTION refuse_sync_commit() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'private-database-detail'; END;
            $$
        """)
        setup.execute("""
            CREATE CONSTRAINT TRIGGER refuse_sync_commit AFTER INSERT ON sync_requests
            DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION refuse_sync_commit()
        """)
    client, _ = api_world(RAG_DB_URL=sync_database)
    response = client.post("/v1/sync")
    assert response.status_code == 503 and "private-database-detail" not in response.text
    with psycopg.connect(sync_database) as check:
        assert active_request(check) is None


def test_a_database_query_error_is_not_reported_as_a_missing_sync(api_world, sync_database):
    with psycopg.connect(sync_database) as setup:
        setup.execute("ALTER TABLE sync_requests RENAME COLUMN status TO broken_status")
    client, _ = api_world(RAG_DB_URL=sync_database)
    assert client.get(f"/v1/sync/{uuid4()}").status_code == 503
    assert client.post("/v1/sync").status_code == 503


def test_partial_document_failures_remain_visible_on_a_completed_sweep(api_world, sync_database, monkeypatch):
    client, service = api_world(RAG_DB_URL=sync_database)
    world = ingest_world(
        monkeypatch, files=[pdf_file("doc-1")], outputs={"doc-1": contract_payload()},
        embeddings=FakeEmbeddings(model=service.config.embed.model,
                                  dimension=service.config.embed.dimension, fail_batches=(0,)),
    )
    sync_id = client.post("/v1/sync").json()["sync_id"]
    with psycopg.connect(sync_database) as conn:
        assert run_requested_sync(conn, replace(world.pipeline, checkpoint=conn.commit))
    response = client.get(f"/v1/sync/{sync_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["counters"]["documents_errored"] == 1
    assert response.json()["counters"]["documents_added"] == 0
