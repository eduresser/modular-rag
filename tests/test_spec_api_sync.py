from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier
from uuid import UUID, uuid4

import anyio
import httpx
import psycopg
import pytest

from src.ingest_hooks import StageHooks
from src.ingest_idempotency import APP_VERSION
from src.sources.gdrive import GoogleDriveSource
from src.sync_ledger import read_run
from src.sync_queue import active_request, read_request
from src.worker import SCHEDULE_FULL_SWEEP, _fire, run_requested_sync, run_schedule
from tests.conftest import contract_payload, ingest_world, pdf_file

CLAUSES = ("Cláusula primeira: sincronização do contrato.", "Cláusula segunda: do prazo.")


class Clock:
    def __init__(self):
        self.current = datetime(2026, 9, 14, 3, tzinfo=timezone.utc)
        self.sleeps = 0

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.current += timedelta(seconds=seconds)
        self.sleeps += 1


def world_for(monkeypatch, dsn, names=("doc-1",), **kwargs):
    return ingest_world(
        monkeypatch, files=[pdf_file(name) for name in names],
        outputs={name: contract_payload(clauses=CLAUSES) for name in names},
        RAG_DB_URL=dsn, **kwargs,
    )


def run_until_settled(dsn, world, client, sync_id):
    clock = Clock()
    with psycopg.connect(dsn) as conn:
        run_schedule(
            conn, replace(world.pipeline, checkpoint=conn.commit),
            should_stop=lambda: clock.sleeps >= 3 or client.get(f"/v1/sync/{sync_id}").json()["status"] in ("completed", "failed"),
            clock=clock.now, sleeper=clock.sleep,
        )
    return client.get(f"/v1/sync/{sync_id}").json()


@pytest.mark.parametrize("body", [None, {}])
def test_ac_221(api_world, sync_database, monkeypatch, body):
    client, service = api_world(RAG_DB_URL=sync_database)
    world = world_for(monkeypatch, sync_database)
    response = client.post("/v1/sync", **({"json": body} if body is not None else {}))
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    sync_id = UUID(payload["sync_id"])
    assert response.headers["location"] == f"/v1/sync/{sync_id}"
    with psycopg.connect(sync_database) as check:
        request = read_request(check, sync_id)
        assert request.status == "queued" and request.run_id is None
        assert check.execute("SELECT count(*) FROM sync_runs").fetchone()[0] == 0
        assert check.execute("SELECT count(*) FROM usage_events").fetchone()[0] == 0
    assert world.reads == 0 and not world.drive.requests
    assert not service.embedder.questions
    other, _ = api_world(RAG_DB_URL=sync_database)
    assert other.get(response.headers["location"]).json() == payload


def test_ac_222(api_world, sync_database):
    client, _ = api_world(RAG_DB_URL=sync_database)
    for token in ("", "Bearer wrong-token"):
        assert client.post("/v1/sync", headers={"Authorization": token}).status_code == 401
        assert client.get(f"/v1/sync/{uuid4()}", headers={"Authorization": token}).status_code == 401
    assert client.post("/v1/sync", headers={"Host": "unrecognised.example.internal"}).status_code == 400
    with psycopg.connect(sync_database) as check:
        assert active_request(check) is None
    assert sorted(tool.name for tool in anyio.run(client.app.state.mcp.list_tools)) == [
        "fetch",
        "filters",
        "outline",
        "search",
    ]


def test_ac_223(api_world, sync_database, monkeypatch):
    client, _ = api_world(RAG_DB_URL=sync_database)
    world = world_for(monkeypatch, sync_database)
    assert world.config.schedule.full_sweep == world.config.schedule.incremental == ""
    sync_id = client.post("/v1/sync").json()["sync_id"]
    result = run_until_settled(sync_database, world, client, sync_id)
    assert result["status"] == "completed" and result["sync_id"] == sync_id
    assert result["counters"]["documents_added"] == 1
    assert result["counters"]["documents_errored"] == 0
    assert result["started_at"] <= result["finished_at"]
    document = client.get("/v1/documents/doc-1").json()
    assert document["status"] == "found"
    assert CLAUSES[0] in "\n".join(piece["text"] for piece in document["pieces"])
    with psycopg.connect(sync_database) as check:
        contents = [row[0] for row in check.execute("SELECT content FROM chunks ORDER BY ordinal")]
        assert CLAUSES[0] in "\n".join(contents)
        assert read_run(check, result["run_id"]).sweep_completed
    assert world.reads == 1
    second = client.post("/v1/sync").json()["sync_id"]
    again = run_until_settled(sync_database, world, client, second)
    assert second != sync_id and again["status"] == "completed"
    assert again["counters"]["documents_skipped"] == 1
    assert world.reads == 1


def test_ac_224(api_world, sync_database, monkeypatch, caplog):
    import logging

    client, _ = api_world(RAG_DB_URL=sync_database)
    barrier = Barrier(2)

    def post():
        barrier.wait(timeout=10)
        return client.post("/v1/sync")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: post(), range(2)))
    assert sorted(response.status_code for response in results) == [202, 409]
    sync_id = next(response.json()["sync_id"] for response in results if response.status_code == 202)
    assert client.post("/v1/sync").json()["sync_id"] == sync_id
    clock = Clock()
    observed = []

    def during_request(document, fetched):
        observed.append(client.post("/v1/sync").status_code)
        assert client.get(f"/v1/sync/{sync_id}").json()["status"] == "running"
        with psycopg.connect(sync_database) as other:
            assert run_requested_sync(other, world.pipeline) is False
        clock.current += timedelta(minutes=2)
        return fetched

    world = world_for(
        monkeypatch, sync_database, hooks=StageHooks(before_send=(during_request,)),
        RAG_SCHEDULE_FULL_SWEEP="* * * * *",
    )
    with caplog.at_level(logging.INFO), psycopg.connect(sync_database) as conn:
        assert _fire(conn, world.pipeline, SCHEDULE_FULL_SWEEP, app_version=APP_VERSION,
                     page_size=50, should_stop=lambda: False) is None
        run_schedule(
            conn, replace(world.pipeline, checkpoint=conn.commit),
            should_stop=lambda: clock.sleeps >= 3 or client.get(f"/v1/sync/{sync_id}").json()["status"] == "completed",
            clock=clock.now, sleeper=clock.sleep,
        )
        assert conn.execute("SELECT count(*) FROM sync_runs").fetchone()[0] == 1
    assert world.reads == 1 and observed == [409]
    assert "requested sync covered its time" in caplog.text

    def during_schedule(document, fetched):
        observed.append(client.post("/v1/sync").status_code)
        return fetched

    scheduled = world_for(monkeypatch, sync_database, names=("doc-2",),
                          hooks=StageHooks(before_send=(during_schedule,)))
    with psycopg.connect(sync_database) as conn:
        outcome = _fire(conn, replace(scheduled.pipeline, checkpoint=conn.commit), SCHEDULE_FULL_SWEEP,
                        app_version=APP_VERSION, page_size=50, should_stop=lambda: False)
        assert outcome.completed
    assert observed == [409, 409]


def test_ac_225(api_world, sync_database, monkeypatch):
    client, _ = api_world(RAG_DB_URL=sync_database)
    sync_id = client.post("/v1/sync").json()["sync_id"]
    world = world_for(monkeypatch, sync_database, names=("doc-1", "doc-2"))
    with psycopg.connect(sync_database) as conn:
        def crash_after_commit():
            conn.commit()
            if conn.execute("SELECT count(*) FROM document_ingest").fetchone()[0] == 1:
                raise SystemExit("simulated process death after a real commit")

        with pytest.raises(SystemExit):
            run_requested_sync(conn, replace(world.pipeline, checkpoint=crash_after_commit), page_size=1)
    assert client.get(f"/v1/sync/{sync_id}").json()["status"] == "running"
    restarted = world_for(monkeypatch, sync_database, names=("doc-1", "doc-2"), storage=world.storage)
    result = run_until_settled(sync_database, restarted, client, sync_id)
    assert result["status"] == "completed" and result["sync_id"] == sync_id
    assert world.reads == restarted.reads == 1
    with psycopg.connect(sync_database) as check:
        assert check.execute("SELECT status FROM sync_runs ORDER BY id").fetchall() == [("halted",), ("completed",)]

    completed_id = client.post("/v1/sync").json()["sync_id"]
    with psycopg.connect(sync_database) as conn:
        def crash_after_finish():
            conn.commit()
            request = read_request(conn, UUID(completed_id))
            if request.run_id and read_run(conn, request.run_id).finished:
                raise SystemExit("simulated death before acknowledgement")

        with pytest.raises(SystemExit):
            run_requested_sync(conn, replace(restarted.pipeline, checkpoint=crash_after_finish))
    with psycopg.connect(sync_database) as check:
        before = check.execute("SELECT count(*) FROM sync_runs").fetchone()[0]
    final = run_until_settled(sync_database, restarted, client, completed_id)
    assert final["status"] == "completed" and restarted.reads == 1
    with psycopg.connect(sync_database) as check:
        assert check.execute("SELECT count(*) FROM sync_runs").fetchone()[0] == before


def test_ac_226(api_world, sync_database, monkeypatch):
    blind, _ = api_world(RAG_DB_URL="postgresql://private-user:private-password@127.0.0.1:1/absent")
    for response in (blind.post("/v1/sync"), blind.get(f"/v1/sync/{uuid4()}")):
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"] == "Synchronisation is unavailable; try again later."
        assert "private-password" not in response.text
    client, _ = api_world(RAG_DB_URL=sync_database)
    assert client.get(f"/v1/sync/{uuid4()}").status_code == 404
    assert client.get("/v1/sync/not-a-uuid").status_code == 422
    assert client.post("/v1/sync", json={"force": True}).status_code == 422
    assert client.post("/v1/sync", json=[]).status_code == 422
    with psycopg.connect(sync_database) as check:
        assert active_request(check) is None

    for fail_at_start in (False, True):
        world = world_for(monkeypatch, sync_database)

        def refused(request):
            if not fail_at_start and request.url.path.endswith("startPageToken"):
                return httpx.Response(200, json={"startPageToken": "t0"})
            return httpx.Response(403, json={"error": {"message": "provider-private-diagnostic"}})

        source = GoogleDriveSource(folder_ids=world.config.source.folder_ids,
                                   http=httpx.Client(transport=httpx.MockTransport(refused)), token=lambda: "fake-token")
        world = replace(world, pipeline=replace(world.pipeline, source=source))
        sync_id = client.post("/v1/sync").json()["sync_id"]
        result = run_until_settled(sync_database, world, client, sync_id)
        assert result["status"] == "failed"
        assert result["error"] == "Synchronisation failed; check the worker logs."
        assert result["counters"]["documents_added"] == 0
        assert "provider-private-diagnostic" not in str(result)
    assert client.post("/v1/sync").status_code == 202
