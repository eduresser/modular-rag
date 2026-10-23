
import pytest

from src.ingest_idempotency import claim, claims_for, in_flight, settle
from src.reprocess import ACTION_READ, stored_state
from src.worker import (
    PAGE_SIZE,
    WorkKey,
    _pages,
    incremental,
    resume,
    sweep,
    unfinished,
    work_key,
)
from src.sync_ledger import advance_position
from tests.conftest import contract_payload, ingest_world, pdf_file

CLAUSES = ("Cláusula primeira: do objeto.", "Cláusula segunda: do prazo.")


def _world(monkeypatch, names=("doc-1",), **kwargs):
    return ingest_world(
        monkeypatch,
        files=[pdf_file(name, content=f"%PDF {name}".encode()) for name in names],
        outputs={name: contract_payload(clauses=CLAUSES) for name in names},
        **kwargs,
    )


def test_the_queue_key_is_the_whole_unit_of_work():
    key = work_key(external_id="doc-1", content_digest="digest", fingerprint="fingerprint")

    assert key == "doc-1:digest:fingerprint"
    assert str(WorkKey("doc-1", "digest", "fingerprint")) == key
    assert work_key(external_id="doc-1", content_digest="digest", fingerprint="other") != key


def test_a_sweep_indexes_everything_and_records_what_it_did(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))

    outcome = sweep(conn, world.pipeline)

    assert outcome.completed is True
    assert outcome.counters.documents_seen == 2
    assert outcome.counters.documents_added == 2
    assert outcome.counters.documents_errored == 0
    assert outcome.indexed == 2
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT status, sweep_completed FROM sync_runs WHERE id = %s", (outcome.run_id,)
        )
        assert cursor.fetchone() == ("completed", True)
        cursor.execute(
            "SELECT external_id FROM sync_observations WHERE run_id = %s ORDER BY external_id",
            (outcome.run_id,),
        )
        assert [row[0] for row in cursor.fetchall()] == ["doc-1", "doc-2"]


def test_a_second_sweep_of_an_unchanged_corpus_counts_it_as_skipped(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1", "doc-2"))
    sweep(conn, world.pipeline)

    again = _world(monkeypatch, names=("doc-1", "doc-2"), storage=world.storage)
    outcome = sweep(conn, again.pipeline)

    assert outcome.counters.documents_skipped == 2
    assert outcome.counters.documents_added == 0
    assert outcome.indexed == 0
    assert again.reads == 0


def test_the_position_is_taken_before_the_enumeration_starts(monkeypatch, conn):
    world = _world(monkeypatch)

    sweep(conn, world.pipeline)

    paths = [str(request.url.path) for request in world.drive.requests]
    assert paths.index("/drive/v3/changes/startPageToken") < paths.index("/drive/v3/files")


def test_an_enumeration_that_stops_closes_the_run_as_failed(monkeypatch, conn):
    world = _world(monkeypatch)

    def explode():
        raise RuntimeError("the source went away")
        yield  # pragma: no cover

    monkeypatch.setattr(world.pipeline.source, "sweep", explode)
    outcome = sweep(conn, world.pipeline)

    assert outcome.completed is False
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT status, sweep_completed, detail FROM sync_runs WHERE id = %s", (outcome.run_id,)
        )
        status, completed, detail = cursor.fetchone()
    assert (status, completed) == ("failed", False)
    assert "the source went away" in detail


def test_the_enumeration_is_paged_without_being_held_in_memory():
    assert [list(page) for page in _pages(iter("abcde"), 2)] == [["a", "b"], ["c", "d"], ["e"]]
    assert list(_pages(iter([]), 2)) == []
    assert PAGE_SIZE >= 1
    with pytest.raises(ValueError):
        list(_pages(iter("ab"), 0))


def test_a_restart_asks_what_this_build_left_in_flight(monkeypatch, conn):
    world = _world(monkeypatch)
    mine = claim(conn, external_id="doc-1", content_digest="d", fingerprint="f", app_version="0.1.0")
    claim(conn, external_id="doc-2", content_digest="d", fingerprint="f", app_version="0.9.0")

    open_claims = unfinished(conn, app_version="0.1.0")

    assert [c.id for c in open_claims] == [mine.claim.id]
    assert [c.external_id for c in unfinished(conn, app_version="0.9.0")] == ["doc-2"]


def test_resuming_takes_up_the_claims_this_build_never_settled(monkeypatch, conn):
    world = _world(monkeypatch)
    claim(
        conn,
        external_id="doc-1",
        content_digest="a-digest-from-before-the-restart",
        fingerprint="a-fingerprint-from-before-the-restart",
        app_version="0.1.0",
    )

    run = resume(conn, world.pipeline, app_version="0.1.0")

    assert [result.external_id for result in run.results] == ["doc-1"]
    assert len(run.indexed) == 1
    assert stored_state(conn, run.results[0].document_id).piece_count == len(CLAUSES)


def test_resuming_leaves_another_build_s_work_alone(monkeypatch, conn):
    world = _world(monkeypatch)
    claim(conn, external_id="doc-1", content_digest="d", fingerprint="f", app_version="0.9.0")

    run = resume(conn, world.pipeline, app_version="0.1.0")

    assert run.results == []
    assert world.reads == 0
    assert [c.state for c in in_flight(conn, app_version="0.9.0")] == ["claimed"]


def test_resuming_skips_a_document_that_vanished_while_the_worker_was_down(monkeypatch, conn):
    world = _world(monkeypatch)
    claim(conn, external_id="doc-404", content_digest="d", fingerprint="f", app_version="0.1.0")

    run = resume(conn, world.pipeline, app_version="0.1.0")

    assert run.results == []
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM documents WHERE external_id = 'doc-404'")
        assert cursor.fetchone()[0] == 0


def test_a_settled_claim_is_not_resumed(monkeypatch, conn):
    world = _world(monkeypatch)
    outcome = claim(conn, external_id="doc-1", content_digest="d", fingerprint="f", app_version="0.1.0")
    settle(conn, outcome.claim.id, action=ACTION_READ)

    assert unfinished(conn, app_version="0.1.0") == ()
    assert resume(conn, world.pipeline, app_version="0.1.0").results == []


def test_an_incremental_run_never_claims_its_observations_are_a_full_sweep(monkeypatch, conn):
    world = _world(monkeypatch, names=("doc-1",))
    world.drive.push_change("doc-1")
    advance_position(conn, position=world.pipeline.source.start_position())

    outcome = incremental(conn, world.pipeline)

    assert outcome.completed is True
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT kind, status, sweep_completed FROM sync_runs WHERE id = %s", (outcome.run_id,)
        )
        assert cursor.fetchone() == ("incremental", "completed", False)
    assert stored_state(conn, outcome.runs[0].results[0].document_id) is not None
    with conn.cursor() as cursor:
        cursor.execute("SELECT position FROM sync_positions WHERE only_row")
        assert cursor.fetchone()[0]


def test_a_change_feed_that_stops_closes_the_run_as_failed(monkeypatch, conn):
    world = _world(monkeypatch)

    def explode(position):
        raise RuntimeError("the feed went away")

    monkeypatch.setattr(world.pipeline.source, "changes_since", explode)
    outcome = incremental(conn, world.pipeline, position="t0")

    assert outcome.completed is False
    with conn.cursor() as cursor:
        cursor.execute("SELECT status, detail FROM sync_runs WHERE id = %s", (outcome.run_id,))
        status, detail = cursor.fetchone()
    assert status == "failed"
    assert "the feed went away" in detail


def test_one_claim_per_document_survives_a_sweep_and_a_resumption(monkeypatch, conn):
    world = _world(monkeypatch)
    sweep(conn, world.pipeline)

    again = _world(monkeypatch, storage=world.storage)
    resume(conn, again.pipeline)

    assert len(claims_for(conn, "doc-1")) == 1
    assert claims_for(conn, "doc-1")[0].state == "done"
