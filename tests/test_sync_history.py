
import psycopg
import pytest

from src.store import upsert_document
from src.sync_history import (
    HistoryError,
    document_history,
    finish_exclusive_run,
    list_runs,
    locked_run,
    mark_document_seen,
    start_exclusive_run,
)
from src.sync_ledger import RUN_KIND_INCREMENTAL, RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, finish_run, start_run
from tests.conftest import TEST_CLIENT


def test_locked_run_is_none_when_nothing_is_running(conn: psycopg.Connection):
    assert locked_run(conn) is None


def test_start_exclusive_run_opens_a_run_when_none_is_running(conn: psycopg.Connection):
    run_id = start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    assert locked_run(conn).id == run_id


def test_start_exclusive_run_refuses_and_names_the_run_in_progress(conn: psycopg.Connection):
    first = start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(HistoryError, match=str(first)):
        start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM sync_runs")
        assert cursor.fetchone()[0] == 1


def test_start_exclusive_run_succeeds_again_once_the_running_one_finished(conn: psycopg.Connection):
    first = start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_exclusive_run(conn, first, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    second = start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    assert second != first
    assert locked_run(conn).id == second


def test_the_database_itself_refuses_a_second_lock_row(conn: psycopg.Connection):
    first = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO sync_run_lock (run_id) VALUES (%s)", (first,))

    second = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO sync_run_lock (run_id) VALUES (%s)", (second,))


def test_the_database_itself_refuses_rewriting_a_finished_run(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    with pytest.raises(psycopg.errors.RaiseException, match="AC-023"), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("UPDATE sync_runs SET detail = 'rewritten' WHERE id = %s", (run_id,))


def test_list_runs_is_newest_first(conn: psycopg.Connection):
    older = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, older, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    newer = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    finish_run(conn, newer, status=RUN_STATUS_COMPLETED)

    runs = list_runs(conn)

    assert [run.id for run in runs] == [newer, older]


def test_mark_document_seen_refuses_a_document_that_does_not_exist(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(HistoryError, match="no-such-document"):
        mark_document_seen(conn, external_id="no-such-document", run_id=run_id)


def test_document_history_is_none_for_a_document_that_does_not_exist(conn: psycopg.Connection):
    assert document_history(conn, "no-such-document") is None


def test_document_history_names_the_run_that_first_saw_it_and_the_one_that_saw_it_last(
    conn: psycopg.Connection,
):
    upsert_document(conn, external_id="doc-1")
    first_run = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    mark_document_seen(conn, external_id="doc-1", run_id=first_run)
    finish_run(conn, first_run, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    later_run = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    mark_document_seen(conn, external_id="doc-1", run_id=later_run)

    history = document_history(conn, "doc-1")

    assert history.first_seen_run_id == first_run
    assert history.last_seen_run_id == later_run


def test_mark_document_seen_leaves_first_seen_alone_on_a_repeat_within_the_same_run(
    conn: psycopg.Connection,
):
    upsert_document(conn, external_id="doc-1")
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    mark_document_seen(conn, external_id="doc-1", run_id=run_id)
    mark_document_seen(conn, external_id="doc-1", run_id=run_id)

    history = document_history(conn, "doc-1")

    assert history.first_seen_run_id == run_id
    assert history.last_seen_run_id == run_id
