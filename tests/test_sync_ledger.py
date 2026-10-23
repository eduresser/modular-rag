
from datetime import datetime, timezone

import psycopg
import pytest

from src.sync_ledger import (
    RUN_KIND_INCREMENTAL,
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_HALTED,
    RUN_STATUS_RUNNING,
    LedgerError,
    RunCounters,
    advance_position,
    commit_page,
    current_position,
    finish_run,
    last_completed_sweep,
    observed,
    read_run,
    recent_runs,
    record_observed,
    start_run,
)
from tests.conftest import TEST_CLIENT


def test_a_run_starts_running_with_no_finish_time(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    run = read_run(conn, run_id)

    assert run.status == RUN_STATUS_RUNNING
    assert run.finished is False
    assert run.finished_at is None
    assert run.kind == RUN_KIND_SWEEP
    assert run.client == TEST_CLIENT


def test_an_unknown_run_kind_is_refused_and_lists_the_kinds():
    class _NoConnection:
        pass

    with pytest.raises(LedgerError) as excinfo:
        start_run(_NoConnection(), client=TEST_CLIENT, kind="partial")

    message = str(excinfo.value)
    assert "partial" in message
    assert RUN_KIND_SWEEP in message
    assert RUN_KIND_INCREMENTAL in message


def test_a_run_without_a_client_is_refused():
    class _NoConnection:
        pass

    with pytest.raises(LedgerError):
        start_run(_NoConnection(), client="   ", kind=RUN_KIND_SWEEP)


def test_a_run_cannot_finish_as_still_running(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(LedgerError) as excinfo:
        finish_run(conn, run_id, status=RUN_STATUS_RUNNING)

    assert RUN_STATUS_COMPLETED in str(excinfo.value)


def test_a_failed_run_cannot_claim_its_sweep_completed(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(LedgerError) as excinfo:
        finish_run(conn, run_id, status=RUN_STATUS_FAILED, sweep_completed=True)

    assert "P-003" in str(excinfo.value)


def test_a_halted_run_cannot_claim_its_sweep_completed_either(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(LedgerError):
        finish_run(conn, run_id, status=RUN_STATUS_HALTED, sweep_completed=True)


def test_finishing_a_run_that_does_not_exist_is_refused(conn: psycopg.Connection):
    with pytest.raises(LedgerError) as excinfo:
        finish_run(conn, 10_000_000, status=RUN_STATUS_COMPLETED)

    assert "10000000" in str(excinfo.value)


def test_only_a_completed_sweep_is_permission_to_remove(conn: psycopg.Connection):
    def _run(kind, status, sweep_completed):
        run_id = start_run(conn, client=TEST_CLIENT, kind=kind)
        finish_run(conn, run_id, status=status, sweep_completed=sweep_completed)
        return read_run(conn, run_id)

    assert _run(RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, True).may_remove is True
    assert _run(RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, False).may_remove is False
    assert _run(RUN_KIND_SWEEP, RUN_STATUS_FAILED, False).may_remove is False
    assert _run(RUN_KIND_INCREMENTAL, RUN_STATUS_COMPLETED, False).may_remove is False


def test_the_database_refuses_a_running_row_with_a_finish_time(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("UPDATE sync_runs SET finished_at = now() WHERE id = %s", (run_id,))


def test_the_database_refuses_a_completed_sweep_on_a_running_row(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("UPDATE sync_runs SET sweep_completed = true WHERE id = %s", (run_id,))


def test_observations_are_idempotent_within_a_run(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    record_observed(conn, run_id, ["a", "b", "a"])
    record_observed(conn, run_id, ["b", "c"])

    assert observed(conn, run_id) == {"a", "b", "c"}


def test_blank_identifiers_are_not_observed(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    assert record_observed(conn, run_id, ["", None and ""]) == 0
    assert observed(conn, run_id) == frozenset()


def test_two_runs_do_not_share_observations(conn: psycopg.Connection):
    sweep = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    incremental = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    record_observed(conn, sweep, ["a", "b", "c"])
    record_observed(conn, incremental, ["b"])

    assert observed(conn, sweep) == {"a", "b", "c"}
    assert observed(conn, incremental) == {"b"}


def test_the_position_starts_absent_and_is_a_single_row(conn: psycopg.Connection):
    assert current_position(conn) is None

    advance_position(conn, position="token-1")
    advance_position(conn, position="token-2")

    assert current_position(conn) == "token-2"
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM sync_positions")
        assert cursor.fetchone()[0] == 1


def test_committing_a_page_records_its_documents_and_moves_the_position(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    commit_page(conn, run_id=run_id, external_ids=["a", "b"], next_position="token-2")

    assert observed(conn, run_id) == {"a", "b"}
    assert current_position(conn) == "token-2"


def test_a_page_cannot_be_committed_without_the_position_after_it(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    with pytest.raises(LedgerError) as excinfo:
        commit_page(conn, run_id=run_id, external_ids=["a"], next_position="")

    assert "AC-011" in str(excinfo.value)


def test_the_counters_are_stored_as_given(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    counters = RunCounters(
        documents_seen=120,
        documents_added=4,
        documents_updated=7,
        documents_skipped=108,
        documents_removed=1,
        documents_errored=2,
        access_lost=3,
    )

    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, counters=counters, sweep_completed=True)

    assert read_run(conn, run_id).counters == counters


def test_a_finished_run_keeps_the_position_it_reached(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    commit_page(conn, run_id=run_id, external_ids=["a"], next_position="token-9")

    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED)

    assert read_run(conn, run_id).feed_position == "token-9"


def test_the_last_completed_sweep_ignores_the_ones_that_did_not_finish(conn: psycopg.Connection):
    failed = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, failed, status=RUN_STATUS_FAILED, detail="credentials expired")
    good = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, good, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    later_failure = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, later_failure, status=RUN_STATUS_FAILED)

    assert last_completed_sweep(conn).id == good


def test_there_may_never_have_been_a_completed_sweep(conn: psycopg.Connection):
    assert last_completed_sweep(conn) is None


def test_recent_runs_come_back_newest_first(conn: psycopg.Connection):
    first = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, first, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    second = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    assert [run.id for run in recent_runs(conn, limit=2)] == [second, first]


def test_reading_a_run_that_does_not_exist_returns_none(conn: psycopg.Connection):
    assert read_run(conn, 10_000_000) is None
