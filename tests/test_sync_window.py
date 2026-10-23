
from datetime import datetime, timezone

import psycopg
import pytest

from src.sync_ledger import (
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_HALTED,
    LedgerError,
    finish_run,
    next_window_start,
    read_run,
    require_baseline_sweep,
    start_delta_run,
    start_run,
)
from tests.conftest import TEST_CLIENT


def test_there_is_no_window_before_any_run_has_completed(conn: psycopg.Connection):
    assert next_window_start(conn) is None


def test_require_baseline_sweep_refuses_with_no_runs_at_all(conn: psycopg.Connection):
    with pytest.raises(LedgerError, match="AC-021"):
        require_baseline_sweep(conn)


def test_require_baseline_sweep_refuses_a_sweep_that_only_failed(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_FAILED, detail="the source timed out")

    with pytest.raises(LedgerError, match="AC-021"):
        require_baseline_sweep(conn)


def test_require_baseline_sweep_passes_once_one_sweep_has_completed(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    require_baseline_sweep(conn)


def test_start_delta_run_refuses_and_opens_nothing_without_a_baseline(conn: psycopg.Connection):
    with pytest.raises(LedgerError, match="AC-021"):
        start_delta_run(conn, client=TEST_CLIENT)

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM sync_runs")
        assert cursor.fetchone()[0] == 0


def test_a_completed_run_can_record_its_window_end(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    edge = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)

    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True, window_end=edge)

    assert read_run(conn, run_id).window_end == edge
    assert next_window_start(conn) == edge


@pytest.mark.parametrize("status", [RUN_STATUS_FAILED, RUN_STATUS_HALTED])
def test_a_run_that_did_not_complete_cannot_record_a_window_end(conn: psycopg.Connection, status: str):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(LedgerError, match="AC-022"):
        finish_run(conn, run_id, status=status, window_end=datetime.now(timezone.utc))


def test_the_database_itself_refuses_a_window_end_on_a_run_that_did_not_complete(
    conn: psycopg.Connection,
):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_FAILED)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("UPDATE sync_runs SET window_end = now() WHERE id = %s", (run_id,))


def test_a_new_runs_window_start_is_the_last_completed_runs_edge(conn: psycopg.Connection):
    baseline = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    edge = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)
    finish_run(conn, baseline, status=RUN_STATUS_COMPLETED, sweep_completed=True, window_end=edge)

    delta = start_delta_run(conn, client=TEST_CLIENT)

    assert read_run(conn, delta).window_start == edge
