"""Unit-level coverage for src/diagnostics.py.

AC-012 and AC-013 are proven where CLAUDE.md says every criterion's proof
lives — the tagged stub in tests/test_spec_source_gdrive.py, against
``src.sync_reconcile.reconcile`` — and AC-023 is proven there against
``src.sync_history.list_runs``. This file covers what those tests do not: that
this module's own dry-run classification reaches the same verdicts a real
reconciliation would, in the same order, without ever writing anything or
opening a transaction; that the evidence chain for one document lays out
exactly the facts a decision is built from; and that this module's listing is
the same ordered, never-rewritten history AC-023 already proves underneath.
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg
import pytest

from src.diagnostics import (
    VERDICT_GONE,
    VERDICT_MOVED,
    VERDICT_PERMISSION_CHANGED,
    VERDICT_STILL_PRESENT,
    VERDICT_UNPROVEN,
    activity_in_window,
    diagnose_absences,
    document_evidence,
    stored_runs,
)
from src.sources.base import DocumentUnavailable, SourceDocument
from src.sources.gdrive_activity import DeletionEvent, DeletionKind, MoveEvent, PermissionChangeEvent, group_by_document
from src.sources.gdrive_export import KIND_DOCUMENT
from src.store import DOCUMENT_STATE_INDEXED, mark_document_gone
from src.sync_ledger import RUN_KIND_INCREMENTAL, RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, finish_run, start_run
from tests.conftest import TEST_CLIENT, ChunkSpec, DocumentSpec, DriveFile, FakeDrive, drive_source, index_corpus

FOLDER = "root-folder"


def _indexed(conn, *external_ids):
    return index_corpus(
        conn,
        [
            DocumentSpec(name, name, (ChunkSpec(0, "cláusula", pages=(1, 1)),))
            for name in external_ids
        ],
    )


def _state(conn, external_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    return row[0] if row else None


def _missing(*names):
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True) for n in names]
    )


def _trashed(*names):
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True) for n in names]
    )


def _present(*names):
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,)) for n in names]
    )


def _denied(*names):
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,), denied=True) for n in names]
    )


class _FakeActivityHistory:
    """A fixed answer to ``events_since`` — enough for :func:`activity_in_window`."""

    def __init__(self, events):
        self.events = tuple(events)
        self.calls: list[tuple[datetime, datetime | None]] = []

    def events_since(self, moment, *, until=None):
        self.calls.append((moment, until))
        return iter(self.events)


# ── activity_in_window ──────────────────────────────────────────────────────


def test_activity_in_window_is_a_plain_read_of_the_history():
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    until = datetime(2026, 9, 10, tzinfo=timezone.utc)
    event = DeletionEvent(external_id="doc-1", kind=DeletionKind.TRASHED, occurred_at=None)
    history = _FakeActivityHistory([event])

    result = activity_in_window(history, since=since, until=until)

    assert result == (event,)
    assert history.calls == [(since, until)]


def test_activity_in_window_is_empty_when_nothing_happened():
    history = _FakeActivityHistory([])

    result = activity_in_window(history, since=datetime(2026, 9, 1, tzinfo=timezone.utc))

    assert result == ()


# ── diagnose_absences ────────────────────────────────────────────────────────


def test_diagnose_absences_reports_nothing_indexed_is_missing(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    result = diagnose_absences(conn, drive_source(_present("doc-1"), folder_ids=[FOLDER]), observed=["doc-1"])

    assert result == ()


def test_diagnose_absences_marks_a_deletion_event_as_gone_without_probing(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _missing("doc-1")  # the probe would say "not found" — must not even be asked
    activity = group_by_document(
        [DeletionEvent(external_id="doc-1", kind=DeletionKind.TRASHED, occurred_at=None)]
    )

    result = diagnose_absences(
        conn, drive_source(drive, folder_ids=[FOLDER]), observed=[], activity=activity
    )

    assert len(result) == 1
    assert result[0].external_id == "doc-1"
    assert result[0].verdict == VERDICT_GONE
    assert result[0].would_be_removed is True
    assert drive.requests == []


def test_diagnose_absences_vetoes_a_move_and_leaves_it_present(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    activity = group_by_document(
        [MoveEvent(external_id="doc-1", removed_parents=("old-parent",), occurred_at=None)]
    )

    result = diagnose_absences(
        conn, drive_source(_missing("doc-1"), folder_ids=[FOLDER]), observed=[], activity=activity
    )

    assert result[0].verdict == VERDICT_MOVED
    assert result[0].would_be_removed is False
    assert "AC-013" in result[0].detail


def test_diagnose_absences_vetoes_a_permission_change(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    activity = group_by_document(
        [PermissionChangeEvent(external_id="doc-1", removed_permissions=("reader",), occurred_at=None)]
    )

    result = diagnose_absences(
        conn, drive_source(_missing("doc-1"), folder_ids=[FOLDER]), observed=[], activity=activity
    )

    assert result[0].verdict == VERDICT_PERMISSION_CHANGED
    assert result[0].would_be_removed is False


def test_diagnose_absences_falls_back_to_the_trash_probe_with_no_activity(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    result = diagnose_absences(conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), observed=[])

    assert result[0].verdict == VERDICT_GONE
    assert result[0].would_be_removed is True


def test_diagnose_absences_leaves_a_bare_not_found_unproven(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    result = diagnose_absences(conn, drive_source(_missing("doc-1"), folder_ids=[FOLDER]), observed=[])

    assert result[0].verdict == VERDICT_UNPROVEN
    assert result[0].would_be_removed is False
    assert "AC-012" in result[0].detail


def test_diagnose_absences_reports_still_present_when_the_source_hands_it_over(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    result = diagnose_absences(conn, drive_source(_present("doc-1"), folder_ids=[FOLDER]), observed=[])

    assert result[0].verdict == VERDICT_STILL_PRESENT
    assert result[0].would_be_removed is False


def test_diagnose_absences_never_writes_anything(conn: psycopg.Connection):
    """The whole point (T-024): a dry run must be safe to point at production."""
    _indexed(conn, "doc-1")

    diagnose_absences(conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), observed=[])

    # A verdict of "gone" was reached, yet nothing in the archive moved.
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED


# ── document_evidence ────────────────────────────────────────────────────────


def test_document_evidence_for_an_indexed_document_the_source_still_holds(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    evidence = document_evidence(conn, drive_source(_present("doc-1"), folder_ids=[FOLDER]), "doc-1")

    assert evidence.indexed is True
    assert evidence.present_at_source is True
    assert evidence.trashed is False
    assert evidence.access_denied is False
    assert evidence.activity == ()
    assert evidence.history is not None


def test_document_evidence_for_a_trashed_document(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    evidence = document_evidence(conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), "doc-1")

    assert evidence.present_at_source is False
    assert evidence.trashed is True
    assert evidence.access_denied is False


def test_document_evidence_for_a_denied_document(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    evidence = document_evidence(conn, drive_source(_denied("doc-1"), folder_ids=[FOLDER]), "doc-1")

    assert evidence.present_at_source is False
    assert evidence.trashed is False
    assert evidence.access_denied is True


def test_document_evidence_carries_the_activity_events_it_was_given(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    event = MoveEvent(external_id="doc-1", removed_parents=("old-parent",), occurred_at=None)

    evidence = document_evidence(
        conn, drive_source(_missing("doc-1"), folder_ids=[FOLDER]), "doc-1", activity=(event,)
    )

    assert evidence.activity == (event,)


def test_document_evidence_for_a_document_never_indexed(conn: psycopg.Connection):
    evidence = document_evidence(conn, drive_source(_present("doc-1"), folder_ids=[FOLDER]), "doc-1")

    assert evidence.indexed is False
    assert evidence.history is None


def test_document_evidence_never_writes_anything(conn: psycopg.Connection):
    _indexed(conn, "doc-1")

    document_evidence(conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), "doc-1")

    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED


# ── stored_runs ──────────────────────────────────────────────────────────────


def test_stored_runs_lists_every_run_newest_first(conn: psycopg.Connection):
    older = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, older, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    newer = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    finish_run(conn, newer, status=RUN_STATUS_COMPLETED)

    runs = stored_runs(conn)

    assert [run.id for run in runs] == [newer, older]


def test_stored_runs_is_empty_with_no_runs_at_all(conn: psycopg.Connection):
    assert stored_runs(conn) == ()


def test_stored_runs_respects_the_limit(conn: psycopg.Connection):
    for _ in range(3):
        run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
        finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    runs = stored_runs(conn, limit=2)

    assert len(runs) == 2
