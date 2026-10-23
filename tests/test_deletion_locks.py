"""Unit-level coverage for src/sync_reconcile.py.

AC-012, AC-013, AC-014, AC-020 and principles P-003, P-004 and P-005 are proven
in tests/test_spec_source_gdrive.py — see CLAUDE.md for why those proofs live
there. This file covers the boundaries of the two locks and the cases where
nothing needs removing: how a candidate is classified when it carries more
than one kind of activity event, what happens with no activity at all, the
cases where nothing is a candidate in the first place, and the shape of the
candidate list itself.
"""

import psycopg
import pytest

from src.sources.gdrive_activity import DeletionEvent, DeletionKind, MoveEvent, PermissionChangeEvent, group_by_document
from src.sources.gdrive_export import KIND_DOCUMENT
from src.store import DOCUMENT_STATE_GONE, DOCUMENT_STATE_INDEXED
from src.sync_ledger import (
    RUN_KIND_INCREMENTAL,
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    finish_run,
    read_run,
    start_run,
)
from src.sync_reconcile import (
    Reconciliation,
    absent_since,
    indexed_count,
    indexed_external_ids,
    reconcile,
)
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


def _missing(*names):
    """A drive where every named file answers not-found — no proof at all."""
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True) for n in names]
    )


def _trashed(*names):
    """A drive where every named file still answers, but marked trashed — proof."""
    return FakeDrive(
        [DriveFile(n, n, KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True) for n in names]
    )


def _sweep(conn, *, completed=True):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=completed)
    return read_run(conn, run_id)


def _state(conn, external_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    return row[0] if row else None


def test_reconciliation_defaults_to_an_empty_outcome():
    outcome = Reconciliation()
    assert outcome.removed_anything is False
    assert outcome.vetoed == ()


def test_nothing_reported_and_nothing_missing_is_not_an_error(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = FakeDrive([DriveFile("doc-1", "doc-1", KIND_DOCUMENT, version="1", parents=(FOLDER,))])

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=[], observed=["doc-1"],
    )

    assert outcome.removed == ()
    assert outcome.refusal is None
    assert outcome.considered == 0
    assert "nothing was reported removed" in outcome.detail
    # The source was never asked, because there was nothing to ask about.
    assert not any(r.url.path.endswith("/doc-1") for r in drive.requests)


def test_a_document_reported_removed_that_is_still_there_is_left_alone(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = FakeDrive([DriveFile("doc-1", "doc-1", KIND_DOCUMENT, version="1", parents=(FOLDER,))])

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn), reported_removals=["doc-1"]
    )

    assert outcome.removed == ()
    assert outcome.still_present == ("doc-1",)
    assert outcome.considered == 1
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED


def test_a_deletion_event_is_proof_with_no_probe_needed(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _missing("doc-1")  # the probe would say "not found" — irrelevant here
    events = group_by_document(
        [DeletionEvent(external_id="doc-1", kind=DeletionKind.TRASHED, occurred_at=None)]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["doc-1"], activity=events,
    )

    assert outcome.removed == ("doc-1",)
    assert not any(r.url.path.endswith("/doc-1") for r in drive.requests), (
        "a deletion event is proof on its own; the source was never asked"
    )


def test_a_move_event_vetoes_the_removal_with_no_probe_needed(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _trashed("doc-1")  # even a live trash flag does not override the veto
    events = group_by_document(
        [MoveEvent(external_id="doc-1", removed_parents=(FOLDER,), occurred_at=None)]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["doc-1"], activity=events,
    )

    assert outcome.removed == ()
    assert outcome.moved == ("doc-1",)
    assert outcome.vetoed == ("doc-1",)
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED
    assert not any(r.url.path.endswith("/doc-1") for r in drive.requests)


def test_a_permission_change_event_vetoes_the_removal(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _missing("doc-1")
    events = group_by_document(
        [PermissionChangeEvent(external_id="doc-1", removed_permissions=("reader",), occurred_at=None)]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["doc-1"], activity=events,
    )

    assert outcome.removed == ()
    assert outcome.permission_changed == ("doc-1",)
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED


def test_a_deletion_event_wins_over_a_veto_event_for_the_same_document(conn: psycopg.Connection):
    # A document moved and later deleted still ends up deleted — the positive
    # statement is checked first, whatever else the history says about it.
    _indexed(conn, "doc-1")
    drive = _missing("doc-1")
    events = group_by_document(
        [
            MoveEvent(external_id="doc-1", removed_parents=(FOLDER,), occurred_at=None),
            DeletionEvent(external_id="doc-1", kind=DeletionKind.PERMANENTLY_DELETED, occurred_at=None),
        ]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["doc-1"], activity=events,
    )

    assert outcome.removed == ("doc-1",)
    assert outcome.moved == ()


def test_no_activity_and_a_trashed_probe_is_the_second_proof(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _trashed("doc-1")

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn), reported_removals=["doc-1"]
    )

    assert outcome.removed == ("doc-1",)
    assert _state(conn, "doc-1") == DOCUMENT_STATE_GONE


def test_no_activity_and_a_plain_not_found_is_unproven_not_removed(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    drive = _missing("doc-1")

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn), reported_removals=["doc-1"]
    )

    assert outcome.removed == ()
    assert outcome.unproven == ("doc-1",)
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED


def test_a_document_with_no_activity_record_at_all_falls_back_to_the_probe(conn: psycopg.Connection):
    _indexed(conn, "with-history", "without-history")
    drive = _trashed("with-history", "without-history")
    # Only one of the two has an activity record — a deletion event, which is
    # exactly as good a proof as the trash flag the other one relies on.
    events = group_by_document(
        [DeletionEvent(external_id="with-history", kind=DeletionKind.TRASHED, occurred_at=None)]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["with-history", "without-history"], activity=events,
    )

    assert set(outcome.removed) == {"with-history", "without-history"}
    requested = {r.url.path.removeprefix("/drive/v3/files/") for r in drive.requests if "without-history" in r.url.path}
    assert requested, "the document with no activity record was never probed"


def test_reported_removals_are_acted_on_without_a_sweep(conn: psycopg.Connection):
    # Lock one governs ABSENCE, not reports. An incremental run's report of a
    # removal is still checked for proof and still acted on — otherwise a
    # deletion would only ever be noticed by a full sweep.
    _indexed(conn, "doc-1")
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED)
    run = read_run(conn, run_id)
    assert run.may_remove is False

    outcome = reconcile(
        conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), run=run, reported_removals=["doc-1"]
    )

    assert outcome.removed == ("doc-1",)


def test_absence_is_only_counted_when_the_caller_passes_what_was_observed(conn: psycopg.Connection):
    # Not passing `observed` means "I am not claiming to have seen everything",
    # which has to be the default: the alternative is a caller that forgets and
    # removes the corpus.
    _indexed(conn, "doc-1", "doc-2")

    outcome = reconcile(
        conn, drive_source(_trashed("doc-1", "doc-2"), folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=[],
    )

    assert outcome.removed == ()
    assert outcome.considered == 0


def test_the_candidate_list_is_the_indexed_corpus_minus_what_was_seen(conn: psycopg.Connection):
    _indexed(conn, "seen", "unseen", "also-unseen")

    assert absent_since(conn, ["seen"]) == ("also-unseen", "unseen")
    assert absent_since(conn, ["seen", "unseen", "also-unseen"]) == ()
    assert indexed_external_ids(conn) == {"seen", "unseen", "also-unseen"}


def test_a_gone_document_is_not_a_candidate_again(conn: psycopg.Connection):
    _indexed(conn, "doc-1")
    reconcile(
        conn, drive_source(_trashed("doc-1"), folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=["doc-1"],
    )

    assert absent_since(conn, []) == ()
    assert indexed_count(conn) == 0


def test_removal_is_all_or_nothing(conn: psycopg.Connection):
    # A partially reconciled corpus is worse than an unreconciled one: every
    # proven removal in this batch lands inside one transaction.
    corpus = ["doc-0", "doc-1", "doc-2"]
    _indexed(conn, *corpus)

    outcome = reconcile(
        conn, drive_source(_trashed(*corpus), folder_ids=[FOLDER]), run=_sweep(conn),
        reported_removals=corpus,
    )

    assert set(outcome.removed) == set(corpus)
    assert all(_state(conn, name) == DOCUMENT_STATE_GONE for name in corpus)
