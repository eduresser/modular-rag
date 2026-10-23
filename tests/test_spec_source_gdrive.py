
from datetime import datetime, timezone

import psycopg
import pytest

from src.sources.base import DocumentUnavailable, SourceError, registered_slugs, source_class
from src.processing.preflight import UNPAGINATED_MEDIA_TYPES, Candidate, preflight
from src.sources.gdrive_export import (
    KIND_DOCUMENT,
    KIND_FOLDER,
    KIND_SHORTCUT,
    KIND_SPREADSHEET,
    MEDIA_DOCX,
    MEDIA_PDF,
)
from src.sources.gdrive_activity import (
    DeletionEvent,
    DeletionKind,
    MoveEvent,
    PermissionChangeEvent,
    group_by_document,
)
from src.sources.identity import Action, StoredIdentity, after_fetch, before_fetch
from src.store import DOCUMENT_STATE_GONE, DOCUMENT_STATE_INDEXED, mark_document_indexed, upsert_document
from src.sync_access import Verdict, probe_removals
from src.sync_history import HistoryError, document_history, list_runs, mark_document_seen, start_exclusive_run
from src.sync_ledger import (
    RUN_KIND_INCREMENTAL,
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_HALTED,
    LedgerError,
    RunCounters,
    advance_position,
    commit_page,
    current_position,
    finish_run,
    observed,
    read_run,
    recent_runs,
    start_delta_run,
    start_run,
)
from src.sources.gdrive_resume import (
    REASON_AHEAD_OF_SOURCE,
    REASON_ANOTHER_IDENTITY,
    REASON_ANOTHER_TREE,
    DriveResume,
)
from src.sync_decision import (
    REASON_NO_BASELINE,
    REASON_NO_POSITION,
    REASON_POSITION_HOLDS,
    decide_next_run,
)
from src.sync_reconcile import reconcile
from src.usage import REASON_CONTENT_CHANGED, REASON_WATERMARK_MOVED
from tests.conftest import (
    DriveChange,
    HEADING_LEVELS,
    TEST_CLIENT,
    ChunkSpec,
    DocumentSpec,
    DriveFile,
    FakeDrive,
    document_xml,
    drive_source,
    heading_styles,
    index_corpus,
    word_document,
)

FOLDER = "root-folder"
FINGERPRINT = "fp-1"


def _document(identifier: str, *, version: str = "1", content: bytes = b"# Titulo\n\ntexto") -> DriveFile:
    return DriveFile(
        identifier, f"Contrato {identifier}", KIND_DOCUMENT,
        version=version, parents=(FOLDER,), content=content,
    )


def _state(conn: psycopg.Connection, external_id: str) -> str | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    return row[0] if row else None


def _indexed(conn: psycopg.Connection, *external_ids: str) -> dict[str, int]:
    return index_corpus(
        conn,
        [
            DocumentSpec(identifier, f"Contrato {identifier}", (ChunkSpec(0, "cláusula", pages=(1, 1)),))
            for identifier in external_ids
        ],
    )


def _completed_sweep(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    return read_run(conn, run_id)


def test_ac_010(conn: psycopg.Connection):
    """AC-010: Only what changed since the last run is fetched @spec:AC-010"""
    drive = FakeDrive([_document("doc-1"), _document("doc-2", version="3")])
    source = drive_source(drive, folder_ids=[FOLDER])
    first = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    commit_page(conn, run_id=first, external_ids=["doc-1", "doc-2"], next_position="t0")
    finish_run(conn, first, status=RUN_STATUS_COMPLETED)
    assert current_position(conn) == "t0"

    drive.push_change("doc-2")

    page = source.changes_since(current_position(conn))

    feed_calls = [r for r in drive.requests if r.url.path.endswith("/changes")]
    assert feed_calls, "the run never asked the change feed"
    assert dict(feed_calls[-1].url.params)["pageToken"] == "t0"
    assert [document.external_id for document in page.documents] == ["doc-2"]

    stored = StoredIdentity(
        external_id="doc-1", content_digest="whatever", indexing_fingerprint=FINGERPRINT,
        revision_marker="1",
    )
    decided_without_bytes = before_fetch(stored=stored, revision_marker="1", fingerprint=FINGERPRINT)
    assert decided_without_bytes is not None
    assert decided_without_bytes.action is Action.TOUCH_ONLY

    downloads = [r for r in drive.requests if dict(r.url.params).get("alt") == "media"]
    exports = [r for r in drive.requests if r.url.path.endswith("/export")]
    assert downloads == [] and exports == []

    for request in drive.requests:
        params = dict(request.url.params)
        assert "fields" in params, request.url
        assert params["includeItemsFromAllDrives"] == "true", request.url


def test_ac_011(conn: psycopg.Connection):
    """AC-011: Progress is not lost when a run dies mid-way @spec:AC-011"""
    drive = FakeDrive([_document(f"doc-{n}") for n in range(4)], page_size=2)
    for n in range(4):
        drive.push_change(f"doc-{n}")
    source = drive_source(drive, folder_ids=[FOLDER])
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    first_page = source.changes_since("t0")
    assert [d.external_id for d in first_page.documents] == ["doc-0", "doc-1"]
    commit_page(
        conn,
        run_id=run_id,
        external_ids=[d.external_id for d in first_page.documents],
        next_position=first_page.next_position,
    )
    position_after_first_page = current_position(conn)
    assert position_after_first_page == "t2"

    second_page = source.changes_since(position_after_first_page)
    assert [d.external_id for d in second_page.documents] == ["doc-2", "doc-3"]
    finish_run(conn, run_id, status=RUN_STATUS_FAILED, detail="killed")

    assert current_position(conn) == "t2"
    resumed = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    retried = source.changes_since(current_position(conn))
    assert [d.external_id for d in retried.documents] == ["doc-2", "doc-3"]

    commit_page(conn, run_id=resumed, external_ids=["doc-2", "doc-3"], next_position="t4")
    commit_page(conn, run_id=resumed, external_ids=["doc-2", "doc-3"], next_position="t4")
    assert observed(conn, resumed) == {"doc-2", "doc-3"}
    written = _indexed(conn, "doc-2")
    again = _indexed(conn, "doc-2")
    assert written == again, "the same identity produced a second document row"


def test_ac_018(conn: psycopg.Connection):
    """AC-018: A document whose watermark moved but whose content did not costs nothing @spec:AC-018"""
    unchanged_bytes = b"# Contrato\n\nCl\xc3\xa1usula quinta"
    drive = FakeDrive([_document("doc-1", version="7", content=unchanged_bytes)])
    source = drive_source(drive, folder_ids=[FOLDER])
    original = source.fetch(next(iter(source.sweep())))
    stored = StoredIdentity(
        external_id="doc-1",
        content_digest=original.content_digest,
        indexing_fingerprint=FINGERPRINT,
        revision_marker="7",
    )

    drive.files["doc-1"].version = "8"

    before = before_fetch(stored=stored, revision_marker="8", fingerprint=FINGERPRINT)
    assert before is None
    document = next(iter(source.sweep()))
    assert document.revision_marker == "8"
    fetched = source.fetch(document)
    decision = after_fetch(
        stored=stored, content_digest=fetched.content_digest, fingerprint=FINGERPRINT
    )

    assert decision.action is Action.TOUCH_ONLY

    assert decision.costs_a_read is False
    assert decision.costs_anything is False
    assert decision.reason == REASON_WATERMARK_MOVED

    drive.files["doc-1"].content = unchanged_bytes + b" alterada"
    edited = source.fetch(next(iter(source.sweep())))
    real_change = after_fetch(
        stored=stored, content_digest=edited.content_digest, fingerprint=FINGERPRINT
    )
    assert real_change.action is Action.INDEX_FROM_SOURCE
    assert real_change.reason == REASON_CONTENT_CHANGED


def test_ac_012(conn: psycopg.Connection):
    """AC-012: A document is removed only on positive proof of deletion @spec:AC-012"""
    drive = FakeDrive(
        [DriveFile("no-proof", "Contrato", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True)]
    )
    _indexed(conn, "no-proof")

    unproven = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["no-proof"],
    )

    assert unproven.removed == ()
    assert unproven.unproven == ("no-proof",)
    assert _state(conn, "no-proof") == DOCUMENT_STATE_INDEXED

    _indexed(conn, "named-in-history")
    drive_history = FakeDrive(
        [DriveFile("named-in-history", "Contrato", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True)]
    )
    deletion = group_by_document(
        [DeletionEvent(external_id="named-in-history", kind=DeletionKind.PERMANENTLY_DELETED, occurred_at=None)]
    )
    proven_by_history = reconcile(
        conn, drive_source(drive_history, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["named-in-history"], activity=deletion,
    )
    assert proven_by_history.removed == ("named-in-history",)
    assert _state(conn, "named-in-history") == DOCUMENT_STATE_GONE

    _indexed(conn, "trashed")
    drive_trashed = FakeDrive(
        [DriveFile("trashed", "Contrato", KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True)]
    )
    proven_by_trash = reconcile(
        conn, drive_source(drive_trashed, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["trashed"],
    )
    assert proven_by_trash.removed == ("trashed",)
    assert _state(conn, "trashed") == DOCUMENT_STATE_GONE


def test_ac_013(conn: psycopg.Connection):
    """AC-013: An explanation other than deletion vetoes the removal @spec:AC-013"""
    drive = FakeDrive(
        [
            DriveFile("moved-doc", "Contrato", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True),
            DriveFile("locked-doc", "Contrato", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True),
        ]
    )
    _indexed(conn, "moved-doc", "locked-doc")
    explanations = group_by_document(
        [
            MoveEvent(external_id="moved-doc", removed_parents=(FOLDER,), occurred_at=None),
            PermissionChangeEvent(external_id="locked-doc", removed_permissions=("reader",), occurred_at=None),
        ]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["moved-doc", "locked-doc"], activity=explanations,
    )

    assert outcome.removed == ()
    assert outcome.moved == ("moved-doc",)
    assert outcome.permission_changed == ("locked-doc",)
    assert outcome.vetoed == ("locked-doc", "moved-doc")
    assert _state(conn, "moved-doc") == DOCUMENT_STATE_INDEXED
    assert _state(conn, "locked-doc") == DOCUMENT_STATE_INDEXED
    assert "1 left indexed — moved out of the watched tree" in outcome.detail
    assert "1 left indexed — a permission change" in outcome.detail


def test_ac_020(conn: psycopg.Connection):
    """AC-020: Absence removes nothing unless the sweep completed @spec:AC-020"""
    drive = FakeDrive(
        [DriveFile("survivor", "Sobrevivente", KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True)]
    )
    source = drive_source(drive, folder_ids=[FOLDER])
    _indexed(conn, "survivor", "also-here")
    failed_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, failed_id, status=RUN_STATUS_FAILED, detail="the process was killed at 30%")
    failed = read_run(conn, failed_id)

    outcome = reconcile(conn, source, run=failed, observed=[])

    assert outcome.removed == ()
    assert outcome.considered == 0
    assert _state(conn, "survivor") == DOCUMENT_STATE_INDEXED
    assert _state(conn, "also-here") == DOCUMENT_STATE_INDEXED
    assert not any(r.url.path.endswith("/survivor") for r in drive.requests)

    assert failed.sweep_completed is False
    assert failed.may_remove is False
    assert outcome.refusal == "sweep_did_not_complete"
    assert "P-003" in outcome.detail

    completed = _completed_sweep(conn)
    allowed = reconcile(conn, source, run=completed, observed=["also-here"])
    assert allowed.removed == ("survivor",)


def test_ac_014(conn: psycopg.Connection):
    """AC-014: A document that comes back is not duplicated @spec:AC-014"""
    drive = FakeDrive([_document("doc-1")])
    source = drive_source(drive, folder_ids=[FOLDER])
    written = _indexed(conn, "doc-1")
    original_id = written["doc-1"]
    drive.files["doc-1"].trashed = True
    reconcile(conn, source, run=_completed_sweep(conn), reported_removals=["doc-1"])
    assert _state(conn, "doc-1") == DOCUMENT_STATE_GONE
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (original_id,))
        assert cursor.fetchone()[0] == 0, "the pieces of a gone document still answer queries"

    drive.files["doc-1"].trashed = False
    reappeared = next(iter(source.sweep()))
    assert reappeared.external_id == "doc-1"
    assert mark_document_indexed(conn, external_id="doc-1") is True
    reindexed = _indexed(conn, "doc-1")

    assert reindexed["doc-1"] == original_id
    assert _state(conn, "doc-1") == DOCUMENT_STATE_INDEXED

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM documents WHERE external_id = %s", ("doc-1",))
        assert cursor.fetchone()[0] == 1


def test_ac_015(conn: psycopg.Connection):
    """AC-015: A shortcut is indexed as its target @spec:AC-015"""
    elsewhere = DriveFile(
        "target", "Documento real", KIND_DOCUMENT, version="4",
        parents=("some-other-folder",), content=b"# Real\n",
    )
    shortcut = DriveFile("shortcut", "Atalho", KIND_SHORTCUT, parents=(FOLDER,), targetId="target")
    second_shortcut = DriveFile(
        "shortcut-2", "Outro atalho", KIND_SHORTCUT, parents=(FOLDER,), targetId="target"
    )
    drive = FakeDrive([elsewhere, shortcut, second_shortcut])
    source = drive_source(drive, folder_ids=[FOLDER])

    found = list(source.sweep())

    assert [document.external_id for document in found] == ["target"]
    assert found[0].kind == KIND_DOCUMENT
    assert found[0].revision_marker == "4"

    assert "shortcut" not in {document.external_id for document in found}
    assert "shortcut-2" not in {document.external_id for document in found}
    assert len(found) == 1


def test_ac_016(conn: psycopg.Connection):
    """AC-016: A native document is exported in a form that keeps its headings @spec:AC-016"""
    drive = FakeDrive([_document("doc-1", content=word_document(HEADING_LEVELS))])
    source = drive_source(drive, folder_ids=[FOLDER])
    document = next(iter(source.sweep()))
    assert document.kind == KIND_DOCUMENT

    fetched = source.fetch(document)

    assert fetched.media_type == MEDIA_PDF
    exports = [r for r in drive.requests if r.url.path.endswith("/export")]
    assert dict(exports[-1].url.params)["mimeType"] == MEDIA_PDF

    assert fetched.media_type not in UNPAGINATED_MEDIA_TYPES

    assert preflight(
        Candidate(
            document_id=1,
            external_id=document.external_id,
            media_type=fetched.media_type,
            byte_size=len(fetched.content),
        )
    ).accepted is True


def test_ac_017(conn: psycopg.Connection):
    """AC-017: A spreadsheet is exported with every tab @spec:AC-017"""
    workbook = b"PK\x03\x04fake-xlsx-with-three-sheets"
    drive = FakeDrive(
        [DriveFile("sheet-1", "Planilha", KIND_SPREADSHEET, version="2", parents=(FOLDER,), content=workbook)]
    )
    source = drive_source(drive, folder_ids=[FOLDER])
    document = next(iter(source.sweep()))

    fetched = source.fetch(document)

    assert fetched.media_type == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    exports = [r for r in drive.requests if r.url.path.endswith("/export")]
    requested = dict(exports[-1].url.params)["mimeType"]
    assert requested.endswith("spreadsheetml.sheet")
    assert "csv" not in requested
    assert fetched.content.startswith(b"PK")


def test_ac_019(conn: psycopg.Connection):
    """AC-019: Each run leaves an auditable record @spec:AC-019"""
    covered_until = datetime(2026, 9, 11, 3, 0, tzinfo=timezone.utc)
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    commit_page(conn, run_id=run_id, external_ids=[], next_position="t9")
    counters = RunCounters(
        documents_seen=412,
        documents_added=9,
        documents_updated=4,
        documents_skipped=398,
        documents_removed=1,
        documents_errored=1,
        access_lost=4,
        documents_unproven=3,
    )
    finish_run(
        conn,
        run_id,
        status=RUN_STATUS_COMPLETED,
        counters=counters,
        sweep_completed=True,
        window_end=covered_until,
        detail="1 document over the export cap",
    )

    record = read_run(conn, run_id)

    assert record.counters == counters
    assert record.counters.documents_seen == 412
    assert record.counters.documents_skipped == 398
    assert record.counters.documents_removed == 1
    assert record.counters.documents_errored == 1
    assert record.counters.access_lost == 4
    assert record.counters.documents_unproven == 3
    assert record.sweep_completed is True
    assert record.window_end == covered_until
    assert record.status == RUN_STATUS_COMPLETED
    assert record.finished is True
    assert record.finished_at is not None
    assert record.feed_position == "t9"
    assert record.detail == "1 document over the export cap"

    following = read_run(conn, start_delta_run(conn, client=TEST_CLIENT))
    assert following.window_start == covered_until


def test_p_003(conn: psycopg.Connection):
    drive = FakeDrive(
        [DriveFile("gone", "Sumiu", KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True)]
    )
    source = drive_source(drive, folder_ids=[FOLDER])
    _indexed(conn, "gone")

    def _run(kind, status, sweep_completed):
        run_id = start_run(conn, client=TEST_CLIENT, kind=kind)
        finish_run(conn, run_id, status=status, sweep_completed=sweep_completed)
        return read_run(conn, run_id)

    refused = (
        _run(RUN_KIND_SWEEP, RUN_STATUS_FAILED, False),
        _run(RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, False),
        _run(RUN_KIND_INCREMENTAL, RUN_STATUS_COMPLETED, False),
    )
    for run in refused:
        outcome = reconcile(conn, source, run=run, observed=[])
        assert outcome.removed == (), run.status
        assert outcome.refusal == "sweep_did_not_complete"
        assert _state(conn, "gone") == DOCUMENT_STATE_INDEXED

    assert not any(r.url.path.endswith("/gone") for r in drive.requests)

    permitted = reconcile(
        conn, source, run=_run(RUN_KIND_SWEEP, RUN_STATUS_COMPLETED, True), observed=[]
    )
    assert permitted.removed == ("gone",)


def test_p_004(conn: psycopg.Connection):
    corpus = [f"doc-{n}" for n in range(10)]
    _indexed(conn, *corpus)
    mostly_trashed = corpus[:9]
    drive = FakeDrive(
        [
            DriveFile(name, name, KIND_DOCUMENT, version="1", parents=(FOLDER,), trashed=True)
            for name in mostly_trashed
        ]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=mostly_trashed,
    )

    assert set(outcome.removed) == set(mostly_trashed)

    _indexed(conn, "lonely")
    lonely_drive = FakeDrive(
        [DriveFile("lonely", "lonely", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True)]
    )
    lonely_outcome = reconcile(
        conn, drive_source(lonely_drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["lonely"],
    )
    assert lonely_outcome.removed == ()
    assert lonely_outcome.unproven == ("lonely",)
    assert _state(conn, "lonely") == DOCUMENT_STATE_INDEXED


def test_p_005(conn: psycopg.Connection):
    _indexed(conn, "not-found")
    not_found_drive = FakeDrive(
        [DriveFile("not-found", "not-found", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True)]
    )
    not_found = reconcile(
        conn, drive_source(not_found_drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["not-found"],
    )
    assert not_found.removed == ()
    assert not_found.unproven == ("not-found",)
    assert _state(conn, "not-found") == DOCUMENT_STATE_INDEXED

    _indexed(conn, "explained")
    explained_drive = FakeDrive(
        [DriveFile("explained", "explained", KIND_DOCUMENT, version="1", parents=(FOLDER,), missing=True)]
    )
    moved = group_by_document(
        [MoveEvent(external_id="explained", removed_parents=(FOLDER,), occurred_at=None)]
    )
    vetoed = reconcile(
        conn, drive_source(explained_drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["explained"], activity=moved,
    )
    assert vetoed.removed == ()
    assert vetoed.moved == ("explained",)
    assert _state(conn, "explained") == DOCUMENT_STATE_INDEXED


def test_ac_021(conn: psycopg.Connection):
    """AC-021: A baseline sweep comes before any delta @spec:AC-021"""
    with pytest.raises(LedgerError, match="baseline sweep"):
        start_delta_run(conn, client=TEST_CLIENT)

    assert recent_runs(conn) == ()


def test_ac_022(conn: psycopg.Connection):
    """AC-022: The delta window starts where the last successful run ended @spec:AC-022"""
    baseline = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    edge = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)
    finish_run(conn, baseline, status=RUN_STATUS_COMPLETED, sweep_completed=True, window_end=edge)

    failed_delta = start_delta_run(conn, client=TEST_CLIENT)
    assert read_run(conn, failed_delta).window_start == edge
    finish_run(conn, failed_delta, status=RUN_STATUS_FAILED, detail="the change feed stopped")

    later_delta = start_delta_run(conn, client=TEST_CLIENT)

    assert read_run(conn, later_delta).window_start == edge


def test_ac_023(conn: psycopg.Connection):
    """AC-023: Every run is stored and listable, newest first @spec:AC-023"""
    older = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, older, status=RUN_STATUS_COMPLETED, sweep_completed=True, counters=RunCounters(documents_seen=3))
    newer = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    finish_run(conn, newer, status=RUN_STATUS_FAILED, detail="the source timed out")

    runs = list_runs(conn)

    assert [run.id for run in runs] == [newer, older]
    assert runs[1].kind == RUN_KIND_SWEEP
    assert runs[1].status == RUN_STATUS_COMPLETED
    assert runs[1].counters.documents_seen == 3
    assert runs[0].kind == RUN_KIND_INCREMENTAL
    assert runs[0].status == RUN_STATUS_FAILED
    assert runs[0].detail == "the source timed out"

    with pytest.raises(psycopg.errors.RaiseException, match="AC-023"), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute("UPDATE sync_runs SET detail = 'rewritten' WHERE id = %s", (older,))


def test_ac_024(conn: psycopg.Connection):
    """AC-024: Two synchronisations never run at once @spec:AC-024"""
    running = start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)

    with pytest.raises(HistoryError) as excinfo:
        start_exclusive_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)

    assert str(running) in str(excinfo.value)
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM sync_runs")
        assert cursor.fetchone()[0] == 1


def test_ac_025(conn: psycopg.Connection):
    """AC-025: A document says which run first saw it and which saw it last @spec:AC-025"""
    upsert_document(conn, external_id="doc-025")
    first_run = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    mark_document_seen(conn, external_id="doc-025", run_id=first_run)
    finish_run(conn, first_run, status=RUN_STATUS_COMPLETED, sweep_completed=True)

    later_run = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_INCREMENTAL)
    mark_document_seen(conn, external_id="doc-025", run_id=later_run)

    history = document_history(conn, "doc-025")

    assert history.first_seen_run_id == first_run
    assert history.last_seen_run_id == later_run

def test_ac_027(conn: psycopg.Connection):
    """AC-027: The delta covers the same tree the sweep covers @spec:AC-027"""
    inside_sub = "sub-folder"
    outside = "other-folder"
    drive = FakeDrive(
        [
            DriveFile(inside_sub, "Subpasta", KIND_FOLDER, parents=(FOLDER,)),
            DriveFile(outside, "Pasta de outro assunto", KIND_FOLDER, parents=("root",)),
            DriveFile("deep", "Contrato no fundo", KIND_DOCUMENT, version="1", parents=(inside_sub,)),
            DriveFile("alien", "Nada nosso", KIND_DOCUMENT, version="1", parents=(outside,)),
        ],
        changes=[DriveChange(fileId="deep"), DriveChange(fileId="alien")],
    )
    source = drive_source(drive, folder_ids=[FOLDER])

    page = source.changes_since("t0")

    arrived = tuple(document.external_id for document in page.documents)
    assert arrived == ("deep",)
    assert "alien" not in arrived

    before = len(drive.requests)
    source.changes_since("t0")
    lookups = [r for r in drive.requests[before:] if r.url.path.endswith(f"/{inside_sub}")]
    assert lookups == []


def test_ac_028(conn: psycopg.Connection):
    """AC-028: A document carried out inside a folder is explained by that folder @spec:AC-028"""
    carrier = "pasta-que-saiu"
    drive = FakeDrive(
        [DriveFile("carried", "Contrato", KIND_DOCUMENT, version="1", parents=(carrier,), missing=True)]
    )
    _indexed(conn, "carried")
    upsert_document(conn, external_id="carried", metadata={"parents": [carrier]})
    explanations = group_by_document(
        [MoveEvent(external_id=carrier, removed_parents=(FOLDER,), occurred_at=None)]
    )

    outcome = reconcile(
        conn, drive_source(drive, folder_ids=[FOLDER]), run=_completed_sweep(conn),
        reported_removals=["carried"], activity=explanations,
    )

    assert outcome.removed == ()
    assert outcome.unproven == ()
    assert outcome.carried_out == (("carried", carrier),)
    assert outcome.vetoed == ("carried",)
    assert _state(conn, "carried") == DOCUMENT_STATE_INDEXED
    assert carrier in outcome.detail
    assert "carried out by" in outcome.detail


def test_ac_026(conn: psycopg.Connection):
    """AC-026: A delta runs only when the stored position still describes this corpus @spec:AC-026"""
    drive = FakeDrive(start_token="7")
    source = drive_source(drive, folder_ids=[FOLDER])
    identity = "rag@example.iam.gserviceaccount.com"
    resume = DriveResume(source=source, identity=identity, folders=(FOLDER,))

    first = decide_next_run(conn, resume)
    assert first.kind == RUN_KIND_SWEEP
    assert first.reason == REASON_NO_BASELINE

    _completed_sweep(conn)
    assert decide_next_run(conn, resume).reason == REASON_NO_POSITION
    advance_position(conn, position="3")
    assert decide_next_run(conn, resume).reason == REASON_NO_POSITION

    advance_position(conn, position="3", issued_by=identity, scope=FOLDER)
    chosen = decide_next_run(conn, resume)
    assert chosen.kind == RUN_KIND_INCREMENTAL
    assert chosen.is_delta is True
    assert chosen.reason == REASON_POSITION_HOLDS

    other_identity = DriveResume(source=source, identity="someone-else@example.com", folders=(FOLDER,))
    assert decide_next_run(conn, other_identity).reason == REASON_ANOTHER_IDENTITY
    other_tree = DriveResume(source=source, identity=identity, folders=("another-folder",))
    assert decide_next_run(conn, other_tree).reason == REASON_ANOTHER_TREE

    advance_position(conn, position="99", issued_by=identity, scope=FOLDER)
    ahead = decide_next_run(conn, resume)
    assert ahead.kind == RUN_KIND_SWEEP
    assert ahead.reason == REASON_AHEAD_OF_SOURCE
    assert "freeze" in ahead.detail

    advance_position(conn, position="1", issued_by=identity, scope=FOLDER)
    assert decide_next_run(conn, resume).is_delta is True


def test_the_registry_and_the_configuration_literal_agree():
    from src.config import REGISTERED_SOURCE_MODULES

    assert registered_slugs() == tuple(sorted(REGISTERED_SOURCE_MODULES))
    for slug in REGISTERED_SOURCE_MODULES:
        assert source_class(slug) is not None
    with pytest.raises(SourceError):
        source_class("dropbox")
