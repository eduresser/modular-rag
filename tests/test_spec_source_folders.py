
import json
from datetime import datetime, timezone

import httpx
import psycopg
import pytest

from src.sources.base import SourceError, group_by_document
from src.sources.gdrive_activity import ActivityHistory, DeletionEvent, MoveEvent
from src.sources.gdrive_export import KIND_DOCUMENT, KIND_FOLDER
from src.sources.gdrive_resume import REASON_ANOTHER_TREE, DriveResume
from src.store import DOCUMENT_STATE_GONE, DOCUMENT_STATE_INDEXED
from src.sync_reconcile import reconcile
from src.sync_decision import REASON_POSITION_HOLDS, decide_next_run
from src.sync_ledger import (
    RUN_KIND_INCREMENTAL,
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    advance_position,
    read_run,
    finish_run,
    start_run,
)
from tests.conftest import (
    TEST_CLIENT,
    ChunkSpec,
    DocumentSpec,
    DriveChange,
    DriveFile,
    FakeDrive,
    drive_source,
    index_corpus,
)

SINCE = datetime(2026, 9, 1, tzinfo=timezone.utc)

FIRST = "first-folder"
SECOND = "second-folder"


def _doc(file_id: str, *, parent: str) -> DriveFile:
    return DriveFile(file_id, file_id, KIND_DOCUMENT, version="1", parents=(parent,))


def _folder(folder_id: str, *, parent: str | None = None) -> DriveFile:
    return DriveFile(
        folder_id, folder_id, KIND_FOLDER, parents=(parent,) if parent else ()
    )


def test_ac_235():
    """AC-235: Every listed folder is swept @spec:AC-235"""
    drive = FakeDrive(
        [
            _folder(FIRST),
            _folder(SECOND),
            _folder("first-sub", parent=FIRST),
            _folder("second-sub", parent=SECOND),
            _doc("top-of-first", parent=FIRST),
            _doc("deep-in-first", parent="first-sub"),
            _doc("top-of-second", parent=SECOND),
            _doc("deep-in-second", parent="second-sub"),
            DriveFile(
                "shortcut-in-second",
                "Atalho",
                "application/vnd.google-apps.shortcut",
                parents=(SECOND,),
                targetId="deep-in-first",
            ),
        ]
    )
    source = drive_source(drive, folder_ids=[FIRST, SECOND])

    found = [document.external_id for document in source.sweep()]

    assert sorted(found) == [
        "deep-in-first",
        "deep-in-second",
        "top-of-first",
        "top-of-second",
    ]

    assert found.count("deep-in-first") == 1
    assert source.folders == (FIRST, SECOND)


def test_ac_236():
    """AC-236: The delta covers every listed folder, and nothing outside them @spec:AC-236"""
    drive = FakeDrive(
        [
            _folder(FIRST),
            _folder(SECOND),
            _folder("second-sub", parent=SECOND),
            _folder("nobodys-folder", parent="somebody-elses-drive"),
            _doc("ours", parent="second-sub"),
            _doc("theirs", parent="nobodys-folder"),
        ],
        changes=[DriveChange(fileId="ours"), DriveChange(fileId="theirs")],
    )
    source = drive_source(drive, folder_ids=[FIRST, SECOND])

    page = source.changes_since("t0")

    assert tuple(document.external_id for document in page.documents) == ("ours",)


def test_ac_237():
    """AC-237: A folder inside another listed folder changes nothing @spec:AC-237"""
    def corpus() -> FakeDrive:
        return FakeDrive(
            [
                _folder(FIRST),
                _folder("middle", parent=FIRST),
                _folder("inner", parent="middle"),
                _doc("at-the-top", parent=FIRST),
                _doc("in-the-middle", parent="middle"),
                _doc("right-inside", parent="inner"),
            ]
        )

    nested = drive_source(corpus(), folder_ids=[FIRST, "inner", FIRST])
    outermost_only = drive_source(corpus(), folder_ids=[FIRST])

    from_nested = sorted(document.external_id for document in nested.sweep())
    from_outermost = sorted(document.external_id for document in outermost_only.sweep())

    assert from_nested == from_outermost == ["at-the-top", "in-the-middle", "right-inside"]
    assert nested.folders == (FIRST,)

    assert nested.changes_since("t0").documents == ()
    assert nested._folder_within("inner") is True


def test_ac_238():
    """AC-238: A folder the credential cannot read stops the run before anything is decided @spec:AC-238"""
    denied = FakeDrive(
        [
            _folder(FIRST),
            DriveFile(SECOND, SECOND, KIND_FOLDER, denied=True),
            _doc("readable", parent=FIRST),
        ]
    )
    missing = FakeDrive(
        [
            _folder(FIRST),
            DriveFile(SECOND, SECOND, KIND_FOLDER, missing=True),
            _doc("readable", parent=FIRST),
        ]
    )
    trashed = FakeDrive(
        [
            _folder(FIRST),
            DriveFile(SECOND, SECOND, KIND_FOLDER, trashed=True),
            _doc("readable", parent=FIRST),
        ]
    )

    with pytest.raises(SourceError) as refused_access:
        list(drive_source(denied, folder_ids=[FIRST, SECOND]).sweep())
    assert SECOND in str(refused_access.value)
    assert "access is denied" in str(refused_access.value)

    with pytest.raises(SourceError) as gone:
        list(drive_source(missing, folder_ids=[FIRST, SECOND]).sweep())
    assert SECOND in str(gone.value)
    assert "no such item" in str(gone.value)

    with pytest.raises(SourceError) as in_the_bin:
        list(drive_source(trashed, folder_ids=[FIRST, SECOND]).sweep())
    assert SECOND in str(in_the_bin.value)
    assert "trash" in str(in_the_bin.value)

    source = drive_source(denied, folder_ids=[FIRST, SECOND])
    swept = source.sweep()
    with pytest.raises(SourceError):
        next(iter(swept))


def test_ac_239():
    """AC-239: An identifier that does not name a folder stops the run @spec:AC-239"""
    drive = FakeDrive(
        [
            _folder(FIRST),
            _doc("a-contract", parent=FIRST),
            _doc("pasted-by-mistake", parent=FIRST),
        ]
    )
    source = drive_source(drive, folder_ids=[FIRST, "pasted-by-mistake"])

    with pytest.raises(SourceError) as excinfo:
        list(source.sweep())

    message = str(excinfo.value)
    assert "pasted-by-mistake" in message
    assert KIND_DOCUMENT in message

    swept = drive_source(drive, folder_ids=[FIRST, "pasted-by-mistake"]).sweep()
    with pytest.raises(SourceError):
        next(iter(swept))


def _activity_per_folder(histories: dict[str, list[dict]]) -> httpx.Client:

    def handler(request: httpx.Request) -> httpx.Response:
        ancestor = json.loads(request.content)["ancestorName"].removeprefix("items/")
        return httpx.Response(
            200, json={"activities": histories.get(ancestor, [])}, request=request
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _deleted(external_id: str) -> dict:
    return {
        "primaryActionDetail": {"delete": {"type": "PERMANENT_DELETE"}},
        "targets": [{"driveItem": {"name": f"items/{external_id}"}}],
        "timestamp": "2026-09-10T12:00:00.000Z",
    }


def _moved_out(external_id: str, *, from_folder: str) -> dict:
    return {
        "primaryActionDetail": {
            "move": {
                "removedParents": [{"driveItem": {"name": f"items/{from_folder}"}}]
            }
        },
        "targets": [{"driveItem": {"name": f"items/{external_id}"}}],
        "timestamp": "2026-09-10T12:00:00.000Z",
    }


def _indexed(conn: psycopg.Connection, *external_ids: str) -> dict[str, int]:
    return index_corpus(
        conn,
        [
            DocumentSpec(
                identifier,
                f"Contrato {identifier}",
                (ChunkSpec(0, "cláusula", pages=(1, 1)),),
            )
            for identifier in external_ids
        ],
    )


def _state(conn: psycopg.Connection, external_id: str) -> str | None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE external_id = %s", (external_id,))
        row = cursor.fetchone()
    return row[0] if row else None


def _completed_sweep(conn: psycopg.Connection):
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    return read_run(conn, run_id)


def test_ac_241(conn: psycopg.Connection):
    """AC-241: A deletion is looked for in the history of every watched folder @spec:AC-241"""
    _indexed(conn, "gone-from-first", "gone-from-second", "merely-moved")
    history = ActivityHistory(
        folder_ids=(FIRST, SECOND),
        http=_activity_per_folder(
            {
                FIRST: [_deleted("gone-from-first")],
                SECOND: [
                    _deleted("gone-from-second"),
                    _moved_out("merely-moved", from_folder=SECOND),
                ],
            }
        ),
        token=lambda: "fake-token",
    )

    events = group_by_document(history.events_since(SINCE))

    assert isinstance(events["gone-from-first"][0], DeletionEvent)
    assert isinstance(events["gone-from-second"][0], DeletionEvent)
    assert isinstance(events["merely-moved"][0], MoveEvent)

    drive = FakeDrive(
        [
            _folder(FIRST),
            _folder(SECOND),
            DriveFile("gone-from-first", "Contrato", KIND_DOCUMENT, version="1", missing=True),
            DriveFile("gone-from-second", "Contrato", KIND_DOCUMENT, version="1", missing=True),
            DriveFile("merely-moved", "Contrato", KIND_DOCUMENT, version="1", missing=True),
        ]
    )
    outcome = reconcile(
        conn,
        drive_source(drive, folder_ids=[FIRST, SECOND]),
        run=_completed_sweep(conn),
        reported_removals=["gone-from-first", "gone-from-second", "merely-moved"],
        activity=events,
    )

    assert sorted(outcome.removed) == ["gone-from-first", "gone-from-second"]
    assert _state(conn, "gone-from-second") == DOCUMENT_STATE_GONE

    assert "merely-moved" not in outcome.removed
    assert _state(conn, "merely-moved") == DOCUMENT_STATE_INDEXED


def test_ac_240(conn: psycopg.Connection):
    """AC-240: Changing which folders are watched forces a sweep; reordering them does not @spec:AC-240"""
    identity = "rag@example.iam.gserviceaccount.com"
    drive = FakeDrive([_folder(FIRST), _folder(SECOND)], start_token="7")
    source = drive_source(drive, folder_ids=[FIRST, SECOND])
    watching_both = DriveResume(source=source, identity=identity, folders=(FIRST, SECOND))

    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)
    stamped_by, stamped_scope = watching_both.origin
    advance_position(conn, position="3", issued_by=stamped_by, scope=stamped_scope)

    reordered = DriveResume(source=source, identity=identity, folders=(SECOND, FIRST))

    chosen = decide_next_run(conn, reordered)
    assert chosen.kind == RUN_KIND_INCREMENTAL
    assert chosen.reason == REASON_POSITION_HOLDS
    repeated = DriveResume(source=source, identity=identity, folders=(FIRST, SECOND, FIRST))
    assert decide_next_run(conn, repeated).kind == RUN_KIND_INCREMENTAL

    added = DriveResume(source=source, identity=identity, folders=(FIRST, SECOND, "a-third"))
    grown = decide_next_run(conn, added)
    assert grown.kind == RUN_KIND_SWEEP
    assert grown.reason == REASON_ANOTHER_TREE

    removed = DriveResume(source=source, identity=identity, folders=(FIRST,))
    shrunk = decide_next_run(conn, removed)
    assert shrunk.kind == RUN_KIND_SWEEP
    assert shrunk.reason == REASON_ANOTHER_TREE

    assert DriveResume(source=source, identity=identity, folders=(FIRST,)).scope == FIRST


def test_every_unreadable_folder_is_named_in_one_refusal():
    drive = FakeDrive(
        [
            _folder(FIRST),
            DriveFile(SECOND, SECOND, KIND_FOLDER, denied=True),
            DriveFile("never-existed", "never-existed", KIND_FOLDER, missing=True),
            _doc("a-document-not-a-folder", parent=FIRST),
        ]
    )
    source = drive_source(
        drive, folder_ids=[FIRST, SECOND, "a-document-not-a-folder", "never-existed"]
    )

    with pytest.raises(SourceError) as excinfo:
        source.folders

    message = str(excinfo.value)
    assert SECOND in message
    assert "a-document-not-a-folder" in message
    assert "never-existed" in message
    assert FIRST not in message, "a folder that is perfectly readable was named too"
