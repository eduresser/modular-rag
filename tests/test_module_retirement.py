import psycopg
import pytest

from src.reprocess import foreign_documents, retire_foreign_pieces
from src.sync_ledger import read_run
from src.sync_queue import request_payload
from src.worker import incremental, sweep
from tests.conftest import (
    ARTIFACT_BUCKET,
    INGEST_DEPLOYMENT,
    LOCAL_DEPLOYMENT,
    FakeObjectStorage,
    ingest_world,
    pdf_bytes,
    pdf_file,
    reader_chunk,
    reader_payload,
    text_block,
)

READABLE = ["Cláusula primeira.", "Cláusula segunda."]


def _files():
    return [
        pdf_file("doc-legivel", name="Contrato", content=pdf_bytes(READABLE)),
        pdf_file("doc-escaneado", name="Digitalizado", content=pdf_bytes(["", ""])),
    ]


def _paid_outputs():
    return {
        external: reader_payload(
            [text_block("b1", f"LIDO PELO DOCAI — {external}", pages=(1, 1), kind="heading-1")],
            [reader_chunk("c1", f"LIDO PELO DOCAI — {external}", ["b1"])],
        )
        for external in ("doc-legivel", "doc-escaneado")
    }


def _swept_under_both(monkeypatch, conn):
    storage = FakeObjectStorage(ARTIFACT_BUCKET)
    files = _files()
    pago = ingest_world(monkeypatch, deployment=INGEST_DEPLOYMENT, files=files,
                        outputs=_paid_outputs(), storage=storage)
    sweep(conn, pago.pipeline)
    gratis = ingest_world(monkeypatch, deployment=LOCAL_DEPLOYMENT, files=files, storage=storage)
    return gratis


def test_the_module_is_recorded_rather_than_guessed_at(monkeypatch, conn: psycopg.Connection):
    storage = FakeObjectStorage(ARTIFACT_BUCKET)
    world = ingest_world(monkeypatch, deployment=INGEST_DEPLOYMENT, files=_files(),
                         outputs=_paid_outputs(), storage=storage)
    sweep(conn, world.pipeline)

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT process_module, reader_version FROM document_ingest"
        )
        assert cursor.fetchall() == [("docai", "pretrained-layout-parser-v1.0-2024-06-03")]

    assert foreign_documents(conn, "docai") == ()
    assert [external for _id, external in foreign_documents(conn, "pdftext")] == [
        "doc-escaneado", "doc-legivel",
    ]


def test_a_row_written_before_the_archive_recorded_the_module_counts_as_foreign(
    monkeypatch, conn: psycopg.Connection
):
    storage = FakeObjectStorage(ARTIFACT_BUCKET)
    world = ingest_world(monkeypatch, deployment=INGEST_DEPLOYMENT, files=_files(),
                         outputs=_paid_outputs(), storage=storage)
    sweep(conn, world.pipeline)
    with conn.cursor() as cursor:
        cursor.execute("UPDATE document_ingest SET process_module = NULL")

    assert len(foreign_documents(conn, "docai")) == 2


def test_the_count_reaches_the_operator_on_the_runs_own_record(
    monkeypatch, conn: psycopg.Connection
):
    gratis = _swept_under_both(monkeypatch, conn)

    outcome = sweep(conn, gratis.pipeline)
    record = read_run(conn, outcome.run_id)

    assert record.counters.documents_retired == 1
    assert record.counters.documents_errored == 1


def test_the_count_reaches_the_sync_response(monkeypatch, conn: psycopg.Connection):
    from src.sync_queue import attach_run, enqueue, read_request

    gratis = _swept_under_both(monkeypatch, conn)
    outcome = sweep(conn, gratis.pipeline)
    request = enqueue(conn)
    attach_run(conn, request.id, outcome.run_id)

    payload = request_payload(conn, read_request(conn, request.id))

    assert payload["counters"]["documents_retired"] == 1


def test_an_incremental_run_reports_what_it_found_and_retires_nothing(
    monkeypatch, conn: psycopg.Connection
):
    gratis = _swept_under_both(monkeypatch, conn)

    outcome = incremental(conn, gratis.pipeline)

    assert outcome.counters.documents_retired == 0
    assert foreign_documents(conn, "pdftext")


def test_retiring_takes_the_ingest_row_with_the_pieces(monkeypatch, conn: psycopg.Connection):
    gratis = _swept_under_both(monkeypatch, conn)
    sweep(conn, gratis.pipeline)

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT d.external_id FROM document_ingest i "
            "  JOIN documents d ON d.id = i.document_id ORDER BY d.external_id"
        )
        assert [row[0] for row in cursor.fetchall()] == ["doc-legivel"]
        cursor.execute("SELECT count(*) FROM documents")
        assert cursor.fetchone()[0] == 2


def test_nothing_is_retired_when_every_document_is_the_configured_modules(
    monkeypatch, conn: psycopg.Connection
):
    storage = FakeObjectStorage(ARTIFACT_BUCKET)
    world = ingest_world(monkeypatch, deployment=INGEST_DEPLOYMENT, files=_files(),
                         outputs=_paid_outputs(), storage=storage)
    sweep(conn, world.pipeline)

    outcome = retire_foreign_pieces(conn, module="docai", may_remove=True)

    assert outcome.clean and outcome.considered == 0 and outcome.retired == ()
    assert "produced by 'docai'" in outcome.detail


def test_a_transient_failure_spares_the_document_the_next_run_will_retry(
    monkeypatch, conn: psycopg.Connection
):
    from src.ingest_errors import FAILURE_PERMANENT, FAILURE_TRANSIENT
    from src.ingest_idempotency import claim, fail

    gratis = _swept_under_both(monkeypatch, conn)
    sweep(conn, gratis.pipeline)

    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = 'doc-legivel'")
        legivel = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE document_ingest SET process_module = 'docai' WHERE document_id = %s",
            (legivel,),
        )
    taken = claim(conn, external_id="doc-legivel", content_digest="d",
                  fingerprint="f", document_id=legivel)
    fail(conn, taken.claim.id, reason="a blip at the write",
         kind=FAILURE_TRANSIENT, document_id=legivel)

    outcome = retire_foreign_pieces(conn, module="pdftext", may_remove=True)

    assert outcome.clean
    assert outcome.spared == ("doc-legivel",)
    assert outcome.retired == ()
    assert "the switch is not finished for them" in outcome.detail
    assert _indexed(conn)["doc-legivel"] == 2

    taken = claim(conn, external_id="doc-legivel", content_digest="d2",
                  fingerprint="f", document_id=legivel)
    fail(conn, taken.claim.id, reason="no text layer",
         kind=FAILURE_PERMANENT, document_id=legivel)
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM ingest_claims WHERE failure_kind = %s", (FAILURE_TRANSIENT,))

    settled = retire_foreign_pieces(conn, module="pdftext", may_remove=True)

    assert settled.spared == ()
    assert settled.refusal == "module_produced_nothing"
    assert settled.retired == ()
    assert _indexed(conn)["doc-legivel"] == 2


def _indexed(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT d.external_id, count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
            " GROUP BY d.external_id"
        )
        return {external: int(count) for external, count in cursor.fetchall()}
