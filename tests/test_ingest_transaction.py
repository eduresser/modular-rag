from dataclasses import replace

import psycopg
import pytest

from src.db import create_schema
from src.embedding.stamp import stamp_from_config
from src.fingerprint import stage_fingerprints
from src.ingest import DocumentAIReader, commit_document, ingest_documents, pages_in
from src.processing.docai import chunks_from_document
from src.reprocess import stored_state
from src.store import Piece, upsert_document
from src.store_versions import active_version, current_version
from tests.conftest import (
    DOCAI_VERSION,
    TEST_DIMENSION,
    contract_payload,
    direction,
    ingest_config,
    ingest_world,
    pdf_file,
    reader_chunk,
    reader_payload,
    text_block,
)

CLAUSES = ("Cláusula primeira: do objeto.", "Cláusula segunda: do prazo.")


def _pieces(count=2):
    return [
        Piece(
            ordinal=ordinal,
            content=f"Cláusula {ordinal}",
            embedding=direction(ordinal * 10.0),
            page_start=1,
            page_end=1,
        )
        for ordinal in range(count)
    ]


def _sweep(world, conn, **kwargs):
    return ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()), **kwargs)


def test_the_pieces_and_the_state_land_together(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")

    written = commit_document(
        conn,
        document_id=document_id,
        stamp=stamp_from_config(config),
        pieces=_pieces(),
        content_digest="digest-1",
        fingerprints=stage_fingerprints(config),
        reader_version=DOCAI_VERSION,
        cut_version="1",
        revision_marker="marker-1",
    )

    assert written == 2
    state = stored_state(conn, document_id)
    assert state.piece_count == 2
    assert state.content_digest == "digest-1"
    assert state.revision_marker == "marker-1"


def test_a_failure_at_the_second_write_undoes_the_first(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")

    def crash(*args, **kwargs):
        raise RuntimeError("the process died between the two writes")

    monkeypatch.setattr("src.ingest.record_state", crash)
    with pytest.raises(RuntimeError):
        commit_document(
            conn,
            document_id=document_id,
            stamp=stamp_from_config(config),
            pieces=_pieces(),
            content_digest="digest-1",
            fingerprints=stage_fingerprints(config),
            reader_version=DOCAI_VERSION,
            cut_version="1",
        )

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        assert cursor.fetchone()[0] == 0
    assert stored_state(conn, document_id) is None


def test_a_failure_at_the_first_write_leaves_no_state_either(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")

    with pytest.raises(Exception):
        commit_document(
            conn,
            document_id=document_id,
            stamp=stamp_from_config(config),
            pieces=[_pieces(1)[0], _pieces(1)[0]],
            content_digest="digest-1",
            fingerprints=stage_fingerprints(config),
            reader_version=DOCAI_VERSION,
            cut_version="1",
        )

    assert stored_state(conn, document_id) is None


def test_committing_again_replaces_the_pieces_rather_than_adding_to_them(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")
    stamp = stamp_from_config(config)
    commit_document(
        conn, document_id=document_id, stamp=stamp, pieces=_pieces(3),
        content_digest="digest-1", fingerprints=stage_fingerprints(config),
        reader_version=DOCAI_VERSION, cut_version="1",
    )

    written = commit_document(
        conn, document_id=document_id, stamp=stamp, pieces=_pieces(1),
        content_digest="digest-2", fingerprints=stage_fingerprints(config),
        reader_version=DOCAI_VERSION, cut_version="1",
    )

    assert written == 1
    assert stored_state(conn, document_id).piece_count == 1
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        assert cursor.fetchone()[0] == 1


def test_a_first_run_ends_with_a_generation_being_served(monkeypatch, conn):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )
    assert current_version(conn) is None

    _sweep(world, conn)

    assert active_version(conn) == stamp_from_config(world.config)


def test_a_later_run_leaves_the_pointer_where_it_is(monkeypatch, conn):
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep(first, conn)
    serving = active_version(conn)

    successor = "google/gemini-embedding-001"
    from tests.conftest import FakeEmbeddings

    second = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage,
        embeddings=FakeEmbeddings(model=successor, dimension=TEST_DIMENSION),
        RAG_EMBED_MODEL=successor,
    )
    _sweep(second, conn)

    assert active_version(conn) == serving


def test_a_document_the_reader_returns_nothing_for_is_not_indexed_empty(monkeypatch, conn):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": reader_payload([text_block("h1", "Contrato", kind="heading-1")], [])},
    )

    run = _sweep(world, conn)

    assert run.errored == 1
    assert run.failures.failures[0].stage == "cut"
    assert run.failures.failures[0].kind == "permanent"
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks")
        assert cursor.fetchone()[0] == 0


def test_the_run_reports_each_document_once_and_in_the_operator_s_terms(monkeypatch, conn):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF um"), pdf_file("doc-2", content=b"%PDF dois")],
        outputs={name: contract_payload(clauses=CLAUSES) for name in ("doc-1", "doc-2")},
    )

    run = _sweep(world, conn)

    assert {result.external_id for result in run.results} == {"doc-1", "doc-2"}
    assert len(run.indexed) == 2
    assert run.touched == ()
    assert run.duplicates == ()
    assert run.errored == 0
    assert run.result_for("doc-1").pieces == len(CLAUSES)
    assert run.result_for("doc-404") is None


def test_the_reader_step_keeps_the_artifact_before_anything_is_derived(monkeypatch, conn):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )

    _sweep(world, conn)

    with conn.cursor() as cursor:
        cursor.execute("SELECT parsed_uri FROM documents WHERE external_id = 'doc-1'")
        (uri,) = cursor.fetchone()
    assert uri.endswith(f"parsed/{DOCAI_VERSION}/doc-1.json")
    assert world.storage.name_of(uri) in world.storage.objects


def test_the_pages_a_read_is_billed_in_come_from_the_blocks_it_produced():
    payload = reader_payload(
        [
            text_block("b1", "primeira", pages=(1, 2)),
            text_block("b2", "segunda", pages=(3, 4)),
        ],
        [],
    )

    assert pages_in(payload) == 4
    assert pages_in(reader_payload([], [])) == 0


def test_the_reader_step_names_the_version_it_stamps(monkeypatch, conn):
    world = ingest_world(monkeypatch)

    assert isinstance(world.pipeline.reader, DocumentAIReader)
    assert world.pipeline.reader.version == DOCAI_VERSION
    assert world.pipeline.reader.read(conn, []).payloads == {}


def test_chunks_carry_the_fingerprint_and_the_reader_version_they_were_made_under(
    monkeypatch, conn
):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )

    _sweep(world, conn)

    with conn.cursor() as cursor:
        cursor.execute("SELECT metadata FROM chunks ORDER BY ordinal LIMIT 1")
        (metadata,) = cursor.fetchone()
    assert metadata["reader_version"] == DOCAI_VERSION
    assert metadata["fingerprint"] == stage_fingerprints(world.config).whole


def _unpaged_block(block_id, text):
    return {"blockId": str(block_id), "textBlock": {"text": text, "type": "paragraph"}}


def test_a_document_that_cannot_be_cited_fails_without_taking_its_batch(monkeypatch, conn):
    title = "CONTRATO DE PRESTAÇÃO DE SERVIÇOS"
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF um"), pdf_file("doc-2", content=b"%PDF dois")],
        outputs={
            "doc-1": reader_payload([_unpaged_block("1", title)], [reader_chunk("c1", title, ["1"])]),
            "doc-2": contract_payload(clauses=CLAUSES),
        },
    )

    run = _sweep(world, conn)

    assert run.errored == 1
    failure = run.failures.failures[0]
    assert (failure.external_id, failure.stage, failure.kind) == ("doc-1", "cut", "permanent")
    assert {result.external_id for result in run.indexed} == {"doc-2"}
    assert run.result_for("doc-2").pieces == len(CLAUSES)
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks")
        assert cursor.fetchone()[0] == len(CLAUSES)


def test_a_document_is_checkpointed_as_soon_as_it_is_finished(monkeypatch, conn):
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF um"), pdf_file("doc-2", content=b"%PDF dois")],
        outputs={
            "doc-1": contract_payload(clauses=CLAUSES),
            "doc-2": contract_payload(clauses=CLAUSES),
        },
    )
    at_each_checkpoint = []

    def observe():
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM document_ingest")
            (finished,) = cursor.fetchone()
        at_each_checkpoint.append(finished)

    ingest_documents(
        conn,
        replace(world.pipeline, checkpoint=observe),
        list(world.pipeline.source.sweep()),
    )

    assert at_each_checkpoint[0] == 0
    assert 1 in at_each_checkpoint
    assert at_each_checkpoint[-1] == 2


def test_finished_work_survives_the_process_that_did_it(monkeypatch, postgres_dsn):
    with psycopg.connect(postgres_dsn) as setup:
        with setup.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS checkpointed")
            cursor.execute("SET search_path TO checkpointed, public")
        create_schema(setup, dimension=TEST_DIMENSION)
        setup.commit()

    try:
        with psycopg.connect(postgres_dsn) as working:
            with working.cursor() as cursor:
                cursor.execute("SET search_path TO checkpointed, public")
            world = ingest_world(
                monkeypatch,
                files=[pdf_file("doc-1", content=b"%PDF um")],
                outputs={"doc-1": contract_payload(clauses=CLAUSES)},
            )
            ingest_documents(
                working,
                replace(world.pipeline, checkpoint=working.commit),
                list(world.pipeline.source.sweep()),
            )
            working.rollback()

        with psycopg.connect(postgres_dsn) as reader:
            with reader.cursor() as cursor:
                cursor.execute("SET search_path TO checkpointed, public")
                cursor.execute("SELECT count(*) FROM document_ingest")
                (finished,) = cursor.fetchone()
                cursor.execute("SELECT count(*) FROM chunks")
                (pieces,) = cursor.fetchone()
                cursor.execute("SELECT state FROM document_reads")
                states = [row[0] for row in cursor.fetchall()]
        assert finished == 1
        assert pieces > 0
        assert states == ["read"]
    finally:
        with psycopg.connect(postgres_dsn) as cleanup:
            with cleanup.cursor() as cursor:
                cursor.execute("DROP SCHEMA IF EXISTS checkpointed CASCADE")
            cleanup.commit()
