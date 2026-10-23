
from src.ingest import ingest_documents
from src.ingest_idempotency import claims_for
from src.reindex import ReindexOutcome, ReindexPlan, reindex
from src.reprocess import ACTION_READ, ACTION_RECUT, ACTION_REEMBED, ACTION_TOUCH, stored_state
from src.search import SearchRequest, search_on
from src.search_result import Found
from src.staleness import archive_staleness
from src.worker import sweep
from tests.conftest import (
    TEST_DIMENSION,
    FakeEmbeddings,
    contract_payload,
    direction,
    ingest_config,
    ingest_world,
    pdf_file,
)

CLAUSES = ("Cláusula primeira: do objeto da locação.", "Cláusula segunda: do prazo.")


def _sweep_into(world, conn):
    return ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))


def _usage(conn, operation=None):
    query = "SELECT operation, reason, units, outcome FROM usage_events"
    with conn.cursor() as cursor:
        if operation is None:
            cursor.execute(f"{query} ORDER BY id")
        else:
            cursor.execute(f"{query} WHERE operation = %s ORDER BY id", (operation,))
        return cursor.fetchall()


def _document_id(conn, external_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", (external_id,))
        return cursor.fetchone()[0]


def _contents(conn, document_id):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT content FROM chunks WHERE document_id = %s ORDER BY ordinal", (document_id,)
        )
        return [row[0] for row in cursor.fetchall()]


def test_ac_100(monkeypatch, conn):
    """AC-100: Unchanged content and unchanged configuration cost nothing @spec:AC-100"""
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep_into(first, conn)
    before = stored_state(conn, _document_id(conn, 'doc-1'))
    assert before is not None and before.piece_count == len(CLAUSES)
    spent = _usage(conn)

    second = ingest_world(monkeypatch, files=files, outputs=outputs, storage=first.storage)
    run = _sweep_into(second, conn)

    assert second.reads == 0
    assert second.embedded == 0
    assert _usage(conn) == spent
    assert [result.action for result in run.results] == [ACTION_TOUCH]
    assert run.results[0].reason is None

    after = stored_state(conn, _document_id(conn, 'doc-1'))
    assert after.last_seen_at > before.last_seen_at
    assert after.indexed_at == before.indexed_at
    assert after.content_digest == before.content_digest
    assert after.fingerprint == before.fingerprint
    assert after.piece_count == before.piece_count
    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)


def test_ac_101(monkeypatch, conn):
    """AC-101: Changed content reprocesses from the source @spec:AC-101"""
    original = contract_payload(clauses=CLAUSES)
    first = ingest_world(
        monkeypatch, files=[pdf_file("doc-1", content=b"%PDF original")],
        outputs={"doc-1": original},
    )
    _sweep_into(first, conn)
    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)

    revised = ("Cláusula primeira: do objeto, alterada.", "Cláusula segunda: do prazo.")
    second = ingest_world(
        monkeypatch, files=[pdf_file("doc-1", content=b"%PDF revised")],
        outputs={"doc-1": contract_payload(clauses=revised)}, storage=first.storage,
    )

    run = _sweep_into(second, conn)

    assert [result.action for result in run.results] == [ACTION_READ]
    assert second.reads == 1
    assert second.embedded == len(revised)
    assert [row[0] for row in _usage(conn)].count("read") == 2

    assert _contents(conn, _document_id(conn, 'doc-1')) == list(revised)
    assert stored_state(conn, _document_id(conn, 'doc-1')).piece_count == len(revised)


def test_ac_102(monkeypatch, conn):
    """AC-102: A changed cutting configuration reprocesses without reading again @spec:AC-102"""
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep_into(first, conn)
    reads_before = len(_usage(conn, "read"))

    second = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage,
        RAG_CHUNK_SIZE="40",
    )

    run = _sweep_into(second, conn)

    assert [result.action for result in run.results] == [ACTION_RECUT]
    assert second.reads == 0
    assert run.results[0].pieces > 0
    assert [row[0] for row in _usage(conn)].count("chunk") == 1

    assert len(_usage(conn, "read")) == reads_before


def test_ac_103(monkeypatch, conn):
    """AC-103: A changed embedding model re-embeds without reading or cutting again @spec:AC-103"""
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep_into(first, conn)
    stored = _contents(conn, _document_id(conn, 'doc-1'))
    spent = [row[0] for row in _usage(conn)]

    successor = "google/gemini-embedding-001"
    provider = FakeEmbeddings(model=successor, dimension=TEST_DIMENSION)
    second = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage,
        embeddings=provider, RAG_EMBED_MODEL=successor,
    )
    artifact_reads_before = len(first.storage.requests)

    run = _sweep_into(second, conn)

    assert [result.action for result in run.results] == [ACTION_REEMBED]
    assert second.embedded == len(CLAUSES)
    assert provider.bodies[0]["model"] == successor
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT embed_model, count(*) FROM chunks GROUP BY embed_model ORDER BY embed_model"
        )
        assert dict(cursor.fetchall()) == {successor: len(CLAUSES), "qwen/qwen3-embedding-8b": len(CLAUSES)}
        cursor.execute(
            "SELECT content FROM chunks WHERE embed_model = %s ORDER BY ordinal", (successor,)
        )
        assert [row[0] for row in cursor.fetchall()] == stored

    assert second.reads == 0
    assert len(first.storage.requests) == artifact_reads_before
    assert [row[0] for row in _usage(conn)] == spent + ["embed"]


def test_ac_104(monkeypatch, conn):
    """AC-104: A document's pieces and its state land together @spec:AC-104"""
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    world = ingest_world(monkeypatch, files=files, outputs=outputs)

    def crash(*args, **kwargs):
        raise RuntimeError("the process died between the two writes")

    monkeypatch.setattr("src.ingest.record_state", crash)
    run = _sweep_into(world, conn)

    assert run.errored == 1
    assert run.failures.failures[0].stage == "store"
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT count(*) FROM document_ingest")
        assert cursor.fetchone()[0] == 0

    monkeypatch.undo()
    again = ingest_world(monkeypatch, files=files, outputs=outputs, storage=world.storage)
    run = _sweep_into(again, conn)

    assert run.errored == 0
    assert len(_contents(conn, _document_id(conn, 'doc-1'))) == len(CLAUSES)
    assert stored_state(conn, _document_id(conn, 'doc-1')).piece_count == len(CLAUSES)


def test_ac_105(monkeypatch, conn):
    """AC-105: The same document processed twice produces one result @spec:AC-105"""
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )
    document = next(iter(world.pipeline.source.sweep()))

    run = ingest_documents(conn, world.pipeline, [document, document])

    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)
    assert stored_state(conn, _document_id(conn, 'doc-1')).piece_count == len(CLAUSES)

    assert len(run.duplicates) == 1
    assert len(run.indexed) == 1
    claims = claims_for(conn, "doc-1")
    assert len(claims) == 1
    assert claims[0].deliveries == 2
    assert claims[0].state == "done"

    assert world.reads == 1
    assert world.embedded == len(CLAUSES)
    assert [row[0] for row in _usage(conn)] == ["read", "embed"]


def test_ac_106(monkeypatch, conn):
    """AC-106: A failure marks the document and does not stop the run @spec:AC-106"""
    unreadable = pdf_file("doc-2", content=b"nota interna", mimeType="text/plain", name="Nota")
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato"), unreadable],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )

    outcome = sweep(conn, world.pipeline)

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.external_id, r.state, r.failure_reason
              FROM document_reads r JOIN documents d ON d.id = r.document_id
             ORDER BY d.external_id
            """
        )
        reads = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    assert reads["doc-2"][0] == "failed"
    assert "text/plain" in reads["doc-2"][1]
    assert reads["doc-1"][0] == "read"
    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT status, documents_seen, documents_added, documents_errored, sweep_completed "
            "FROM sync_runs WHERE id = %s",
            (outcome.run_id,),
        )
        status, seen, added, errored, completed = cursor.fetchone()
    assert (status, seen, added, errored, completed) == ("completed", 2, 1, 1, True)

    claim = claims_for(conn, "doc-2")[0]
    assert (claim.state, claim.failure_kind) == ("failed", "permanent")


def test_ac_107(monkeypatch, conn):
    """AC-107: The service reports whether the archive is stale, and why @spec:AC-107"""
    files = [pdf_file("doc-1", content=b"%PDF um"), pdf_file("doc-2", content=b"%PDF dois")]
    outputs = {name: contract_payload(clauses=CLAUSES) for name in ("doc-1", "doc-2")}
    world = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep_into(world, conn)

    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="400", RAG_CHUNK_INCLUDE_HEADINGS="false")
    report = archive_staleness(conn, moved)

    assert report.current is False
    assert report.affected == 2
    assert {change.setting: (change.recorded, change.configured) for change in report.changes} == {
        "chunk.size": (800, 400),
        "chunk.include_headings": (True, False),
    }
    assert report.stages == ("cut",)
    assert "chunk.size" in report.summary()

    unchanged = archive_staleness(conn, ingest_config(monkeypatch))
    assert unchanged.current is True
    assert unchanged.changes == ()
    assert "current" in unchanged.summary()


def test_ac_108(monkeypatch, conn):
    """AC-108: Nothing is reprocessed automatically @spec:AC-108"""
    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=CLAUSES)},
    )
    _sweep_into(world, conn)
    indexed = stored_state(conn, _document_id(conn, 'doc-1'))
    spent = _usage(conn)
    reads_before = world.reads

    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="400")
    report = archive_staleness(conn, moved)
    assert report.current is False
    outcome = search_on(
        conn,
        SearchRequest(question="prazo da locação", query_vector=direction(0.0), result_count=10),
        defaults=moved.search,
    )

    assert world.reads == reads_before
    assert world.embedded == len(CLAUSES)
    assert _usage(conn) == spent
    assert stored_state(conn, _document_id(conn, 'doc-1')) == indexed
    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)

    assert isinstance(outcome, Found)
    assert {hit.provenance.external_id for hit in outcome.hits} == {"doc-1"}


def test_ac_109(monkeypatch, conn, price_book):
    """AC-109: Reprocessing is an explicit operation that reports its cost first @spec:AC-109"""
    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=CLAUSES)}
    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    _sweep_into(first, conn)
    spent = _usage(conn)

    stale = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage,
        RAG_DOCAI_PROCESSOR_VERSION="pretrained-layout-parser-v1.5",
    )

    plan = reindex(conn, stale.pipeline, price_book)

    assert isinstance(plan, ReindexPlan)
    assert plan.started is False
    assert plan.estimate.documents == 1
    assert plan.estimate.for_action(ACTION_READ).documents == 1
    assert plan.estimate.cost > 0
    assert plan.staleness.current is False
    assert stale.reads == 0
    assert stale.embedded == 0
    assert _usage(conn) == spent
    assert _contents(conn, _document_id(conn, 'doc-1')) == list(CLAUSES)

    outcome = reindex(conn, stale.pipeline, price_book, confirm=True)

    assert isinstance(outcome, ReindexOutcome)
    assert outcome.started is True
    assert len(outcome.run.indexed) == 1
    assert stale.reads > 0
    assert stale.embedded == 0
    assert [row[0] for row in _usage(conn)] != [row[0] for row in spent]
    assert stored_state(conn, _document_id(conn, 'doc-1')).fingerprint != first.pipeline.fingerprints.whole
