
import datetime
import json

import psycopg
import pytest

from src.models import Chunk, VersionStamps
from src.prices import OPERATION_READ, OPERATION_CHUNK, PriceBook
from src.processing.artifact import (
    READ_STATE_FAILED,
    READ_STATE_READ,
    ChunksBeforeArtifact,
    load_parsed_artifact,
    parsed_artifact,
    read_record,
    require_parsed_artifact,
    store_parsed_artifact,
)
from src.processing.cost import record_recut
from src.processing.docai import (
    MAX_CONCURRENT_BATCHES,
    ChunkingOptions,
    InputDocument,
    ReaderError,
    plan_batches,
)
from src.processing.docai import chunks_from_document as docai_cut
from src.processing.preflight import (
    BATCH_PAGE_LIMIT,
    Candidate,
    record_refusals,
    screen,
)
from src.sources.gdrive_export import KIND_DOCUMENT, export_plan
from src.processing.recut import recut_document
from src.processing.docai import chunks_from_document
from src.search_result import Provenance
from src.store import upsert_document
from src.usage import REASON_CONFIGURATION_CHANGED, usage_between
from tests.conftest import (
    ARTIFACT_BUCKET,
    DOCAI_PROCESSOR,
    DOCAI_VERSION,
    TEST_CLIENT,
    FakeDocumentAI,
    FakeObjectStorage,
    list_block,
    reader_chunk,
    reader_payload,
    table_block,
    text_block,
)

EXTERNAL_ID = "1AbCdEf-drive-id"
STAMPS = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-processing")

CLAUSE_ITEMS = (
    "pagamento do aluguel até o quinto dia útil",
    "conservação do imóvel durante a vigência",
    "devolução das chaves ao término do prazo",
)
TABLE_CAPTION = "Tabela de parcelas do aluguel reajustado pelo IGP-M"


def _contract() -> dict:
    blocks = [
        text_block("b0", "Instrumento particular firmado entre as partes abaixo.", pages=(1, 1)),
        text_block("b1", "CONTRATO DE LOCAÇÃO COMERCIAL", pages=(1, 1), kind="heading-1"),
        text_block("b2", "Cláusula Primeira — Do Objeto", pages=(1, 1), kind="heading-2"),
        text_block("b3", "O objeto é o imóvel situado na Rua das Laranjeiras, 120.", pages=(1, 1)),
        text_block(
            "b3a",
            "A locação destina-se exclusivamente ao uso comercial, vedada a sublocação "
            "total ou parcial sem anuência expressa do locador.",
            pages=(1, 1),
        ),
        text_block(
            "b3b",
            "O prazo de vigência é de sessenta meses, contados da assinatura.",
            pages=(1, 2),
        ),
        text_block("b4", "Das obrigações do locatário", pages=(2, 2), kind="heading-3"),
        list_block("b5", list(CLAUSE_ITEMS), pages=(2, 2)),
        text_block("b6", "Cláusula Segunda — Do Valor", pages=(2, 2), kind="heading-2"),
        table_block(
            "b7",
            ["Parcela", "Vencimento", "Valor"],
            [["1", "05/10/2026", "R$ 12.400,00"], ["2", "05/11/2026", "R$ 12.400,00"]],
            pages=(2, 3),
            annotation=TABLE_CAPTION,
        ),
    ]
    chunks = [
        reader_chunk("c0", "Instrumento particular firmado entre as partes abaixo.", ["b0"]),
        reader_chunk(
            "c1",
            "CONTRATO DE LOCAÇÃO COMERCIAL\nCláusula Primeira — Do Objeto\n"
            "O objeto é o imóvel situado na Rua das Laranjeiras, 120.",
            ["b1", "b2", "b3", "b3a", "b3b"],
        ),
        reader_chunk(
            "c2",
            "Das obrigações do locatário\n" + "\n".join(f"{n}. {item}" for n, item in enumerate(CLAUSE_ITEMS, 1)),
            ["b4", "b5"],
        ),
        reader_chunk(
            "c3",
            "Cláusula Segunda — Do Valor\n"
            "| Parcela | Vencimento | Valor |\n| --- | --- | --- |\n"
            "| 1 | 05/10/2026 | R$ 12.400,00 |\n| 2 | 05/11/2026 | R$ 12.400,00 |",
            ["b6", "b7"],
        ),
    ]
    return reader_payload(blocks, chunks)


def _document(conn: psycopg.Connection, external_id: str = EXTERNAL_ID) -> int:
    return upsert_document(conn, external_id=external_id, title="Contrato de locação")


def _stored(conn: psycopg.Connection, storage: FakeObjectStorage, *, payload=None, pages: int = 3):
    document_id = _document(conn)
    artifact = store_parsed_artifact(
        conn,
        storage.store(),
        document_id=document_id,
        external_id=EXTERNAL_ID,
        payload=payload if payload is not None else _contract(),
        reader=DOCAI_PROCESSOR,
        reader_version=DOCAI_VERSION,
        page_count=pages,
    )
    return document_id, artifact


def _chunk_count(conn: psycopg.Connection, document_id: int) -> int:
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        return int(cursor.fetchone()[0])


def test_ac_030():
    """AC-030: Each piece carries the document, the page and the heading path @spec:AC-030"""
    mapped = chunks_from_document(_contract(), external_id=EXTERNAL_ID, stamps=STAMPS)

    assert len(mapped.chunks) == 4
    for chunk in mapped.chunks:
        assert chunk.document_external_id == EXTERNAL_ID
        assert chunk.pages is not None, chunk.ordinal
        Provenance(
            document_id=1,
            external_id=chunk.document_external_id,
            title="Contrato de locação",
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            heading_path=chunk.heading_path,
        )

    by_ordinal = {chunk.ordinal: chunk for chunk in mapped.chunks}
    assert by_ordinal[1].heading_path == ("CONTRATO DE LOCAÇÃO COMERCIAL", "Cláusula Primeira — Do Objeto")
    assert by_ordinal[2].heading_path == (
        "CONTRATO DE LOCAÇÃO COMERCIAL",
        "Cláusula Primeira — Do Objeto",
        "Das obrigações do locatário",
    )
    assert by_ordinal[1].pages == (1, 2)
    assert by_ordinal[2].pages == (2, 2)
    assert by_ordinal[3].pages == (2, 3)

    opening = by_ordinal[0]
    assert opening.heading_path == ()
    assert opening.document_external_id == EXTERNAL_ID
    assert opening.pages == (1, 1)

    assert by_ordinal[2].source_block_ids == ("b4", "b5")


def test_ac_031():
    """AC-031: Lists survive @spec:AC-031"""
    payload = _contract()
    payload["document"]["documentLayout"]["blocks"].append(
        list_block("b8", ["primeira testemunha", "segunda testemunha"], pages=(3, 3), ordered=False)
    )
    payload["document"]["chunkedDocument"]["chunks"].append(
        reader_chunk("c4", "- primeira testemunha\n- segunda testemunha", ["b8"], pages=(3, 3))
    )

    mapped = chunks_from_document(payload, external_id=EXTERNAL_ID, stamps=STAMPS)

    everything = "\n".join(chunk.content for chunk in mapped.chunks)
    for item in CLAUSE_ITEMS:
        assert item in everything, item
    assert "primeira testemunha" in everything
    assert "segunda testemunha" in everything

    assert len(mapped.chunks) == len(payload["document"]["chunkedDocument"]["chunks"])
    assert [chunk.ordinal for chunk in mapped.chunks] == list(range(len(mapped.chunks)))

    entry = mapped.index["b5-1"]
    assert entry.page_start == 2
    assert entry.heading_path[-1] == "Das obrigações do locatário"

    orphaned = reader_payload([], [reader_chunk("c9", "cláusula órfã", ["nao-existe"])])
    with pytest.raises(Exception) as excinfo:
        chunks_from_document(orphaned, external_id=EXTERNAL_ID, stamps=STAMPS)
    assert "quietly smaller" in str(excinfo.value)


def test_ac_032():
    """AC-032: Tables arrive as content, not as a hole @spec:AC-032"""
    mapped = chunks_from_document(_contract(), external_id=EXTERNAL_ID, stamps=STAMPS)

    table_piece = next(chunk for chunk in mapped.chunks if "12.400,00" in chunk.content)
    for value in ("Parcela", "Vencimento", "05/10/2026", "05/11/2026", "R$ 12.400,00"):
        assert value in table_piece.content, value
    assert TABLE_CAPTION in table_piece.content

    assert table_piece.content.count(TABLE_CAPTION) == 1

    without_caption = reader_payload(
        [table_block("t1", ["Parcela", "Valor"], [["1", "R$ 12.400,00"]], pages=(1, 1))],
        [reader_chunk("ct", "", ["t1"], pages=(1, 1))],
    )
    silent = chunks_from_document(without_caption, external_id=EXTERNAL_ID, stamps=STAMPS)
    assert "R$ 12.400,00" in silent.chunks[0].content
    assert TABLE_CAPTION not in silent.chunks[0].content


def test_ac_033(conn: psycopg.Connection):
    """AC-033: The reader's output is stored before any piece is derived @spec:AC-033"""
    storage = FakeObjectStorage()
    docai = FakeDocumentAI()
    document_id = _document(conn)
    assert parsed_artifact(conn, document_id) is None
    with pytest.raises(ChunksBeforeArtifact):
        require_parsed_artifact(conn, document_id)

    artifact = store_parsed_artifact(
        conn,
        storage.store(),
        document_id=document_id,
        external_id=EXTERNAL_ID,
        payload=_contract(),
        reader=DOCAI_PROCESSOR,
        reader_version=DOCAI_VERSION,
        page_count=3,
    )

    name = storage.store().name_of(artifact.uri)
    assert name in storage.objects
    assert json.loads(storage.objects[name])["document"]["chunkedDocument"]["chunks"]
    with conn.cursor() as cursor:
        cursor.execute("SELECT parsed_uri FROM documents WHERE id = %s", (document_id,))
        assert cursor.fetchone()[0] == artifact.uri
    assert _chunk_count(conn, document_id) == 0

    result = recut_document(
        conn,
        storage.store(),
        document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=800, cutter="local"),
        stamps=STAMPS,
    )
    assert len(result.chunks) > 0
    assert result.used_reader is False
    assert docai.requests == []


def test_ac_034(conn: psycopg.Connection, price_book: PriceBook):
    """AC-034: Re-cutting an already-read document does not re-read it @spec:AC-034"""
    storage = FakeObjectStorage()
    docai = FakeDocumentAI()
    document_id, artifact = _stored(conn, storage)
    first = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=800, cutter="docai"), stamps=STAMPS,
    )

    smaller = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=120, cutter="docai"), stamps=STAMPS,
    )
    record_recut(
        conn, price_book,
        client=TEST_CLIENT, document_id=document_id, reader_version=DOCAI_VERSION,
        pages=artifact.page_count, reason=REASON_CONFIGURATION_CHANGED, duration_ms=40,
    )

    assert smaller.used_reader is False
    assert docai.requests == []
    assert smaller.chunks
    assert smaller.artifact.uri == artifact.uri

    assert [chunk.content for chunk in smaller.chunks] == [
        chunk.content for chunk in first.chunks
    ]

    rows = usage_between(
        conn,
        since=datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc),
        until=datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc),
    )
    for_document = [row for row in rows if row.document_id == document_id]
    assert [row.operation for row in for_document] == [OPERATION_CHUNK]
    assert OPERATION_READ not in {row.operation for row in for_document}
    (recut_row,) = for_document
    assert recut_row.reason == REASON_CONFIGURATION_CHANGED
    assert recut_row.units == artifact.page_count
    assert recut_row.provider == "internal"
    assert recut_row.cost == 0


def test_ac_035(conn: psycopg.Connection):
    """AC-035: A document above the reader's page limit is reported, not silently truncated @spec:AC-035"""
    oversized = Candidate(
        document_id=_document(conn, "processo-volumoso"),
        external_id="processo-volumoso",
        media_type="application/pdf",
        byte_size=90_000_000,
        page_count=BATCH_PAGE_LIMIT + 140,
    )
    others = (
        Candidate(
            document_id=_document(conn, "contrato-0"),
            external_id="contrato-0",
            media_type="application/pdf",
            byte_size=1_200_000,
            page_count=12,
        ),
        Candidate(
            document_id=_document(conn, "contrato-1"),
            external_id="contrato-1",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            byte_size=340_000,
        ),
    )
    authored = Candidate(
        document_id=_document(conn, "minuta-no-editor"),
        external_id="minuta-no-editor",
        media_type=export_plan(KIND_DOCUMENT).media_type,
        byte_size=48_000,
    )

    screening = screen([oversized, *others, authored])
    record_refusals(conn, screening, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION)

    assert [verdict.candidate.external_id for verdict in screening.rejected] == ["processo-volumoso"]
    record = read_record(conn, oversized.document_id)
    assert record is not None and record.state == READ_STATE_FAILED
    assert str(BATCH_PAGE_LIMIT) in record.failure_reason
    assert str(oversized.page_count) in record.failure_reason
    assert "not sent" in record.failure_reason
    batches = plan_batches(
        [
            InputDocument(candidate.external_id, f"gs://{ARTIFACT_BUCKET}/{candidate.external_id}.pdf", "application/pdf")
            for candidate in screening.accepted
        ],
        output_prefix=f"gs://{ARTIFACT_BUCKET}/runs/1",
    )
    submitted = {document.external_id for batch in batches for document in batch.documents}
    assert "processo-volumoso" not in submitted

    assert [candidate.external_id for candidate in screening.accepted] == [
        *(candidate.external_id for candidate in others),
        authored.external_id,
    ]
    assert submitted == {candidate.external_id for candidate in (*others, authored)}
    for candidate in others:
        assert read_record(conn, candidate.document_id) is None
    assert authored in screening.accepted
    assert authored.external_id in submitted
    assert read_record(conn, authored.document_id) is None


def test_ac_036(conn: psycopg.Connection):
    """AC-036: The reader version is pinned, and a run records which version read each document @spec:AC-036"""
    storage = FakeObjectStorage()
    docai = FakeDocumentAI(rounds_until_done=0)
    reader = docai.reader()
    batches = plan_batches(
        [InputDocument(EXTERNAL_ID, f"gs://{ARTIFACT_BUCKET}/{EXTERNAL_ID}.pdf", "application/pdf")],
        output_prefix=f"gs://{ARTIFACT_BUCKET}/runs/1",
    )

    run = reader.run(batches)
    assert run.succeeded

    submitted = str(docai.requests[0].url)
    assert f"/processorVersions/{DOCAI_VERSION}:batchProcess" in submitted
    assert submitted.endswith(f"{DOCAI_PROCESSOR}/processorVersions/{DOCAI_VERSION}:batchProcess")
    with pytest.raises(ReaderError) as excinfo:
        docai.reader(processor_version="")
    assert "explicit processor version" in str(excinfo.value)

    document_id = _document(conn)
    store_parsed_artifact(
        conn,
        storage.store(),
        document_id=document_id,
        external_id=EXTERNAL_ID,
        payload=_contract(),
        reader=DOCAI_PROCESSOR,
        reader_version=DOCAI_VERSION,
        page_count=3,
    )
    record = read_record(conn, document_id)
    assert record is not None
    assert record.state == READ_STATE_READ
    assert record.reader_version == DOCAI_VERSION
    assert record.reader == DOCAI_PROCESSOR
    result = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=800, cutter="local"), stamps=STAMPS,
    )
    assert {chunk.stamps.reader_version for chunk in result.chunks} == {DOCAI_VERSION}


def test_ac_037():
    """AC-037: Concurrency stays within the provider's ceiling @spec:AC-037"""
    docai = FakeDocumentAI(rounds_until_done=2)
    reader = docai.reader()
    documents = [
        InputDocument(f"doc-{n}", f"gs://{ARTIFACT_BUCKET}/doc-{n}.pdf", "application/pdf")
        for n in range(14)
    ]
    batches = plan_batches(
        documents, output_prefix=f"gs://{ARTIFACT_BUCKET}/runs/2", max_documents_per_batch=1
    )
    assert len(batches) > MAX_CONCURRENT_BATCHES

    run = reader.run(batches)

    assert docai.peak_active <= MAX_CONCURRENT_BATCHES
    assert docai.peak_active == MAX_CONCURRENT_BATCHES
    assert run.peak_in_flight == MAX_CONCURRENT_BATCHES

    assert run.succeeded
    assert len(run.batches) == len(batches)
    assert {outcome.external_id for outcome in run.documents()} == {
        document.external_id for document in documents
    }
    assert docai.waits, "the batches over the ceiling never waited for a slot"

    assert reader.run(batches[:3]).peak_in_flight <= MAX_CONCURRENT_BATCHES
    with pytest.raises(ReaderError) as excinfo:
        docai.reader(max_concurrent_batches=MAX_CONCURRENT_BATCHES + 1)
    assert str(MAX_CONCURRENT_BATCHES) in str(excinfo.value)


def test_p_009(conn: psycopg.Connection, price_book: PriceBook):
    storage = FakeObjectStorage()
    document_id = _document(conn)

    with pytest.raises(ChunksBeforeArtifact) as excinfo:
        recut_document(
            conn, storage.store(), document_id=document_id,
        cut=docai_cut,
            chunking=ChunkingOptions(size=800, cutter="local"), stamps=STAMPS,
        )
    assert "P-009" in str(excinfo.value)

    artifact = store_parsed_artifact(
        conn,
        storage.store(),
        document_id=document_id,
        external_id=EXTERNAL_ID,
        payload=_contract(),
        reader=DOCAI_PROCESSOR,
        reader_version=DOCAI_VERSION,
        page_count=3,
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT parsed_uri FROM documents WHERE id = %s", (document_id,))
        assert cursor.fetchone()[0] == artifact.uri
    assert storage.store().name_of(artifact.uri) in storage.objects

    other = FakeObjectStorage()
    with pytest.raises(Exception):
        store_parsed_artifact(
            conn,
            other.store(),
            document_id=10**9,
            external_id="orfao",
            payload=_contract(),
            reader=DOCAI_PROCESSOR,
            reader_version=DOCAI_VERSION,
        )
    assert other.objects, "the read was paid for and its output was not kept"

    kept = load_parsed_artifact(conn, storage.store(), require_parsed_artifact(conn, document_id))
    assert kept["document"]["chunkedDocument"]["chunks"]
    result = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=200, cutter="local"), stamps=STAMPS,
    )
    assert result.used_reader is False
    row = record_recut(
        conn, price_book,
        client=TEST_CLIENT, document_id=document_id, reader_version=DOCAI_VERSION,
        pages=3, reason=REASON_CONFIGURATION_CHANGED, duration_ms=12,
    )
    assert row > 0

    assert all(isinstance(chunk, Chunk) for chunk in result.chunks)
    assert all(chunk.pages is not None or chunk.heading_path for chunk in result.chunks)
