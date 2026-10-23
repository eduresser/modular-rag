import psycopg
import pytest

from src.models import VersionStamps
from src.processing.artifact import ChunksBeforeArtifact, store_parsed_artifact
from src.processing.docai import ChunkingOptions, index_blocks
from src.processing.docai import chunks_from_document as docai_cut
from src.processing.recut import RECUT_LOCAL, RECUT_PROVIDER, recut_document
from src.processing.to_chunks import CutPiece, pieces_from
from src.store import upsert_document
from tests.conftest import (
    DOCAI_PROCESSOR,
    DOCAI_VERSION,
    FakeDocumentAI,
    FakeObjectStorage,
    list_block,
    reader_chunk,
    reader_payload,
    text_block,
)

STAMPS = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")
EXTERNAL_ID = "doc-1"


def _tree():
    return reader_payload(
        [
            text_block("h1", "CONTRATO", pages=(1, 1), kind="heading-1"),
            text_block("h2", "Cláusula Primeira", pages=(1, 1), kind="heading-2"),
            text_block("p1", "a" * 90, pages=(1, 1)),
            text_block("p2", "b" * 90, pages=(1, 2)),
            text_block("h3", "Cláusula Segunda", pages=(2, 2), kind="heading-2"),
            text_block("p3", "c" * 40, pages=(2, 2)),
            list_block("l1", ["primeiro item", "segundo item"], pages=(2, 2)),
        ],
        [reader_chunk("c1", "irrelevante", ["p1"])],
    )


def _stored(conn: psycopg.Connection, storage: FakeObjectStorage) -> int:
    document_id = upsert_document(conn, external_id=EXTERNAL_ID, title="Contrato")
    store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=_tree(), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION, page_count=2,
    )
    return document_id


def test_a_recut_without_a_reader_speaks_to_nobody(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    docai = FakeDocumentAI()
    document_id = _stored(conn, storage)

    result = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=400, cutter="local"), stamps=STAMPS,
    )

    assert result.path == RECUT_LOCAL
    assert result.used_reader is False
    assert docai.requests == []
    assert result.pages == 2


def test_the_provider_path_is_reached_only_when_a_reader_is_handed_in(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _stored(conn, storage)
    docai = FakeDocumentAI(
        response=reader_payload(
            [text_block("b1", "recortado", pages=(1, 1))],
            [reader_chunk("c1", "recortado", ["b1"])],
        )
    )

    result = recut_document(
        conn, storage.store(), document_id=document_id,
        cut=docai_cut,
        chunking=ChunkingOptions(size=256, cutter="local"), stamps=STAMPS, reader=docai.reader(),
    )

    assert result.path == RECUT_PROVIDER
    assert result.used_reader is True
    body = docai.requests[0].content.decode()
    assert "inlineDocument" in body
    assert '"chunkSize":256' in body.replace(" ", "")


def test_a_document_that_was_never_read_cannot_be_recut(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = upsert_document(conn, external_id="nunca-lido", title="x")

    with pytest.raises(ChunksBeforeArtifact):
        recut_document(
            conn, storage.store(), document_id=document_id,
        cut=docai_cut,
            chunking=ChunkingOptions(size=400, cutter="local"), stamps=STAMPS,
        )


def test_the_cutter_has_no_default_because_a_default_is_how_they_disagreed():
    with pytest.raises(TypeError):
        ChunkingOptions(size=800)


def test_a_piece_that_names_no_page_of_its_own_reads_one_off_its_blocks():
    index = index_blocks(_tree())

    walked = CutPiece(text="um pedaço", blocks=(index["p1"],), headings=("CONTRATO",))

    pieces = pieces_from([walked], external_id=EXTERNAL_ID, stamps=STAMPS)

    assert walked.page_start is None and walked.page_end is None
    assert pieces[0].content == "um pedaço"
    assert pieces[0].pages == (1, 1)
    assert pieces[0].heading_path == ("CONTRATO",)
    assert pieces[0].source_block_ids == ()

