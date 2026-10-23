
import pytest

from src.models import VersionStamps
from src.processing.docai import (
    BLOCK_TEXT,
    InputDocument,
    chunks_from_document,
    index_of,
    plan_batches,
)
from src.processing.to_chunks import Box
from tests.conftest import (
    ARTIFACT_BUCKET,
    DOCAI_VERSION,
    FakeDocumentAI,
    reader_chunk,
    reader_payload,
    text_block,
)

_STAMPS = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")


def test_ac_201():
    """AC-201: A running header or footer is not offered as an answer @spec:AC-201"""
    blocks = [
        text_block("h1", "Contrato nº 123 — página 3", pages=(3, 3), kind="header"),
        text_block("f1", "Confidencial", pages=(3, 3), kind="footer"),
        text_block("p1", "texto da cláusula", pages=(3, 3)),
    ]
    index = index_of(reader_payload(blocks, []))
    assert {block_id: block.kinds for block_id, block in index.furniture.items()} == {
        "h1": (BLOCK_TEXT, "header"),
        "f1": (BLOCK_TEXT, "footer"),
    }
    assert "h1" not in index.blocks
    assert "f1" not in index.blocks
    assert index["p1"].text == "texto da cláusula"


def test_ac_202():
    """AC-202: What the reader says is a page header is not repeated into a piece @spec:AC-202"""
    blocks = [
        text_block("h1", "Contrato nº 123 — página 3", pages=(3, 3), kind="header"),
        text_block("f1", "Confidencial", pages=(3, 3), kind="footer"),
        text_block("p1", "texto da cláusula", pages=(3, 3)),
    ]
    raw_chunk = {
        "chunkId": "c1",
        "content": "",
        "sourceBlockIds": ["h1", "p1", "f1"],
        "pageHeaders": [{"text": "Contrato nº 123 — página 3"}],
        "pageFooters": [{"text": "Confidencial"}],
        "chunkFields": [{"fieldName": "valor"}],
    }
    mapped = chunks_from_document(
        reader_payload(blocks, [raw_chunk]), external_id="doc-1", stamps=_STAMPS
    )
    assert mapped.chunks[0].content == "texto da cláusula"

def test_ac_203():
    """AC-203: A merged cell keeps its span, and its neighbours keep their columns @spec:AC-203"""
    from src.processing.docai import _grid_of

    def cell(text, *, row_span=1, col_span=1):
        body: dict = {"blocks": [text_block(f"b-{text}", text, pages=(1, 1))]}
        if row_span > 1:
            body["rowSpan"] = row_span
        if col_span > 1:
            body["colSpan"] = col_span
        return body

    table = {
        "blockId": "t1",
        "pageSpan": {"pageStart": 1, "pageEnd": 1},
        "tableBlock": {
            "bodyRows": [
                {"cells": [cell("A", col_span=2, row_span=2), cell("B")]},
                {"cells": [cell("C")]},
            ],
        },
    }
    grid = _grid_of(table["tableBlock"])
    cells = {cell.text: cell for cell in grid.cells}

    assert (cells["A"].column, cells["A"].column + cells["A"].col_span) == (0, 2)
    assert (cells["B"].column, cells["B"].column + cells["B"].col_span) == (2, 3)
    assert (cells["C"].column, cells["C"].column + cells["C"].col_span) == (2, 3)
    assert grid.columns == 3
    assert index_of(reader_payload([table], []))["t1"].text.splitlines() == [
        "| A |  | B |",
        "|  |  | C |",
    ]

def test_ac_204():
    """AC-204: A table's caption and the reader's description are two texts, and both reach the piece @spec:AC-204"""
    table = {
        "blockId": "t1",
        "pageSpan": {"pageStart": 1, "pageEnd": 1},
        "tableBlock": {
            "caption": "Tabela 1 — Valores do contrato",
            "annotations": {"description": "Tabela com os valores mensais do contrato"},
            "bodyRows": [
                {"cells": [{"blocks": [text_block("b1", "R$ 100,00", pages=(1, 1))]}]}
            ],
        },
    }
    block = index_of(reader_payload([table], []))["t1"]

    assert block.caption == "Tabela 1 — Valores do contrato"
    assert block.description == "Tabela com os valores mensais do contrato"
    mapped = chunks_from_document(
        reader_payload([table], [reader_chunk("c1", "", ["t1"])]),
        external_id="doc-1",
        stamps=_STAMPS,
    )
    content = mapped.chunks[0].content
    assert content.count("Tabela 1 — Valores do contrato") == 1
    assert content.count("Tabela com os valores mensais do contrato") == 1

def test_ac_205():
    """AC-205: The reader's transcription of an image reaches the piece @spec:AC-205"""
    image = {
        "blockId": "i1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {
            "annotations": {"description": "Assinatura e carimbo do cartório"},
            "imageText": "Cartório do 5º Ofício de Notas",
        },
    }
    mapped = chunks_from_document(
        reader_payload([image], [reader_chunk("c1", "", ["i1"])]),
        external_id="doc-1",
        stamps=_STAMPS,
    )
    content = mapped.chunks[0].content
    assert "Assinatura e carimbo do cartório" in content
    assert "Cartório do 5º Ofício de Notas" in content

    inlined = "Assinatura e carimbo do cartório\n\nCartório do 5º Ofício de Notas"
    reinlined = chunks_from_document(
        reader_payload([image], [reader_chunk("c2", inlined, ["i1"])]),
        external_id="doc-1",
        stamps=_STAMPS,
    )
    assert reinlined.chunks[0].content.count("Assinatura e carimbo do cartório") == 1
    assert reinlined.chunks[0].content.count("Cartório do 5º Ofício de Notas") == 1

def test_ac_206():
    """AC-206: A converted document carries its origin and its own name @spec:AC-206"""
    blocks = [text_block("b1", "CONTRATO", pages=(1, 1), kind="heading-1")]
    identity = index_of(
        reader_payload(blocks, []),
        external_id="doc-42",
        filename="contrato-locacao.pdf",
        media_type="application/pdf",
        content_digest="ab" * 32,
    ).identity
    assert identity.name == "contrato-locacao.pdf"
    assert identity.filename == "contrato-locacao.pdf"
    assert identity.media_type == "application/pdf"
    assert identity.external_id == "doc-42"
    assert identity.uri == "source:doc-42"

def test_ac_207(conn):
    """AC-207: A split response is refused, naming what happened @spec:AC-207"""
    from src.processing.artifact import ArtifactError, read_batch_output
    from tests.conftest import ARTIFACT_BUCKET, FakeObjectStorage

    storage = FakeObjectStorage()
    destination = f"gs://{ARTIFACT_BUCKET}/runs/1/batch-0000/0/"
    storage.put(
        "runs/1/batch-0000/0/output-1.json",
        reader_payload(
            [text_block("b1", "Cláusula primeira", pages=(1, 1))],
            [reader_chunk("c1", "Cláusula primeira", ["b1"])],
        ),
    )
    storage.put(
        "runs/1/batch-0000/0/output-2.json",
        reader_payload(
            [text_block("b2", "Cláusula segunda", pages=(2, 2))],
            [reader_chunk("c2", "Cláusula segunda", ["b2"])],
        ),
    )

    with pytest.raises(ArtifactError, match=r"runs/1/batch-0000/0.*2 parts"):
        read_batch_output(conn, storage.store(), destination)

def test_ac_208(conn):
    """AC-208: A document refused for want of a locator is counted and nameable @spec:AC-208"""
    from src.analytics.corpus import refused_for_want_of_locator
    from src.processing.artifact import record_read_failure
    from src.processing.to_chunks import NO_LOCATOR_REASON, NoLocatorError
    from src.store import upsert_document

    blocks = [
        text_block("b1", "texto sem página nem título", pages=(None, None)),
        text_block("b2", "outro trecho sem localizador", pages=(None, None)),
    ]
    chunks = [reader_chunk("c1", "", ["b1"]), reader_chunk("c2", "", ["b2"])]
    with pytest.raises(NoLocatorError, match=NO_LOCATOR_REASON) as excinfo:
        chunks_from_document(
            reader_payload(blocks, chunks), external_id="doc-sem-localizador", stamps=_STAMPS
        )
    document_id = upsert_document(
        conn, external_id="doc-sem-localizador", title="Contrato sem localizador"
    )

    record_read_failure(
        conn,
        document_id=document_id,
        reason=str(excinfo.value),
        reader="docai-layout-parser",
        reader_version=DOCAI_VERSION,
    )

    refused = refused_for_want_of_locator(conn)
    assert len(refused) == 1
    assert NO_LOCATOR_REASON in refused[0].failure_reason
    assert refused[0].external_id == "doc-sem-localizador"
    assert refused[0].document_id == document_id

def test_ac_209():
    """AC-209: Geometry is requested, and what comes back reaches the piece @spec:AC-209"""
    docai = FakeDocumentAI(rounds_until_done=0)
    docai.reader().run(
        plan_batches(
            [InputDocument("doc-1", f"gs://{ARTIFACT_BUCKET}/doc-1.pdf", "application/pdf")],
            output_prefix=f"gs://{ARTIFACT_BUCKET}/runs/1",
        )
    )
    layout = docai.submissions[0]["processOptions"]["layoutConfig"]
    assert layout["returnBoundingBoxes"] is True

    with_geometry = text_block("p1", "texto da cláusula", pages=(3, 3))
    with_geometry["boundingBox"] = {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.4}
    kept = chunks_from_document(
        reader_payload([with_geometry], [reader_chunk("c1", "texto da cláusula", ["p1"])]),
        external_id="doc-1",
        stamps=_STAMPS,
    )
    assert kept.index["p1"].box == Box(left=0.1, top=0.2, right=0.9, bottom=0.4)
    assert kept.chunks[0].pages == (3, 3)

    without_geometry = text_block("p2", "outra cláusula", pages=(4, 4))
    mapped = chunks_from_document(
        reader_payload([without_geometry], [reader_chunk("c1", "outra cláusula", ["p2"])]),
        external_id="doc-2",
        stamps=_STAMPS,
    )
    assert mapped.chunks[0].pages == (4, 4)
