
from __future__ import annotations

import psycopg
import pytest

from src.filters import FIELDS, OPERATORS
from src.usecases.vocabulary import published_vocabulary
from tests.conftest import ChunkSpec, DocumentSpec, index_corpus


def _record_ingest(
    conn: psycopg.Connection, *, document_id: int, process_module: str, reader_version: str = "v1"
) -> None:
    conn.execute(
        """
        INSERT INTO document_ingest (
            document_id, content_digest, fingerprint, read_fingerprint,
            cut_fingerprint, embed_fingerprint, reader_version, cut_version
        )
        VALUES (%s, 'digest', 'fp', 'fp-read', 'fp-cut', 'fp-embed', %s, 'cut-v1')
        ON CONFLICT (document_id) DO UPDATE SET reader_version = EXCLUDED.reader_version
        """,
        (document_id, reader_version),
    )
    conn.execute(
        "UPDATE document_ingest SET process_module = %s WHERE document_id = %s",
        (process_module, document_id),
    )


def test_every_field_is_listed_with_its_type_and_operators(conn: psycopg.Connection):
    fields = published_vocabulary(conn)

    assert {field.name for field in fields} == set(FIELDS)
    for field in fields:
        assert field.type == FIELDS[field.name].type
        assert field.operators == OPERATORS[field.type]


def test_the_published_list_is_the_vocabulary_search_validates_against(conn: psycopg.Connection):
    from src.usecases import vocabulary as vocabulary_module

    assert vocabulary_module.FIELDS is FIELDS
    assert vocabulary_module.OPERATORS is OPERATORS


def test_a_text_or_number_field_carries_no_values(conn: psycopg.Connection):
    fields = {field.name: field for field in published_vocabulary(conn)}

    for name in (
        "document_id", "title", "content", "size_bytes", "page",
        "ordinal", "modified_at", "indexed_at", "reader_version",
    ):
        assert fields[name].values is None
        assert fields[name].values_open is False


def test_an_enum_field_is_counted_over_live_documents_only(conn: psycopg.Connection):
    written = index_corpus(
        conn,
        [
            DocumentSpec("pdf-one", "PDF um", (ChunkSpec(0, "conteudo", pages=(1, 1)),)),
            DocumentSpec("pdf-two", "PDF dois", (ChunkSpec(0, "conteudo", pages=(1, 1)),)),
            DocumentSpec("docx-one", "DOCX um", (ChunkSpec(0, "conteudo", pages=(1, 1)),)),
            DocumentSpec("gone-one", "Sumiu", (ChunkSpec(0, "conteudo", pages=(1, 1)),), gone=True),
        ],
    )
    conn.execute(
        "UPDATE documents SET media_type = 'application/pdf' WHERE external_id IN (%s, %s)",
        ("pdf-one", "pdf-two"),
    )
    conn.execute(
        "UPDATE documents SET media_type = "
        "'application/vnd.openxmlformats-officedocument.wordprocessingml.document' "
        "WHERE external_id = %s",
        ("docx-one",),
    )
    conn.execute(
        "UPDATE documents SET media_type = 'application/pdf' WHERE external_id = %s",
        ("gone-one",),
    )

    _record_ingest(conn, document_id=written["pdf-one"], process_module="docai")
    _record_ingest(conn, document_id=written["pdf-two"], process_module="pdftext")
    _record_ingest(conn, document_id=written["docx-one"], process_module="docai")
    _record_ingest(conn, document_id=written["gone-one"], process_module="docai")

    fields = {field.name: field for field in published_vocabulary(conn)}

    media_type_values = {v.value: v.document_count for v in fields["media_type"].values}
    assert media_type_values == {
        "application/pdf": 2,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 1,
    }

    process_module_values = {v.value: v.document_count for v in fields["process_module"].values}
    assert process_module_values == {"docai": 2, "pdftext": 1}


def test_a_field_whose_values_grow_with_the_corpus_publishes_none(conn: psycopg.Connection):
    index_corpus(
        conn,
        [
            DocumentSpec(
                "with-two-pieces", "Doc",
                (
                    ChunkSpec(0, "primeira", headings=("Cláusula 5", "Parágrafo 1")),
                    ChunkSpec(1, "segunda", headings=("Cláusula 5", "Parágrafo 2")),
                ),
            ),
            DocumentSpec(
                "other-doc", "Outro",
                (ChunkSpec(0, "terceira", headings=("Cláusula 9",)),),
            ),
        ],
    )
    conn.execute(
        "UPDATE documents SET location = ARRAY['Contratos', '2024'] WHERE external_id = %s",
        ("with-two-pieces",),
    )

    fields = {field.name: field for field in published_vocabulary(conn)}

    assert fields["folder"].values is None
    assert fields["folder"].values_open is True
    assert fields["heading_path"].values is None
    assert fields["heading_path"].values_open is True

    from src.filters import OPERATORS, TYPE_TEXT_LIST

    assert fields["folder"].operators == OPERATORS[TYPE_TEXT_LIST]
    assert fields["heading_path"].operators == OPERATORS[TYPE_TEXT_LIST]


def test_block_kind_is_counted_from_the_pieces_own_metadata(conn: psycopg.Connection):
    from src.store import upsert_document, upsert_pieces
    from src.store import Piece
    from tests.conftest import TEST_STAMP, direction

    document_id = upsert_document(conn, external_id="with-blocks", title="Com blocos")
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=TEST_STAMP,
        pieces=[
            Piece(
                ordinal=0,
                content="texto",
                embedding=direction(0),
                page_start=1,
                page_end=1,
                metadata={"block_kinds": ["tableBlock", "textBlock", "heading-1"]},
            ),
        ],
    )

    fields = {field.name: field for field in published_vocabulary(conn)}
    values = {v.value: v.document_count for v in fields["block_kind"].values}

    assert values == {"tableBlock": 1, "textBlock": 1, "heading-1": 1}


def test_a_field_present_nowhere_publishes_an_empty_list(conn: psycopg.Connection):
    index_corpus(conn, [DocumentSpec("plain", "Doc", (ChunkSpec(0, "conteudo", pages=(1, 1)),))])

    fields = {field.name: field for field in published_vocabulary(conn)}

    assert fields["media_type"].values == ()
    assert fields["process_module"].values == ()
