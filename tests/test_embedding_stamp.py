"""The model stamp (T-053): which model produced a vector, on the vector's row.

AC-053 is proven in tests/test_spec_embedding_provider.py — see CLAUDE.md for
why the proof lives there. This file covers the pairing of pieces to vectors and
the question a re-embedding actually has to ask the archive.
"""

import pytest

from src.config import load_config
from src.embedding.stamp import (
    StampError,
    attach,
    documents_missing,
    stamp_from_config,
    stamp_of,
    stamps_present,
)
from src.models import Chunk, VersionStamps
from src.store import ModelStamp, upsert_document, upsert_pieces
from tests.conftest import TEST_DIMENSION, TEST_STAMP, ChunkSpec, DocumentSpec, index_corpus, vector_for
from tests.conftest import FULL_DEPLOYMENT, deployment_config

# Deliberately NOT the stamp the fixtures write under: the whole of what this
# constant is for is being a second generation.
OTHER_STAMP = ModelStamp(model="qwen/qwen3-embedding-4b", dimension=TEST_DIMENSION)


def _chunk(ordinal: int, content: str) -> Chunk:
    return Chunk(
        document_external_id="doc-1",
        ordinal=ordinal,
        content=content,
        stamps=VersionStamps(reader_version="v1", fingerprint="fp1"),
        page_start=1,
        page_end=1,
    )


def test_the_stamp_comes_from_the_two_settings_and_nothing_else():
    stamp = stamp_from_config(deployment_config(FULL_DEPLOYMENT))

    assert stamp == ModelStamp(model="qwen/qwen3-embedding-8b", dimension=2048)


def test_pieces_and_vectors_are_paired_in_order():
    chunks = [_chunk(0, "primeiro"), _chunk(1, "segundo")]
    vectors = [vector_for("primeiro"), vector_for("segundo")]

    pieces = attach(chunks, vectors, TEST_STAMP)

    assert [piece.ordinal for piece in pieces] == [0, 1]
    assert pieces[0].embedding == vector_for("primeiro")


def test_a_missing_vector_is_refused_rather_than_shifting_every_piece_along():
    # Pairing by position past a gap gives every piece its neighbour's meaning,
    # and no single row would look wrong.
    with pytest.raises(StampError) as excinfo:
        attach([_chunk(0, "a"), _chunk(1, "b")], [vector_for("a")], TEST_STAMP)

    assert "cannot be paired" in str(excinfo.value)


def test_a_vector_of_the_wrong_width_never_reaches_the_store():
    with pytest.raises(StampError) as excinfo:
        attach([_chunk(0, "a")], [(0.1, 0.2)], TEST_STAMP)

    assert str(TEST_DIMENSION) in str(excinfo.value)


def test_a_stored_piece_names_the_model_and_the_width_that_produced_it(conn):
    index_corpus(
        conn,
        [DocumentSpec(external_id="doc-1", title="Contrato", chunks=(ChunkSpec(0, "cláusula", pages=(1, 1)),))],
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM chunks")
        (chunk_id,) = cursor.fetchone()

    assert stamp_of(conn, chunk_id) == TEST_STAMP


def test_a_piece_that_is_not_there_has_no_stamp(conn):
    assert stamp_of(conn, 999_999) is None


def test_two_generations_of_vectors_sit_in_the_archive_side_by_side(conn):
    # The normal state during a re-embedding, not a fault.
    documents = [
        DocumentSpec(external_id="doc-1", title="Contrato", chunks=(ChunkSpec(0, "cláusula", pages=(1, 1)),))
    ]
    index_corpus(conn, documents)
    index_corpus(conn, documents, stamp=OTHER_STAMP)

    assert set(stamps_present(conn)) == {TEST_STAMP, OTHER_STAMP}


def test_the_archive_says_which_documents_have_no_vectors_under_a_stamp(conn):
    # The antidote to the failure T-053 exists for: a pipeline comparing content
    # digests finds every document unchanged when only the model changed, skips
    # all of them, and leaves the index serving the previous generation.
    index_corpus(
        conn,
        [
            DocumentSpec(external_id="doc-1", title="A", chunks=(ChunkSpec(0, "um", pages=(1, 1)),)),
            DocumentSpec(external_id="doc-2", title="B", chunks=(ChunkSpec(0, "dois", pages=(1, 1)),)),
        ],
    )
    document_id = upsert_document(conn, external_id="doc-1")
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=OTHER_STAMP,
        pieces=attach([_chunk(0, "um")], [vector_for("um")], OTHER_STAMP),
    )

    missing = documents_missing(conn, OTHER_STAMP)

    assert missing == (upsert_document(conn, external_id="doc-2"),)


def test_nothing_is_missing_once_every_document_has_been_re_embedded(conn):
    documents = [
        DocumentSpec(external_id="doc-1", title="A", chunks=(ChunkSpec(0, "um", pages=(1, 1)),))
    ]
    index_corpus(conn, documents)
    index_corpus(conn, documents, stamp=OTHER_STAMP)

    assert documents_missing(conn, OTHER_STAMP) == ()
