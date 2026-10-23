"""Unit-level coverage for src/models.py that is not itself an acceptance criterion.

AC-030 is proven in tests/test_spec_processing_docai.py and principle P-002 in
tests/test_spec_store_and_search.py — see CLAUDE.md for why the proofs live
there. This file covers the guards the chunk record enforces on its own, which
are what make an uncitable piece impossible to build rather than merely
unlikely.
"""

import psycopg
import pytest

from src.models import (
    METADATA_FINGERPRINT,
    METADATA_READER_VERSION,
    METADATA_SOURCE_BLOCK_IDS,
    Chunk,
    ChunkError,
    ReaderCutChunk,
    VersionStamps,
)
from src.store import ModelStamp, StoreError, upsert_document, upsert_pieces
from tests.conftest import TEST_DIMENSION

STAMPS = VersionStamps(reader_version="pretrained-layout-parser-v1.0-2024-06-03", fingerprint="fp-1")


def _chunk(**overrides) -> Chunk:
    """A piece as the reader's cut produces one — the base plus its own half.

    The variant rather than the base, because that is what every piece in the
    archive today is: the guards below live on the base and hold for either.
    """
    base = dict(
        document_external_id="doc-1",
        ordinal=0,
        content="Cláusula quarta: da rescisão antecipada.",
        stamps=STAMPS,
        page_start=4,
        page_end=4,
        heading_path=("CONTRATO", "Cláusula Quarta"),
        source_block_ids=("b12",),
    )
    base.update(overrides)
    return ReaderCutChunk(**base)


def test_a_chunk_with_a_page_but_no_heading_is_citable():
    # The opening paragraph of a document has nothing above it, and it still
    # has to be quotable (AC-030).
    chunk = _chunk(heading_path=())

    assert chunk.pages == (4, 4)
    assert chunk.heading_path == ()


def test_a_chunk_with_a_heading_but_no_page_is_citable():
    # A source without pages at all — a spreadsheet tab, a web export — is not
    # uncitable, it is cited by its heading path.
    chunk = _chunk(page_start=None, page_end=None)

    assert chunk.pages is None
    assert chunk.heading_path


def test_a_chunk_with_neither_page_nor_heading_is_refused():
    with pytest.raises(ChunkError, match="P-002"):
        _chunk(page_start=None, page_end=None, heading_path=())


def test_a_chunk_with_no_document_is_refused():
    with pytest.raises(ChunkError, match="P-002"):
        _chunk(document_external_id="  ")


def test_an_empty_chunk_is_refused():
    # Retrievable, uncitable and always wrong: it can be returned by a search
    # and says nothing.
    with pytest.raises(ChunkError, match="no text"):
        _chunk(content="   \n ")


def test_a_backwards_page_range_is_refused():
    with pytest.raises(ChunkError, match="backwards"):
        _chunk(page_start=9, page_end=4)


def test_a_negative_position_is_refused():
    with pytest.raises(ChunkError, match="negative"):
        _chunk(ordinal=-1)


def test_one_page_open_at_either_end_still_reads_as_a_range():
    assert _chunk(page_start=7, page_end=None).pages == (7, 7)
    assert _chunk(page_start=None, page_end=7).pages == (7, 7)


def test_stamps_refuse_to_be_blank():
    # A chunk with no reader version cannot be re-read when the reader is
    # upgraded, and one with no fingerprint cannot tell a configuration change
    # from a document change.
    with pytest.raises(ChunkError, match="reader version"):
        VersionStamps(reader_version="", fingerprint="fp-1")
    with pytest.raises(ChunkError, match="fingerprint"):
        VersionStamps(reader_version="v1", fingerprint=" ")


def test_the_piece_it_becomes_carries_the_same_provenance():
    piece = _chunk().to_piece([0.1] * 8)

    assert piece.ordinal == 0
    assert piece.page_start == 4 and piece.page_end == 4
    assert piece.heading_path == ("CONTRATO", "Cláusula Quarta")
    assert piece.metadata[METADATA_SOURCE_BLOCK_IDS] == ["b12"]
    assert piece.metadata[METADATA_READER_VERSION] == STAMPS.reader_version
    assert piece.metadata[METADATA_FINGERPRINT] == "fp-1"


def test_the_chunks_own_metadata_survives_the_translation():
    piece = _chunk(metadata={"origem": "sweep"}).to_piece([0.1] * 8)

    assert piece.metadata["origem"] == "sweep"
    assert piece.metadata[METADATA_READER_VERSION] == STAMPS.reader_version


def test_the_store_accepts_what_the_chunk_record_produced(conn: psycopg.Connection):
    # The two records agree about what makes a piece citable, which is the
    # point of there being one type: the store's own guard, which refuses an
    # uncitable row, has nothing to say about a piece built from a chunk.
    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")

    written = upsert_pieces(
        conn,
        document_id=document_id,
        stamp=ModelStamp(model="m", dimension=TEST_DIMENSION),
        pieces=[_chunk().to_piece([0.1] * TEST_DIMENSION)],
    )

    assert written == 1


def test_a_piece_of_the_wrong_width_is_still_the_stores_business(conn: psycopg.Connection):
    # models.py does not check the vector: the width belongs to the embedding
    # stamp, which is applied a step later, and the store is where it is caught.
    document_id = upsert_document(conn, external_id="doc-2", title="Contrato")

    with pytest.raises(StoreError, match="dimension"):
        upsert_pieces(
            conn,
            document_id=document_id,
            stamp=ModelStamp(model="m", dimension=TEST_DIMENSION),
            pieces=[_chunk().to_piece([0.1] * 3)],
        )
