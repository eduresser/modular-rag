
from __future__ import annotations

import psycopg
import pytest
from conftest import API_CORPUS, TEST_DIMENSION, TEST_STAMP, api_config, index_corpus

from src.responses import result_id
from src.store import ModelStamp, upsert_document
from src.store_versions import activate_version
from src.usecases.fetch import (
    FetchedDocument,
    FetchError,
    FetchFailed,
    NotHeld,
    Outline,
    fetch,
    fetch_on,
    outline_on,
    page_range,
    window,
)
from conftest import ChunkSpec, DocumentSpec

PREVIOUS = ModelStamp(model="text-embedding-2", dimension=TEST_DIMENSION)


@pytest.fixture
def corpus(conn: psycopg.Connection) -> psycopg.Connection:
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    return conn


def _config(monkeypatch, **overrides):
    return api_config(monkeypatch, "postgresql://localhost/unused", **overrides)


def test_a_request_that_names_no_window_gets_the_deployments_own(monkeypatch):
    config = _config(monkeypatch, RAG_API_MAX_DOCUMENT_PIECES="20")

    assert window(config, offset=0, limit=None) == (0, 20)


def test_a_smaller_window_is_allowed_and_a_larger_one_is_refused_by_name(monkeypatch):
    config = _config(monkeypatch, RAG_API_MAX_DOCUMENT_PIECES="20")

    assert window(config, offset=5, limit=3) == (5, 3)
    with pytest.raises(FetchError) as excinfo:
        window(config, offset=0, limit=21)

    assert "api.limits.fetch.limit" in str(excinfo.value)
    assert "20" in str(excinfo.value)


@pytest.mark.parametrize("offset,limit", [(-1, None), (0, 0), (0, -3)])
def test_a_window_that_cannot_return_anything_is_refused(monkeypatch, offset, limit):
    config = _config(monkeypatch)

    with pytest.raises(FetchError):
        window(config, offset=offset, limit=limit)


def test_a_document_comes_back_with_its_pieces_in_order_and_its_whole_count(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    found = fetch_on(corpus, "drive-rescisao", config=config)

    assert isinstance(found, FetchedDocument)
    assert found.title == "Contrato de locação — Sala 402"
    assert found.piece_count == 2
    assert [piece.ordinal for piece in found.pieces] == [0, 1]
    assert found.pieces[0].page_start == 4
    assert found.pieces[1].heading_path == ("Da rescisão", "Da multa")


def test_the_count_is_the_documents_and_never_the_windows(monkeypatch, corpus):
    config = _config(monkeypatch)

    windowed = fetch_on(corpus, "drive-rescisao", config=config, offset=1, limit=1)

    assert isinstance(windowed, FetchedDocument)
    assert windowed.piece_count == 2
    assert windowed.offset == 1
    assert [piece.ordinal for piece in windowed.pieces] == [1]


def test_a_window_past_the_end_is_an_empty_window_and_not_a_missing_document(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    beyond = fetch_on(corpus, "drive-rescisao", config=config, offset=99)

    assert isinstance(beyond, FetchedDocument)
    assert beyond.pieces == ()
    assert beyond.piece_count == 2


def test_an_identifier_issued_before_this_change_still_reads_as_its_document(monkeypatch, corpus):
    config = _config(monkeypatch)

    with corpus.cursor() as cursor:
        cursor.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.external_id = 'drive-rescisao' ORDER BY c.ordinal LIMIT 1"
        )
        (chunk_id,) = cursor.fetchone()

    found = fetch_on(corpus, f"drive-rescisao#{chunk_id}", config=config)

    assert isinstance(found, FetchedDocument)
    assert found.external_id == "drive-rescisao"
    assert [piece.ordinal for piece in found.pieces] == [0, 1]


def test_an_identifier_naming_a_piece_opens_the_window_there(monkeypatch, corpus):
    config = _config(monkeypatch)
    with corpus.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = 'drive-rescisao'")
        (document_id,) = cursor.fetchone()

    found = fetch_on(corpus, result_id(document_id, 1), config=config)

    assert isinstance(found, FetchedDocument)
    assert [piece.ordinal for piece in found.pieces] == [1]
    assert found.pieces[0].heading_path == ("Da rescisão", "Da multa")


def test_naming_an_earlier_piece_opens_the_window_earlier(monkeypatch, corpus):
    config = _config(monkeypatch)
    with corpus.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = 'drive-rescisao'")
        (document_id,) = cursor.fetchone()

    found = fetch_on(corpus, result_id(document_id, 0), config=config)

    assert isinstance(found, FetchedDocument)
    assert [piece.ordinal for piece in found.pieces] == [0, 1]


def test_page_range_needs_both_of_its_bounds():
    with pytest.raises(FetchError):
        page_range(4, None)
    with pytest.raises(FetchError):
        page_range(None, 5)


def test_page_range_is_refused_before_the_first_page_or_running_backwards():
    with pytest.raises(FetchError):
        page_range(0, 1)
    with pytest.raises(FetchError):
        page_range(5, 4)

    assert page_range(4, 5) == (4, 5)


def test_a_document_is_readable_by_page_range(monkeypatch, corpus):
    config = _config(monkeypatch)

    found = fetch_on(corpus, "drive-rescisao", config=config, page_start=4, page_end=5)

    assert isinstance(found, FetchedDocument)
    assert [piece.ordinal for piece in found.pieces] == [0, 1]
    assert found.piece_count == 2


def test_a_narrower_page_range_excludes_pieces_outside_it(monkeypatch, corpus):
    config = _config(monkeypatch)

    found = fetch_on(corpus, "drive-rescisao", config=config, page_start=4, page_end=4)

    assert isinstance(found, FetchedDocument)
    assert [piece.ordinal for piece in found.pieces] == [0]


def test_a_page_range_matching_no_piece_is_an_empty_window_not_a_missing_document(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    found = fetch_on(corpus, "drive-rescisao", config=config, page_start=40, page_end=41)

    assert isinstance(found, FetchedDocument)
    assert found.pieces == ()
    assert found.piece_count == 2


def test_a_page_range_on_a_document_with_no_pagination_is_refused_with_its_reason(
    monkeypatch, conn
):
    unpaginated = DocumentSpec(
        "drive-sem-paginas",
        "Termo de aditamento — sem paginação",
        (
            ChunkSpec(
                0,
                "Cláusula única: as partes ratificam os demais termos.",
                degrees=0,
                headings=("Da ratificação",),
            ),
        ),
    )
    index_corpus(conn, (unpaginated,))
    activate_version(conn, TEST_STAMP)
    config = _config(monkeypatch)

    with pytest.raises(FetchError) as excinfo:
        fetch_on(conn, "drive-sem-paginas", config=config, page_start=1, page_end=1)

    reason = str(excinfo.value)
    assert "no pagination" in reason
    assert "piece offset" in reason


def test_a_document_the_archive_does_not_hold_is_an_empty_and_not_a_failure(
    monkeypatch, corpus
):
    config = _config(monkeypatch)

    assert isinstance(fetch_on(corpus, "drive-nao-existe", config=config), NotHeld)


def test_a_document_that_is_gone_is_not_returned(monkeypatch, corpus):
    config = _config(monkeypatch)

    assert isinstance(fetch_on(corpus, "drive-removido", config=config), NotHeld)


def test_an_empty_identifier_is_refused_rather_than_read(monkeypatch, corpus):
    config = _config(monkeypatch)

    with pytest.raises(FetchError):
        fetch_on(corpus, "   ", config=config)


def test_a_fetch_is_scoped_to_the_generation_the_archive_is_serving(monkeypatch, conn):
    index_corpus(conn, API_CORPUS, stamp=PREVIOUS)
    index_corpus(
        conn,
        (API_CORPUS[0],),
        stamp=TEST_STAMP,
    )
    activate_version(conn, PREVIOUS)
    config = _config(monkeypatch)

    under_previous = fetch_on(conn, "drive-rescisao", config=config)
    assert isinstance(under_previous, FetchedDocument)
    assert under_previous.piece_count == 2

    activate_version(conn, TEST_STAMP)
    under_current = fetch_on(conn, "drive-reajuste", config=config)
    assert isinstance(under_current, FetchedDocument)
    assert under_current.piece_count == 0


def test_an_unreachable_archive_is_a_failure_carrying_the_reason(monkeypatch):
    config = _config(monkeypatch, RAG_DB_URL="postgresql://127.0.0.1:1/nowhere")

    outcome = fetch("drive-rescisao", config=config)

    assert isinstance(outcome, FetchFailed)
    assert "could not reach the archive" in outcome.reason


def test_an_archive_with_no_active_version_is_a_failure_and_not_a_missing_document(
    monkeypatch, conn
):
    index_corpus(conn, API_CORPUS)
    config = _config(monkeypatch)

    outcome = fetch("drive-rescisao", config=config, connection=conn)

    assert isinstance(outcome, FetchFailed)


def test_an_outline_is_built_from_headings_when_the_document_has_them(corpus):
    found = outline_on(corpus, "drive-rescisao")

    assert isinstance(found, Outline)
    assert found.basis == "headings"
    assert [entry.ordinal for entry in found.entries] == [0, 1]
    assert found.entries[0].label == "Da rescisão"
    assert found.entries[1].label == "Da rescisão › Da multa"


def test_an_outline_degrades_to_pages_when_the_document_has_no_headings(conn):
    unheaded = DocumentSpec(
        "drive-sem-titulos",
        "Aditivo sem títulos",
        (
            ChunkSpec(0, "Primeira cláusula do aditivo.", degrees=0, pages=(1, 1)),
            ChunkSpec(1, "Segunda cláusula do aditivo.", degrees=0, pages=(2, 2)),
        ),
    )
    index_corpus(conn, (unheaded,))
    activate_version(conn, TEST_STAMP)

    found = outline_on(conn, "drive-sem-titulos")

    assert isinstance(found, Outline)
    assert found.basis == "pages"
    assert found.entries != ()
    assert [entry.ordinal for entry in found.entries] == [0]
    assert found.entries[0].label == "pages 1\u20132"
    assert found.entries[0].ordinal_end == 1


def test_an_outline_degrades_to_pieces_when_the_document_has_neither(conn):
    from src.db import halfvec_literal
    from conftest import direction

    document_id = upsert_document(conn, external_id="drive-sem-nada", title="Documento sem estrutura")
    with conn.cursor() as cursor:
        for ordinal in (0, 1):
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id, ordinal, content, embed_model, embed_dimension, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s::halfvec)
                """,
                (
                    document_id,
                    ordinal,
                    f"Peça {ordinal}.",
                    TEST_STAMP.model,
                    TEST_STAMP.dimension,
                    halfvec_literal(direction(0)),
                ),
            )
    activate_version(conn, TEST_STAMP)

    found = outline_on(conn, "drive-sem-nada")

    assert isinstance(found, Outline)
    assert found.basis == "pieces"
    assert found.entries != ()
    assert [entry.ordinal for entry in found.entries] == [0]
    assert found.entries[0].label == "pieces 0\u20131"
    assert found.entries[0].ordinal_end == 1


def test_an_outline_of_a_document_the_archive_does_not_hold_is_an_empty_and_not_a_failure(
    corpus,
):
    assert isinstance(outline_on(corpus, "drive-nao-existe"), NotHeld)


def test_an_outline_of_a_document_that_is_gone_is_not_returned(corpus):
    assert isinstance(outline_on(corpus, "drive-removido"), NotHeld)
