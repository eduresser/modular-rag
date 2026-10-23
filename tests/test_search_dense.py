from __future__ import annotations

import psycopg
import pytest

from src.search_dense import dense_candidates
from src.store import ModelStamp
from tests.conftest import TEST_DIMENSION, TEST_STAMP, ChunkSpec, DocumentSpec, direction, index_corpus

PREVIOUS_STAMP = ModelStamp(model="text-embedding-ada-002", dimension=TEST_DIMENSION)


def _corpus(conn: psycopg.Connection) -> dict[str, int]:
    return index_corpus(
        conn,
        [
            DocumentSpec("near", "Contrato próximo", (ChunkSpec(0, "rescisão antecipada", degrees=0, pages=(1, 1)),)),
            DocumentSpec("mid", "Contrato intermediário", (ChunkSpec(0, "reajuste anual", degrees=30, pages=(1, 1)),)),
            DocumentSpec("far", "Contrato distante", (ChunkSpec(0, "foro de eleição", degrees=80, pages=(1, 1)),)),
        ],
    )


def _ids(candidates) -> list[str]:
    return [candidate.provenance.external_id for candidate in candidates]


def test_candidates_come_back_closest_first_and_ranked_from_one(conn: psycopg.Connection) -> None:
    _corpus(conn)
    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40
    )
    assert _ids(candidates) == ["near", "mid", "far"]
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]
    assert candidates[0].leg_score > candidates[-1].leg_score


def test_depth_caps_how_many_candidates_come_back(conn: psycopg.Connection) -> None:
    _corpus(conn)
    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=2, index_search_effort=40
    )
    assert _ids(candidates) == ["near", "mid"]


def test_a_piece_of_a_document_that_is_gone_is_never_a_candidate(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [
            DocumentSpec("kept", "Mantido", (ChunkSpec(0, "rescisão", degrees=0, pages=(1, 1)),)),
            DocumentSpec("removed", "Removido", (ChunkSpec(0, "rescisão", degrees=1, pages=(1, 1)),), gone=True),
        ],
    )
    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40
    )
    assert _ids(candidates) == ["kept"]


def test_a_piece_from_an_inactive_stamp_is_never_a_candidate(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [DocumentSpec("old", "Geração anterior", (ChunkSpec(0, "rescisão", degrees=0, pages=(1, 1)),))],
        stamp=PREVIOUS_STAMP,
    )
    index_corpus(
        conn,
        [DocumentSpec("new", "Geração atual", (ChunkSpec(0, "rescisão", degrees=45, pages=(1, 1)),))],
    )
    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40
    )
    assert _ids(candidates) == ["new"]


def test_the_effort_reaches_the_query_as_a_transaction_local_setting(conn: psycopg.Connection) -> None:
    _corpus(conn)
    dense_candidates(conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=137)
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_setting('hnsw.ef_search')")
        assert cursor.fetchone()[0] == "137"


def test_a_question_embedded_by_a_different_model_is_refused(conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="different models"):
        dense_candidates(
            conn,
            query_vector=[1.0, 0.0],
            stamp=TEST_STAMP,
            depth=10,
            index_search_effort=40,
        )


def test_a_depth_below_one_is_refused(conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        dense_candidates(conn, query_vector=direction(0), stamp=TEST_STAMP, depth=0, index_search_effort=40)


def test_an_empty_archive_yields_no_candidates_and_no_error(conn: psycopg.Connection) -> None:
    assert dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40
    ) == []


def test_every_candidate_carries_the_locator_it_was_indexed_with(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [
            DocumentSpec(
                "located",
                "Contrato de locação",
                (ChunkSpec(0, "rescisão", degrees=0, pages=(4, 5), headings=("Cláusula Quinta",)),),
            )
        ],
    )
    (candidate,) = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40
    )
    assert candidate.provenance.page_label() == "pp. 4-5"
    assert candidate.provenance.heading_path == ("Cláusula Quinta",)
    assert candidate.provenance.title == "Contrato de locação"
