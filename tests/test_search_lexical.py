from __future__ import annotations

import psycopg
import pytest

from src.search_lexical import lexical_candidates
from src.store import ModelStamp
from tests.conftest import TEST_DIMENSION, TEST_STAMP, ChunkSpec, DocumentSpec, index_corpus

PREVIOUS_STAMP = ModelStamp(model="text-embedding-ada-002", dimension=TEST_DIMENSION)
CASE_NUMBER = "0001234-56.2020.8.26.0100"


def _ids(candidates) -> list[str]:
    return [candidate.provenance.external_id for candidate in candidates]


def _corpus(conn: psycopg.Connection) -> dict[str, int]:
    return index_corpus(
        conn,
        [
            DocumentSpec(
                "processo",
                "Petição inicial",
                (ChunkSpec(0, f"O processo {CASE_NUMBER} tramita na 3ª Vara Cível", pages=(1, 1)),),
            ),
            DocumentSpec(
                "rescisao",
                "Contrato de locação",
                (ChunkSpec(0, "Da rescisão antecipada do contrato pelo locatário", pages=(4, 4)),),
            ),
            DocumentSpec(
                "reajuste",
                "Aditivo",
                (ChunkSpec(0, "Do reajuste anual pelo índice acordado", pages=(2, 2)),),
            ),
        ],
    )


def test_a_case_number_is_found_by_its_characters(conn: psycopg.Connection) -> None:
    _corpus(conn)
    assert _ids(lexical_candidates(conn, question=CASE_NUMBER, stamp=TEST_STAMP, depth=10)) == ["processo"]


def test_a_quoted_phrase_is_taken_as_a_phrase(conn: psycopg.Connection) -> None:
    _corpus(conn)
    found = lexical_candidates(conn, question='"rescisão antecipada"', stamp=TEST_STAMP, depth=10)
    assert _ids(found) == ["rescisao"]


def test_the_portuguese_configuration_stems_the_question_and_the_corpus_alike(
    conn: psycopg.Connection,
) -> None:
    _corpus(conn)
    assert _ids(lexical_candidates(conn, question="locatários", stamp=TEST_STAMP, depth=10)) == ["rescisao"]


def test_punctuation_a_user_typed_does_not_raise(conn: psycopg.Connection) -> None:
    _corpus(conn)
    assert lexical_candidates(conn, question="rescisão???! & |", stamp=TEST_STAMP, depth=10)


def test_a_question_with_no_searchable_term_matches_nothing_without_failing(
    conn: psycopg.Connection,
) -> None:
    _corpus(conn)
    assert lexical_candidates(conn, question="!!! ???", stamp=TEST_STAMP, depth=10) == []


def test_candidates_are_ranked_from_one_best_first(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [
            DocumentSpec("uma", "Uma menção", (ChunkSpec(0, "rescisão do contrato", pages=(1, 1)),)),
            DocumentSpec(
                "muitas",
                "Muitas menções",
                (ChunkSpec(0, "rescisão, rescisão e nova rescisão do contrato", pages=(1, 1)),),
            ),
        ],
    )
    found = lexical_candidates(conn, question="rescisão", stamp=TEST_STAMP, depth=10)
    assert _ids(found) == ["muitas", "uma"]
    assert [candidate.rank for candidate in found] == [1, 2]


def test_depth_caps_how_many_candidates_come_back(conn: psycopg.Connection) -> None:
    _corpus(conn)
    assert len(lexical_candidates(conn, question="contrato do", stamp=TEST_STAMP, depth=1)) <= 1


def test_a_piece_of_a_document_that_is_gone_is_never_a_candidate(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [
            DocumentSpec("kept", "Mantido", (ChunkSpec(0, "cláusula de rescisão", pages=(1, 1)),)),
            DocumentSpec("removed", "Removido", (ChunkSpec(0, "cláusula de rescisão", pages=(1, 1)),), gone=True),
        ],
    )
    assert _ids(lexical_candidates(conn, question="rescisão", stamp=TEST_STAMP, depth=10)) == ["kept"]


def test_a_piece_from_an_inactive_stamp_is_never_a_candidate(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [DocumentSpec("old", "Anterior", (ChunkSpec(0, "cláusula de rescisão", pages=(1, 1)),))],
        stamp=PREVIOUS_STAMP,
    )
    assert lexical_candidates(conn, question="rescisão", stamp=TEST_STAMP, depth=10) == []


def test_a_depth_below_one_is_refused(conn: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        lexical_candidates(conn, question="rescisão", stamp=TEST_STAMP, depth=0)


def test_the_ranking_has_no_term_rarity_component(conn: psycopg.Connection) -> None:
    index_corpus(
        conn,
        [
            DocumentSpec("a", "A", (ChunkSpec(0, "contrato comodato", pages=(1, 1)),)),
            DocumentSpec("b", "B", (ChunkSpec(0, "contrato de locação", pages=(1, 1)),)),
            DocumentSpec("c", "C", (ChunkSpec(0, "contrato de prestação", pages=(1, 1)),)),
        ],
    )
    common = lexical_candidates(conn, question="contrato", stamp=TEST_STAMP, depth=10)
    rare = lexical_candidates(conn, question="comodato", stamp=TEST_STAMP, depth=10)
    score_for_common = next(c.leg_score for c in common if c.provenance.external_id == "a")
    score_for_rare = next(c.leg_score for c in rare if c.provenance.external_id == "a")
    assert score_for_rare == pytest.approx(score_for_common)
