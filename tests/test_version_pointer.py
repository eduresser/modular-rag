
from __future__ import annotations

import psycopg
import pytest

from src.store import ModelStamp, Piece, upsert_document, upsert_pieces
from src.store_versions import (
    NoActiveVersion,
    activate_version,
    active_version,
    current_version,
    discard_other_generations,
    pieces_under,
)
from tests.conftest import TEST_DIMENSION

CURRENT = ModelStamp(model="text-embedding-3", dimension=TEST_DIMENSION)
NEXT = ModelStamp(model="text-embedding-3-large", dimension=TEST_DIMENSION)


def _store(conn: psycopg.Connection, document_id: int, stamp: ModelStamp, *, count: int) -> None:
    vector = [1.0] + [0.0] * (TEST_DIMENSION - 1)
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=stamp,
        pieces=[
            Piece(ordinal=n, content=f"peça {n}", embedding=vector, page_start=1, page_end=1)
            for n in range(count)
        ],
    )


def test_an_archive_before_its_first_cutover_has_no_active_version(conn: psycopg.Connection) -> None:
    assert current_version(conn) is None


def test_asking_for_the_active_version_before_one_exists_says_why(conn: psycopg.Connection) -> None:
    with pytest.raises(NoActiveVersion, match="active embedding model"):
        active_version(conn)


def test_the_first_cutover_reports_that_there_was_nothing_before(conn: psycopg.Connection) -> None:
    assert activate_version(conn, CURRENT) is None
    assert active_version(conn) == CURRENT


def test_a_cutover_returns_the_stamp_it_replaced(conn: psycopg.Connection) -> None:
    activate_version(conn, CURRENT)
    assert activate_version(conn, NEXT) == CURRENT
    assert active_version(conn) == NEXT


def test_a_rollback_is_the_same_update_in_reverse(conn: psycopg.Connection) -> None:
    activate_version(conn, CURRENT)
    replaced = activate_version(conn, NEXT)
    activate_version(conn, replaced)
    assert active_version(conn) == CURRENT


def test_the_pointer_never_becomes_two_rows(conn: psycopg.Connection) -> None:
    activate_version(conn, CURRENT)
    activate_version(conn, NEXT)
    activate_version(conn, CURRENT)
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM active_model_version")
        assert cursor.fetchone()[0] == 1


def test_a_cutover_moves_no_pieces(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="version-1")
    vector = [1.0] + [0.0] * (TEST_DIMENSION - 1)
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT,
                  pieces=[Piece(ordinal=0, content="a", embedding=vector, page_start=1, page_end=1)])
    upsert_pieces(conn, document_id=document_id, stamp=NEXT,
                  pieces=[Piece(ordinal=0, content="a", embedding=vector, page_start=1, page_end=1),
                          Piece(ordinal=1, content="b", embedding=vector, page_start=2, page_end=2)])
    activate_version(conn, CURRENT)
    activate_version(conn, NEXT)
    assert pieces_under(conn, CURRENT) == 1
    assert pieces_under(conn, NEXT) == 2


def test_a_stamp_nothing_was_written_under_counts_zero(conn: psycopg.Connection) -> None:
    assert pieces_under(conn, ModelStamp(model="never-used", dimension=TEST_DIMENSION)) == 0


def test_the_cutover_does_not_commit_on_its_own(conn: psycopg.Connection) -> None:
    activate_version(conn, CURRENT)
    conn.rollback()
    assert current_version(conn) is None


def test_a_cutover_discards_the_generation_it_replaced(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="doc-1")
    _store(conn, document_id, CURRENT, count=2)
    _store(conn, document_id, NEXT, count=2)
    activate_version(conn, CURRENT)

    gone = discard_other_generations(conn, NEXT)

    assert gone == {CURRENT: 2}
    assert pieces_under(conn, NEXT) == 2
    assert pieces_under(conn, CURRENT) == 0


def test_discarding_refuses_to_keep_a_generation_that_holds_nothing(
    conn: psycopg.Connection,
) -> None:
    document_id = upsert_document(conn, external_id="doc-1")
    _store(conn, document_id, CURRENT, count=2)

    with pytest.raises(ValueError) as excinfo:
        discard_other_generations(conn, NEXT)

    assert "no pieces at all" in str(excinfo.value)
    assert pieces_under(conn, CURRENT) == 2


def test_discarding_with_only_one_generation_present_changes_nothing(
    conn: psycopg.Connection,
) -> None:
    document_id = upsert_document(conn, external_id="doc-1")
    _store(conn, document_id, CURRENT, count=3)

    assert discard_other_generations(conn, CURRENT) == {}
    assert pieces_under(conn, CURRENT) == 3
