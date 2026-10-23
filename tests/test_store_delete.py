
from __future__ import annotations

import psycopg
import pytest

from src.store import (
    DOCUMENT_STATE_GONE,
    DOCUMENT_STATE_INDEXED,
    ModelStamp,
    Piece,
    mark_document_gone,
    upsert_document,
    upsert_pieces,
)
from tests.conftest import TEST_DIMENSION

CURRENT = ModelStamp(model="text-embedding-3", dimension=TEST_DIMENSION)


def _piece(ordinal: int = 0) -> Piece:
    return Piece(
        ordinal=ordinal,
        content=f"cláusula {ordinal}",
        embedding=[1.0] + [0.0] * (TEST_DIMENSION - 1),
        page_start=ordinal + 1,
        page_end=ordinal + 1,
    )


def _indexed_document(conn: psycopg.Connection, external_id: str, pieces: int = 2) -> int:
    document_id = upsert_document(conn, external_id=external_id)
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=CURRENT,
        pieces=[_piece(ordinal) for ordinal in range(pieces)],
    )
    return document_id


def _state_and_piece_count(conn: psycopg.Connection, document_id: int) -> tuple[str, int]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE id = %s", (document_id,))
        (state,) = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        (count,) = cursor.fetchone()
    return state, int(count)


def _raise_on(conn: psycopg.Connection, event: str, table: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "CREATE FUNCTION pg_temp.injected_failure() RETURNS trigger AS "
            "$$ BEGIN RAISE EXCEPTION 'injected failure'; END; $$ LANGUAGE plpgsql"
        )
        cursor.execute(
            f"CREATE TRIGGER injected_failure BEFORE {event} ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION pg_temp.injected_failure()"
        )


def test_removing_a_document_nobody_knows_reports_that_it_was_not_there(conn: psycopg.Connection) -> None:
    assert mark_document_gone(conn, external_id="never-seen") is False


def test_removing_the_same_document_twice_is_harmless(conn: psycopg.Connection) -> None:
    document_id = _indexed_document(conn, "drive-20")
    assert mark_document_gone(conn, external_id="drive-20") is True
    assert mark_document_gone(conn, external_id="drive-20") is True
    assert _state_and_piece_count(conn, document_id) == (DOCUMENT_STATE_GONE, 0)


def test_a_removal_touches_no_other_document(conn: psycopg.Connection) -> None:
    kept = _indexed_document(conn, "drive-21")
    removed = _indexed_document(conn, "drive-22")
    mark_document_gone(conn, external_id="drive-22")
    assert _state_and_piece_count(conn, kept) == (DOCUMENT_STATE_INDEXED, 2)
    assert _state_and_piece_count(conn, removed) == (DOCUMENT_STATE_GONE, 0)


def test_a_removal_takes_every_model_stamp_with_it(conn: psycopg.Connection) -> None:
    document_id = _indexed_document(conn, "drive-23")
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=ModelStamp(model="text-embedding-ada-002", dimension=TEST_DIMENSION),
        pieces=[_piece(0)],
    )
    mark_document_gone(conn, external_id="drive-23")
    assert _state_and_piece_count(conn, document_id) == (DOCUMENT_STATE_GONE, 0)


def test_a_failure_deleting_the_pieces_leaves_the_state_untouched(conn: psycopg.Connection) -> None:
    document_id = _indexed_document(conn, "drive-24")
    _raise_on(conn, "DELETE", "chunks")
    with pytest.raises(psycopg.errors.RaiseException):
        mark_document_gone(conn, external_id="drive-24")
    assert _state_and_piece_count(conn, document_id) == (DOCUMENT_STATE_INDEXED, 2)


def test_a_failure_marking_the_state_leaves_the_pieces_untouched(conn: psycopg.Connection) -> None:
    document_id = _indexed_document(conn, "drive-25")
    _raise_on(conn, "UPDATE", "documents")
    with pytest.raises(psycopg.errors.RaiseException):
        mark_document_gone(conn, external_id="drive-25")
    assert _state_and_piece_count(conn, document_id) == (DOCUMENT_STATE_INDEXED, 2)


def test_the_connection_stays_usable_after_an_injected_failure(conn: psycopg.Connection) -> None:
    _indexed_document(conn, "drive-26")
    _raise_on(conn, "DELETE", "chunks")
    with pytest.raises(psycopg.errors.RaiseException):
        mark_document_gone(conn, external_id="drive-26")
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM documents")
        assert cursor.fetchone()[0] >= 1
