
from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

from src.store import ModelStamp, Piece, StoreError, upsert_document, upsert_pieces
from tests.conftest import TEST_DIMENSION

CURRENT = ModelStamp(model="text-embedding-3", dimension=TEST_DIMENSION)
PREVIOUS = ModelStamp(model="text-embedding-ada-002", dimension=TEST_DIMENSION)


def _vector(seed: float = 1.0) -> list[float]:
    return [seed] + [0.0] * (TEST_DIMENSION - 1)


def _piece(ordinal: int = 0, content: str = "cláusula de rescisão", seed: float = 1.0) -> Piece:
    return Piece(
        ordinal=ordinal,
        content=content,
        embedding=_vector(seed),
        page_start=ordinal + 1,
        page_end=ordinal + 1,
        heading_path=["Contrato", "Rescisão"],
    )


def _count_pieces(conn: psycopg.Connection, document_id: int, stamp: ModelStamp) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM chunks WHERE document_id = %s AND embed_model = %s AND embed_dimension = %s",
            (document_id, stamp.model, stamp.dimension),
        )
        (count,) = cursor.fetchone()
    return int(count)


def test_the_same_document_seen_twice_keeps_one_row(conn: psycopg.Connection) -> None:
    first = upsert_document(conn, external_id="drive-1", title="Contrato A")
    second = upsert_document(conn, external_id="drive-1", title="Contrato A (revisado)")
    assert first == second
    with conn.cursor() as cursor:
        cursor.execute("SELECT title FROM documents WHERE id = %s", (first,))
        assert cursor.fetchone()[0] == "Contrato A (revisado)"


def test_a_parsed_artifact_already_recorded_is_not_erased_by_a_later_write(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-2", parsed_uri="gs://bucket/parsed/2.json")
    upsert_document(conn, external_id="drive-2", title="Contrato B")
    with conn.cursor() as cursor:
        cursor.execute("SELECT parsed_uri FROM documents WHERE id = %s", (document_id,))
        assert cursor.fetchone()[0] == "gs://bucket/parsed/2.json"


def test_upsert_document_writes_the_record_columns(conn: psycopg.Connection) -> None:
    modified = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    document_id = upsert_document(
        conn,
        external_id="drive-record-1",
        media_type="application/pdf",
        link="https://drive.google.com/file/d/1",
        location=["folder-a", "folder-b"],
        modified_at=modified,
        size_bytes=4096,
    )
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT media_type, link, location, modified_at, size_bytes "
            "FROM documents WHERE id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
    assert row == ("application/pdf", "https://drive.google.com/file/d/1", ["folder-a", "folder-b"], modified, 4096)


def test_upsert_document_leaves_the_record_columns_alone_when_not_given(conn: psycopg.Connection) -> None:
    document_id = upsert_document(
        conn,
        external_id="drive-record-2",
        media_type="application/pdf",
        link="https://drive.google.com/file/d/2",
        location=["folder-a"],
        size_bytes=100,
    )
    upsert_document(conn, external_id="drive-record-2", title="renamed")
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT media_type, link, location, size_bytes FROM documents WHERE id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
    assert row == ("application/pdf", "https://drive.google.com/file/d/2", ["folder-a"], 100)


def test_upsert_document_stores_no_value_for_a_fact_the_source_never_gave(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-record-3", parsed_uri="gs://bucket/3.json")
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT media_type, link, location, modified_at, size_bytes FROM documents WHERE id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
    assert row == (None, None, [], None, None)


def test_pieces_are_written_with_their_locator_and_stamp(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-3")
    written = upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(1)])
    assert written == 2
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT ordinal, page_start, page_end, heading_path, embed_model, embed_dimension "
            "FROM chunks WHERE document_id = %s ORDER BY ordinal",
            (document_id,),
        )
        rows = cursor.fetchall()
    assert rows == [
        (0, 1, 1, ["Contrato", "Rescisão"], CURRENT.model, CURRENT.dimension),
        (1, 2, 2, ["Contrato", "Rescisão"], CURRENT.model, CURRENT.dimension),
    ]


def test_writing_the_same_pieces_twice_leaves_the_same_set(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-4")
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(1)])
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(1)])
    assert _count_pieces(conn, document_id, CURRENT) == 2


def test_rewriting_a_document_that_shrank_drops_the_pieces_that_no_longer_exist(
    conn: psycopg.Connection,
) -> None:
    document_id = upsert_document(conn, external_id="drive-5")
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(1), _piece(2)])
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0)])
    assert _count_pieces(conn, document_id, CURRENT) == 1


def test_writing_no_pieces_clears_that_stamp_and_nothing_else(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-6")
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0)])
    upsert_pieces(conn, document_id=document_id, stamp=PREVIOUS, pieces=[_piece(0)])
    assert upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[]) == 0
    assert _count_pieces(conn, document_id, CURRENT) == 0
    assert _count_pieces(conn, document_id, PREVIOUS) == 1


def test_one_model_stamp_never_disturbs_another(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-7")
    upsert_pieces(conn, document_id=document_id, stamp=PREVIOUS, pieces=[_piece(0), _piece(1)])
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0)])
    assert _count_pieces(conn, document_id, PREVIOUS) == 2
    assert _count_pieces(conn, document_id, CURRENT) == 1


def test_the_store_never_commits_on_its_own(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-8")
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0)])
    conn.rollback()
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM documents WHERE external_id = 'drive-8'")
        assert cursor.fetchone()[0] == 0


def test_a_vector_of_the_wrong_width_names_the_stamp_it_disagrees_with(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-9")
    wrong = Piece(ordinal=0, content="x", embedding=[0.0] * (TEST_DIMENSION - 1))
    with pytest.raises(StoreError) as excinfo:
        upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[wrong])
    message = str(excinfo.value)
    assert CURRENT.model in message
    assert str(TEST_DIMENSION) in message


def test_two_pieces_claiming_the_same_position_are_refused(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-10")
    with pytest.raises(StoreError):
        upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(0)])


def test_a_rejected_batch_leaves_the_previous_pieces_intact(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-11")
    upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[_piece(0), _piece(1)])
    broken = [_piece(0), Piece(ordinal=1, content="x", embedding=[0.0])]
    with pytest.raises(StoreError):
        upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=broken)
    assert _count_pieces(conn, document_id, CURRENT) == 2


def test_the_store_does_not_import_the_embedding_module() -> None:
    source = (__import__("pathlib").Path("src/store.py")).read_text(encoding="utf-8")
    assert "src.embedding" not in source


def test_a_piece_with_no_locator_is_refused_at_write_time(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-12")
    unciteable = Piece(ordinal=0, content="cláusula sem origem", embedding=_vector())
    with pytest.raises(StoreError, match="P-002"):
        upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[unciteable])


def test_a_heading_path_alone_is_locator_enough_to_be_written(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-13")
    written = Piece(
        ordinal=0,
        content="cláusula sem páginas",
        embedding=_vector(),
        heading_path=["Contrato", "Rescisão"],
    )
    assert upsert_pieces(conn, document_id=document_id, stamp=CURRENT, pieces=[written]) == 1


def test_leaving_a_field_out_keeps_it_rather_than_clearing_it(conn: psycopg.Connection) -> None:
    document_id = upsert_document(
        conn,
        external_id="drive-14",
        title="Contrato C",
        parsed_uri="gs://bucket/parsed/14.json",
        metadata={"origem": "drive"},
    )
    upsert_document(conn, external_id="drive-14", title="Contrato C (revisado)")
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT title, parsed_uri, metadata FROM documents WHERE id = %s", (document_id,)
        )
        title, parsed_uri, metadata = cursor.fetchone()
    assert title == "Contrato C (revisado)"
    assert parsed_uri == "gs://bucket/parsed/14.json"
    assert metadata == {"origem": "drive"}


def test_an_empty_mapping_is_still_the_way_to_clear_the_metadata(conn: psycopg.Connection) -> None:
    document_id = upsert_document(conn, external_id="drive-15", metadata={"origem": "drive"})
    upsert_document(conn, external_id="drive-15", metadata={})
    with conn.cursor() as cursor:
        cursor.execute("SELECT metadata FROM documents WHERE id = %s", (document_id,))
        assert cursor.fetchone()[0] == {}
