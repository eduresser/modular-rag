from __future__ import annotations

import migrations
from src.db import DERIVED_MIGRATIONS


def test_016_is_appended_to_both_registries():
    assert migrations.ORDER[-1] == "016_filters"
    assert DERIVED_MIGRATIONS[-1] == "016_filters"
    assert migrations.ORDER.index("016_filters") == len(migrations.ORDER) - 1


def test_filter_column_is_nullable_and_defaults_to_null(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_name = 'search_events' AND column_name = 'filter'"
        )
        row = cursor.fetchone()
    assert row is not None
    is_nullable, column_default = row
    assert is_nullable == "YES"
    assert column_default is None


def test_pg_trgm_extension_is_installed(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        assert cursor.fetchone() is not None


def test_the_five_filter_indexes_exist_with_the_right_operator_class(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE indexname IN ("
            "  'documents_title_trgm_idx', 'documents_location_idx',"
            "  'chunks_heading_path_idx', 'chunks_metadata_idx', 'chunks_page_range_idx'"
            ")"
        )
        rows = dict(cursor.fetchall())

    assert set(rows) == {
        "documents_title_trgm_idx",
        "documents_location_idx",
        "chunks_heading_path_idx",
        "chunks_metadata_idx",
        "chunks_page_range_idx",
    }
    assert "gin_trgm_ops" in rows["documents_title_trgm_idx"]
    assert "USING gin" in rows["documents_location_idx"]
    assert "USING gin" in rows["chunks_heading_path_idx"]
    assert "jsonb_path_ops" in rows["chunks_metadata_idx"]
    assert "btree" in rows["chunks_page_range_idx"]
    assert "page_start" in rows["chunks_page_range_idx"]
    assert "page_end" in rows["chunks_page_range_idx"]


def test_applying_016_twice_is_a_no_op(conn):
    migrations.load("016_filters").apply(conn)
    migrations.load("016_filters").apply(conn)
