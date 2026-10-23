from __future__ import annotations

import psycopg
import pytest

from src.db import MAX_INDEXABLE_DIMENSION, create_schema, stored_dimension, validate_dimension
from src.sync_ledger import read_run, start_run
from tests.conftest import TEST_DIMENSION


def test_sync_requests_reach_an_existing_archive_and_survive_reapplying_migrations(conn):
    from src.sync_queue import enqueue, read_request

    conn.execute("DROP TABLE sync_requests")
    document_id = conn.execute(
        "INSERT INTO documents (external_id, title) VALUES ('before-sync-api', 'Contrato preservado') RETURNING id"
    ).fetchone()[0]
    create_schema(conn, dimension=TEST_DIMENSION)
    request = enqueue(conn)
    create_schema(conn, dimension=TEST_DIMENSION)
    assert read_request(conn, request.id) == request
    assert conn.execute("SELECT title FROM documents WHERE id = %s", (document_id,)).fetchone()[0] == "Contrato preservado"


def test_vector_column_is_created_for_an_explicit_dimension(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
        (declared,) = cursor.fetchone()
    assert declared == f"halfvec({TEST_DIMENSION})"
    assert stored_dimension(conn) == TEST_DIMENSION


def test_stored_dimension_is_none_before_any_schema_exists(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn) as fresh:
        with fresh.cursor() as cursor:
            cursor.execute("CREATE SCHEMA empty_boot")
            cursor.execute("SET search_path TO empty_boot")
            assert stored_dimension(fresh) is None
        fresh.rollback()


@pytest.mark.parametrize("rejected", [0, -1, MAX_INDEXABLE_DIMENSION + 1, True, 1.5, "1536"])
def test_a_dimension_that_cannot_be_indexed_is_refused(rejected: object) -> None:
    with pytest.raises(ValueError):
        validate_dimension(rejected)  # type: ignore[arg-type]


def test_the_embedding_index_is_hnsw_over_cosine_distance(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_idx'")
        (definition,) = cursor.fetchone()
    assert "USING hnsw" in definition
    assert "halfvec_cosine_ops" in definition


def test_the_text_search_column_is_configured_for_portuguese(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO documents (external_id) VALUES ('doc-pt') RETURNING id"
        )
        (document_id,) = cursor.fetchone()
        cursor.execute(
            "INSERT INTO chunks (document_id, ordinal, content, embed_model, embed_dimension, embedding) "
            "VALUES (%s, 0, %s, 'm', %s, %s::halfvec) RETURNING search_text",
            (document_id, "As rescisões antecipadas do contrato", TEST_DIMENSION, "[" + ",".join(["0"] * TEST_DIMENSION) + "]"),
        )
        (search_text,) = cursor.fetchone()
    assert "rescis" in search_text
    assert "rescisões" not in search_text


def test_a_document_cannot_hold_an_invented_state(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO documents (external_id, state) VALUES ('doc-bad', 'deleted')")


def test_a_piece_cannot_claim_a_dimension_the_column_cannot_hold(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO documents (external_id) VALUES ('doc-dim') RETURNING id")
        (document_id,) = cursor.fetchone()
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chunks (document_id, ordinal, content, embed_model, embed_dimension, embedding) "
                    "VALUES (%s, 0, 'x', 'm', %s, %s::halfvec)",
                    (document_id, TEST_DIMENSION + 1, "[" + ",".join(["0"] * TEST_DIMENSION) + "]"),
                )


def test_the_active_model_pointer_can_only_ever_hold_one_row(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO active_model_version (embed_model, embed_dimension) VALUES ('m', %s)",
            (TEST_DIMENSION,),
        )
    with pytest.raises((psycopg.errors.UniqueViolation, psycopg.errors.CheckViolation)):
        with conn.transaction():
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO active_model_version (only_row, embed_model, embed_dimension) "
                    "VALUES (false, 'other', %s)",
                    (TEST_DIMENSION,),
                )


def test_the_document_record_columns_exist_with_the_right_shape(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_name = 'documents' "
            "AND column_name IN ('media_type', 'link', 'location', 'modified_at', 'size_bytes') "
            "ORDER BY column_name"
        )
        rows = {row[0]: row[1:] for row in cursor.fetchall()}
    assert rows["media_type"][:2] == ("text", "YES")
    assert rows["link"][:2] == ("text", "YES")
    assert rows["modified_at"][:2] == ("timestamp with time zone", "YES")
    assert rows["size_bytes"][:2] == ("bigint", "YES")
    assert rows["location"][:2] == ("ARRAY", "NO")
    assert rows["location"][2] is not None


def test_the_document_record_columns_default_to_absent(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO documents (external_id) VALUES ('doc-record') RETURNING id")
        (document_id,) = cursor.fetchone()
        cursor.execute(
            "SELECT media_type, link, location, modified_at, size_bytes "
            "FROM documents WHERE id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
    assert row == (None, None, [], None, None)


def test_an_archive_created_before_the_document_record_still_gets_it(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn) as older:
        with older.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS older_documents")
            cursor.execute("SET search_path TO older_documents, public")
            cursor.execute(
                """
                CREATE TABLE documents (
                    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    external_id   text        NOT NULL UNIQUE,
                    title         text        NOT NULL DEFAULT '',
                    state         text        NOT NULL DEFAULT 'indexed',
                    parsed_uri    text,
                    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
                    first_seen_at timestamptz NOT NULL DEFAULT now(),
                    updated_at    timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            from migrations import load as load_migration

            load_migration("011_document_record").apply(older)
            cursor.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'documents' AND column_name = 'media_type'"
            )
            assert cursor.fetchone() is not None
        older.rollback()


def test_no_table_carries_a_tenant_column(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND column_name IN ('tenant', 'tenant_id', 'client_id', 'account_id', 'org_id')"
        )
        assert cursor.fetchall() == []


def test_applying_the_baseline_twice_changes_nothing(conn: psycopg.Connection) -> None:
    with conn.transaction():
        create_schema(conn, dimension=TEST_DIMENSION)
    assert stored_dimension(conn) == TEST_DIMENSION


def test_predeploy_command_initialises_a_fresh_database_and_preserves_existing_rows(postgres_dsn, monkeypatch, tmp_path):
    from migrations.__main__ import main
    from tests.conftest import INGEST_DEPLOYMENT
    import yaml

    database = "migration_test"
    dsn = postgres_dsn.rsplit("/", 1)[0] + "/" + database
    with psycopg.connect(postgres_dsn, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    try:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**INGEST_DEPLOYMENT, "db": {"url_secret": "env:DATABASE_URL"}}))
        monkeypatch.setenv("DATABASE_URL", dsn)
        assert main(path=str(path)) == 0
        with psycopg.connect(dsn) as connection:
            assert stored_dimension(connection) == TEST_DIMENSION
            connection.execute("INSERT INTO documents (external_id, title) VALUES ('kept', 'Existing contract')")
        assert main(path=str(path)) == 0
        with psycopg.connect(dsn) as connection:
            assert connection.execute("SELECT title FROM documents WHERE external_id = 'kept'").fetchone() == ("Existing contract",)
            assert connection.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone() == ("vector",)
    finally:
        with psycopg.connect(postgres_dsn, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database} WITH (FORCE)")


def test_an_archive_created_before_a_counter_existed_still_gets_it(postgres_dsn: str) -> None:
    with psycopg.connect(postgres_dsn) as older:
        with older.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS older_archive")
            cursor.execute("SET search_path TO older_archive, public")
            cursor.execute(
                """
                CREATE TABLE sync_runs (
                    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    client            text        NOT NULL,
                    kind              text        NOT NULL,
                    status            text        NOT NULL DEFAULT 'running',
                    started_at        timestamptz NOT NULL DEFAULT now(),
                    finished_at       timestamptz,
                    documents_seen    integer     NOT NULL DEFAULT 0,
                    documents_added   integer     NOT NULL DEFAULT 0,
                    documents_updated integer     NOT NULL DEFAULT 0,
                    documents_skipped integer     NOT NULL DEFAULT 0,
                    documents_removed integer     NOT NULL DEFAULT 0,
                    documents_errored integer     NOT NULL DEFAULT 0,
                    access_lost       integer     NOT NULL DEFAULT 0,
                    sweep_completed   boolean     NOT NULL DEFAULT false,
                    feed_position     text,
                    detail            text
                )
                """
            )
        older.commit()

        try:
            create_schema(older, dimension=TEST_DIMENSION)
            older.commit()

            run_id = start_run(older, client="example", kind="sweep")
            assert read_run(older, run_id).counters.documents_unproven == 0
        finally:
            older.rollback()
            with older.cursor() as cursor:
                cursor.execute("DROP SCHEMA IF EXISTS older_archive CASCADE")
            older.commit()
