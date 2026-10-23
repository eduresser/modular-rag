from __future__ import annotations

import psycopg

from src.db import validate_dimension

DOCUMENT_STATES: tuple[str, ...] = ("indexed", "gone")


def statements(dimension: int) -> tuple[str, ...]:
    width = validate_dimension(dimension)
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS documents (
            id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            external_id   text        NOT NULL UNIQUE,
            title         text        NOT NULL DEFAULT '',
            state         text        NOT NULL DEFAULT 'indexed',
            parsed_uri    text,
            metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
            first_seen_at timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT documents_state_known CHECK (state IN ('indexed', 'gone'))
        )
        """,
        "CREATE INDEX IF NOT EXISTS documents_state_idx ON documents (state)",
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id     bigint      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            ordinal         integer     NOT NULL,
            content         text        NOT NULL,
            page_start      integer,
            page_end        integer,
            heading_path    text[]      NOT NULL DEFAULT '{{}}',
            embed_model     text        NOT NULL,
            embed_dimension integer     NOT NULL,
            embedding       halfvec({width}) NOT NULL,
            metadata        jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
            created_at      timestamptz NOT NULL DEFAULT now(),
            search_text     tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED,
            CONSTRAINT chunks_dimension_matches_column CHECK (embed_dimension = {width}),
            CONSTRAINT chunks_pages_ordered
                CHECK (page_start IS NULL OR page_end IS NULL OR page_end >= page_start),
            CONSTRAINT chunks_one_piece_per_version
                UNIQUE (document_id, ordinal, embed_model, embed_dimension)
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id)",
        "CREATE INDEX IF NOT EXISTS chunks_version_idx ON chunks (embed_model, embed_dimension)",
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding halfvec_cosine_ops)
        """,
        "CREATE INDEX IF NOT EXISTS chunks_search_text_idx ON chunks USING gin (search_text)",
        """
        CREATE TABLE IF NOT EXISTS active_model_version (
            only_row        boolean     PRIMARY KEY DEFAULT true,
            embed_model     text        NOT NULL,
            embed_dimension integer     NOT NULL,
            activated_at    timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT active_model_version_is_a_single_row CHECK (only_row)
        )
        """,
    )


def apply(conn: psycopg.Connection, *, dimension: int) -> None:
    with conn.cursor() as cursor:
        for statement in statements(dimension):
            cursor.execute(statement)  # type: ignore[arg-type]
