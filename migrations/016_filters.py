from __future__ import annotations

import psycopg

STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE search_events ADD COLUMN IF NOT EXISTS filter jsonb",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS documents_title_trgm_idx ON documents USING gin (title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS documents_location_idx ON documents USING gin (location)",
    "CREATE INDEX IF NOT EXISTS chunks_heading_path_idx ON chunks USING gin (heading_path)",
    "CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING gin (metadata jsonb_path_ops)",
    "CREATE INDEX IF NOT EXISTS chunks_page_range_idx ON chunks (page_start, page_end)",
)


def apply(conn: psycopg.Connection) -> None:
    with conn.cursor() as cursor:
        for statement in STATEMENTS:
            cursor.execute(statement)
