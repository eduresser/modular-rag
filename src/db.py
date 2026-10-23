from __future__ import annotations

from collections.abc import Sequence

import psycopg

from migrations import load as load_migration

BASELINE_MIGRATION = "001_baseline"

DERIVED_MIGRATIONS: tuple[str, ...] = (
    "002_sync_runs",
    "003_usage",
    "004_search_events",
    "005_processing",
    "006_ingest",
    "007_sync_window",
    "008_sync_history",
    "009_position_identity",
    "010_unproven_counter",
    "011_document_record",
    "012_sync_requests",
    "013_artifacts",
    "014_process_module",
    "015_retired_counter",
    "016_filters",
)

# pgvector HNSW indexes half-precision vectors up to 4000 dims; beyond that every search is a sequential scan (ASM-041)
MAX_INDEXABLE_DIMENSION = 4000


class DatabaseUnavailable(Exception):
    pass


def connect(dsn: str) -> psycopg.Connection:
    try:
        return psycopg.connect(dsn)
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailable(f"could not reach the archive: {exc}".strip()) from exc


def create_schema(conn: psycopg.Connection, *, dimension: int) -> None:
    load_migration(BASELINE_MIGRATION).apply(conn, dimension=dimension)
    for name in DERIVED_MIGRATIONS:
        load_migration(name).apply(conn)


def validate_dimension(dimension: int) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValueError(f"embedding dimension must be an integer, got {dimension!r}")
    if not 1 <= dimension <= MAX_INDEXABLE_DIMENSION:
        raise ValueError(
            f"embedding dimension {dimension} is outside 1..{MAX_INDEXABLE_DIMENSION}; "
            "a wider column cannot carry an HNSW index, which would make every search a "
            "sequential scan over the whole corpus"
        )
    return dimension


def stored_dimension(conn: psycopg.Connection) -> int | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT atttypmod
              FROM pg_attribute
             WHERE attrelid = to_regclass('chunks')
               AND attname = 'embedding'
               AND NOT attisdropped
            """
        )
        row = cursor.fetchone()
    if row is None or row[0] is None or row[0] < 0:
        return None
    return int(row[0])


def halfvec_literal(vector: Sequence[float]) -> str:
    if not vector:
        raise ValueError("an embedding vector cannot be empty")
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"
