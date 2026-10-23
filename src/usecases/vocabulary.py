from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.filters import FIELDS, OPERATORS


@dataclass(frozen=True, slots=True)
class FieldValue:

    value: str
    document_count: int


@dataclass(frozen=True, slots=True)
class PublishedField:

    name: str
    type: str
    operators: tuple[str, ...]
    values: tuple[FieldValue, ...] | None = None
    values_open: bool = False


_VALUE_QUERIES: dict[str, str] = {
    "media_type": """
        SELECT media_type, count(*)
          FROM documents
         WHERE state <> 'gone' AND media_type IS NOT NULL
         GROUP BY media_type
    """,
    "process_module": """
        SELECT di.process_module, count(*)
          FROM document_ingest di
          JOIN documents d ON d.id = di.document_id
         WHERE d.state <> 'gone' AND di.process_module IS NOT NULL
         GROUP BY di.process_module
    """,
    "block_kind": """
        SELECT value, count(DISTINCT c.document_id)
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
         CROSS JOIN LATERAL jsonb_array_elements_text(
                                 coalesce(c.metadata -> 'block_kinds', '[]'::jsonb)
                             ) AS value
         WHERE d.state <> 'gone'
         GROUP BY value
    """,
}


def _counted_values(conn: psycopg.Connection, field_name: str) -> tuple[FieldValue, ...]:
    query = _VALUE_QUERIES[field_name]
    with conn.cursor() as cursor:
        cursor.execute(query)  # type: ignore[arg-type]
        rows = cursor.fetchall()
    return tuple(
        FieldValue(value=value, document_count=count)
        for value, count in sorted(rows, key=lambda row: row[0])
    )


def published_vocabulary(conn: psycopg.Connection) -> tuple[PublishedField, ...]:
    return tuple(
        PublishedField(
            name=spec.name,
            type=spec.type,
            operators=OPERATORS[spec.type],
            values=_counted_values(conn, spec.name) if spec.published else None,
            values_open=spec.open_values,
        )
        for spec in FIELDS.values()
    )
