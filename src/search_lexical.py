from __future__ import annotations

from typing import Any

import psycopg

from src.filters import DEFAULT_FILTER_TIMEOUT_MS, FilterNode, compile_filter
from src.search_result import Candidate, Provenance
from src.store import DOCUMENT_STATE_GONE, ModelStamp

TEXT_SEARCH_CONFIGURATION = "portuguese"


def _named_filter(compiled: tuple[str, list[Any]], prefix: str) -> tuple[str, dict[str, Any]]:
    fragment, params = compiled
    parts = fragment.split("%s")
    pieces = [parts[0]]
    named: dict[str, Any] = {}
    for index, (part, value) in enumerate(zip(parts[1:], params)):
        name = f"{prefix}{index}"
        named[name] = value
        pieces.append(f"%({name})s")
        pieces.append(part)
    return "".join(pieces), named


_CANDIDATES = """
    WITH question AS (
        SELECT websearch_to_tsquery(%(configuration)s, %(question)s) AS tsquery
    )
    SELECT c.id,
           c.content,
           c.page_start,
           c.page_end,
           c.heading_path,
           d.id,
           d.external_id,
           d.title,
           ts_rank_cd(c.search_text, question.tsquery) AS rank_score
      FROM chunks c
      JOIN documents d ON d.id = c.document_id
     CROSS JOIN question
     WHERE d.state <> %(gone)s
       AND c.embed_model = %(model)s
       AND c.embed_dimension = %(dimension)s
       AND c.search_text @@ question.tsquery
       {filter_sql}
     ORDER BY rank_score DESC, c.id
     LIMIT %(depth)s
"""


def lexical_candidates(
    conn: psycopg.Connection,
    *,
    question: str,
    stamp: ModelStamp,
    depth: int,
    filter_: FilterNode | None = None,
    filter_timeout_ms: int = DEFAULT_FILTER_TIMEOUT_MS,
) -> list[Candidate]:
    if depth < 1:
        raise ValueError(f"candidate depth must be at least 1, got {depth}")

    params: dict[str, Any] = {
        "configuration": TEXT_SEARCH_CONFIGURATION,
        "question": question,
        "gone": DOCUMENT_STATE_GONE,
        "model": stamp.model,
        "dimension": stamp.dimension,
        "depth": depth,
    }
    compiled = compile_filter(filter_)
    if compiled is None:
        filter_sql = ""
    else:
        fragment, filter_params = _named_filter(compiled, "filter")
        filter_sql = f"AND ({fragment})"
        params.update(filter_params)

    with conn.cursor() as cursor:
        if filter_ is not None:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)", (str(filter_timeout_ms),)
            )
        cursor.execute(_CANDIDATES.format(filter_sql=filter_sql), params)
        rows = cursor.fetchall()
    return [
        Candidate(
            chunk_id=row[0],
            content=row[1],
            provenance=Provenance(
                document_id=row[5],
                external_id=row[6],
                title=row[7],
                page_start=row[2],
                page_end=row[3],
                heading_path=tuple(row[4]),
            ),
            rank=position,
            leg_score=float(row[8]),
        )
        for position, row in enumerate(rows, start=1)
    ]
