from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg

from src.db import halfvec_literal
from src.filters import DEFAULT_FILTER_TIMEOUT_MS, FilterNode, compile_filter
from src.search_result import Candidate, Provenance
from src.store import DOCUMENT_STATE_GONE, ModelStamp

MAX_MATCH_DISTANCE = 1.0


def _similarity(distance: float) -> float:
    return 1.0 - float(distance)


_MIN_MAX_SCAN_TUPLES = 20_000
_FILTER_SELECTIVITY_FLOOR = 0.02


def _max_scan_tuples(depth: int) -> int:
    return max(int(depth / _FILTER_SELECTIVITY_FLOOR), _MIN_MAX_SCAN_TUPLES)


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
    SELECT c.id,
           c.content,
           c.page_start,
           c.page_end,
           c.heading_path,
           d.id,
           d.external_id,
           d.title,
           c.embedding <=> %(vector)s::halfvec AS distance
      FROM chunks c
      JOIN documents d ON d.id = c.document_id
     WHERE d.state <> %(gone)s
       AND c.embed_model = %(model)s
       AND c.embed_dimension = %(dimension)s
       AND c.embedding <=> %(vector)s::halfvec < %(max_distance)s
       {filter_sql}
     ORDER BY c.embedding <=> %(vector)s::halfvec
     LIMIT %(depth)s
"""


def dense_candidates(
    conn: psycopg.Connection,
    *,
    query_vector: Sequence[float],
    stamp: ModelStamp,
    depth: int,
    index_search_effort: int,
    filter_: FilterNode | None = None,
    filter_timeout_ms: int = DEFAULT_FILTER_TIMEOUT_MS,
) -> list[Candidate]:
    if len(query_vector) != stamp.dimension:
        raise ValueError(
            f"the question was embedded into {len(query_vector)} dimensions but the archive's "
            f"active model {stamp.model} stamps {stamp.dimension} — the question and the corpus "
            "were embedded by different models"
        )
    if depth < 1:
        raise ValueError(f"candidate depth must be at least 1, got {depth}")
    vector = halfvec_literal(query_vector)

    params: dict[str, Any] = {
        "vector": vector,
        "gone": DOCUMENT_STATE_GONE,
        "model": stamp.model,
        "dimension": stamp.dimension,
        "depth": depth,
        "max_distance": MAX_MATCH_DISTANCE,
    }
    compiled = compile_filter(filter_)
    if compiled is None:
        filter_sql = ""
    else:
        fragment, filter_params = _named_filter(compiled, "filter")
        filter_sql = f"AND ({fragment})"
        params.update(filter_params)

    with conn.cursor() as cursor:
        cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(index_search_effort),))
        if filter_ is not None:
            cursor.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            cursor.execute(
                "SELECT set_config('hnsw.max_scan_tuples', %s, true)",
                (str(_max_scan_tuples(depth)),),
            )
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
            leg_score=_similarity(row[8]),
        )
        for position, row in enumerate(rows, start=1)
    ]
