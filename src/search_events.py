from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import psycopg
from psycopg.types.json import Json

SEARCH_OUTCOME_FOUND = "found"
SEARCH_OUTCOME_NOTHING_MATCHED = "nothing_matched"
SEARCH_OUTCOME_FAILED = "failed"

SEARCH_OUTCOMES: tuple[str, ...] = (
    SEARCH_OUTCOME_FOUND,
    SEARCH_OUTCOME_NOTHING_MATCHED,
    SEARCH_OUTCOME_FAILED,
)

_WHITESPACE = re.compile(r"\s+")


class SearchEventError(Exception):
    pass


def question_digest(question: str) -> str:
    normalised = _WHITESPACE.sub(" ", question).strip().casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SearchEvent:
    caller: str
    question: str
    dense_weight: float
    lexical_weight: float
    candidate_depth: int
    result_count: int
    index_search_effort: int
    duration_ms: Decimal | float
    outcome: str
    document_ids: Sequence[int] = field(default_factory=tuple)
    chunk_ids: Sequence[int] = field(default_factory=tuple)
    rerank_model: str | None = None
    occurred_at: datetime | None = None
    filter: Mapping | None = None
    """The filter as it was applied, or ``None`` for an unfiltered search
    (AC-291). Never ``{}`` for "no filter" — the two are different requests
    and the record has to keep telling them apart."""

    def __post_init__(self) -> None:
        if not self.caller.strip():
            raise SearchEventError(
                "a search record has to name its caller; 'which application is asking' is "
                "the first question asked of this table and it cannot be reconstructed later"
            )
        if self.outcome not in SEARCH_OUTCOMES:
            raise SearchEventError(
                f"outcome {self.outcome!r} is not one of {', '.join(SEARCH_OUTCOMES)}; "
                "a search that matched nothing and a search that could not run are "
                "different things (principle P-006)"
            )
        if self.outcome == SEARCH_OUTCOME_FOUND and not self.chunk_ids:
            raise SearchEventError(
                f"outcome {SEARCH_OUTCOME_FOUND!r} with nothing returned is really "
                f"{SEARCH_OUTCOME_NOTHING_MATCHED!r}"
            )
        if self.candidate_depth < self.result_count:
            raise SearchEventError(
                f"candidate depth {self.candidate_depth} is below the {self.result_count} "
                "results asked for, so the fusion had nothing to fuse"
            )
        if self.index_search_effort < 1:
            raise SearchEventError(f"index search effort must be at least 1, got {self.index_search_effort}")
        if Decimal(str(self.duration_ms)) < 0:
            raise SearchEventError(f"a duration cannot be negative, got {self.duration_ms}")


@dataclass(frozen=True, slots=True)
class SearchRecord:
    id: int
    client: str
    caller: str
    question_digest: str
    question_text: str | None
    dense_weight: Decimal
    lexical_weight: Decimal
    candidate_depth: int
    result_count: int
    index_search_effort: int
    document_ids: tuple[int, ...]
    chunk_ids: tuple[int, ...]
    rerank_model: str | None
    duration_ms: Decimal
    outcome: str
    occurred_at: datetime
    filter: dict | None


_COLUMNS = """
    id, client, caller, question_digest, question_text, dense_weight, lexical_weight,
    candidate_depth, result_count, index_search_effort, document_ids, chunk_ids,
    rerank_model, duration_ms, outcome, occurred_at, filter
"""


def record_search(
    conn: psycopg.Connection,
    event: SearchEvent,
    *,
    client: str,
    record_question_text: bool = False,
) -> int:
    if not client.strip():
        raise SearchEventError(
            "a search record has to name the client whose deployment produced it (AC-168)"
        )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO search_events (
                client, caller, question_digest, question_text, dense_weight, lexical_weight,
                candidate_depth, result_count, index_search_effort, document_ids, chunk_ids,
                rerank_model, duration_ms, outcome, occurred_at, filter
            ) VALUES (
                %(client)s, %(caller)s, %(digest)s, %(text)s, %(dense)s, %(lexical)s,
                %(depth)s, %(count)s, %(effort)s, %(documents)s, %(chunks)s,
                %(rerank)s, %(duration)s, %(outcome)s, COALESCE(%(occurred_at)s, now()), %(filter)s
            ) RETURNING id
            """,
            {
                "client": client,
                "caller": event.caller,
                "digest": question_digest(event.question),
                "text": event.question if record_question_text else None,
                "dense": event.dense_weight,
                "lexical": event.lexical_weight,
                "depth": event.candidate_depth,
                "count": event.result_count,
                "effort": event.index_search_effort,
                "documents": sorted({int(value) for value in event.document_ids}),
                "chunks": [int(value) for value in event.chunk_ids],
                "rerank": event.rerank_model,
                "duration": Decimal(str(event.duration_ms)),
                "outcome": event.outcome,
                "occurred_at": event.occurred_at,
                "filter": Json(dict(event.filter)) if event.filter is not None else None,
            },
        )
        (event_id,) = cursor.fetchone()
    return int(event_id)


def read_search(conn: psycopg.Connection, event_id: int) -> SearchRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT {_COLUMNS} FROM search_events WHERE id = %s", (event_id,))
        row = cursor.fetchone()
    if row is None:
        return None
    return SearchRecord(
        id=int(row[0]),
        client=row[1],
        caller=row[2],
        question_digest=row[3],
        question_text=row[4],
        dense_weight=row[5],
        lexical_weight=row[6],
        candidate_depth=int(row[7]),
        result_count=int(row[8]),
        index_search_effort=int(row[9]),
        document_ids=tuple(int(value) for value in row[10]),
        chunk_ids=tuple(int(value) for value in row[11]),
        rerank_model=row[12],
        duration_ms=row[13],
        outcome=row[14],
        occurred_at=row[15],
        filter=row[16],
    )
