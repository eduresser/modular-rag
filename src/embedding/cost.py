from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from urllib.parse import urlparse

import psycopg

from src.embedding.batch import BatchOutcome, EmbeddingRun
from src.prices import OPERATION_EMBED, UNIT_TOKEN, PriceBook
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_QUESTION_ASKED,
    UsageError,
    priced,
    record_usage,
)

PROVIDER_UNKNOWN = "unknown"


def provider_of(base_url: str) -> str:
    host = urlparse(base_url).hostname or ""
    labels = [label for label in host.split(".") if label]
    if not labels:
        return PROVIDER_UNKNOWN
    return labels[-2] if len(labels) > 1 else labels[0]


def record_embedding(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    document_id: int | None,
    model: str,
    provider: str,
    tokens: int,
    reason: str,
    duration_ms: Decimal | int | float,
    outcome: str = OUTCOME_SUCCEEDED,
) -> int:
    if reason == REASON_QUESTION_ASKED:
        raise UsageError(
            "a document is never embedded because a question was asked; that reason "
            "belongs to a query embed, and using it here would make what indexing "
            "costs and what answering costs indistinguishable"
        )
    return record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_EMBED,
            reason=reason,
            provider=provider,
            model=model,
            unit=UNIT_TOKEN,
            units=tokens,
            duration_ms=duration_ms,
            outcome=outcome,
            document_id=document_id,
        ),
        client=client,
    )


def record_query_embedding(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    model: str,
    provider: str,
    tokens: int,
    duration_ms: Decimal | int | float,
    outcome: str = OUTCOME_SUCCEEDED,
) -> int:
    return record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_EMBED,
            reason=REASON_QUESTION_ASKED,
            provider=provider,
            model=model,
            unit=UNIT_TOKEN,
            units=tokens,
            duration_ms=duration_ms,
            outcome=outcome,
        ),
        client=client,
    )


def record_batch(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    outcome: BatchOutcome,
    model: str,
    provider: str,
    reason: str,
    document_ids: Mapping[str, int] | None = None,
) -> int:
    keys = set(outcome.batch.keys)
    document_id = None
    if len(keys) == 1 and document_ids:
        document_id = document_ids.get(next(iter(keys)))
    return record_embedding(
        conn,
        book,
        client=client,
        document_id=document_id,
        model=model,
        provider=provider,
        tokens=outcome.tokens,
        reason=reason,
        duration_ms=outcome.duration_ms,
        outcome=OUTCOME_SUCCEEDED if outcome.succeeded else OUTCOME_FAILED,
    )


def record_run(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    run: EmbeddingRun,
    model: str,
    provider: str,
    reason: str,
    document_ids: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    return tuple(
        record_batch(
            conn,
            book,
            client=client,
            outcome=outcome,
            model=model,
            provider=provider,
            reason=reason,
            document_ids=document_ids,
        )
        for outcome in run.outcomes
    )
