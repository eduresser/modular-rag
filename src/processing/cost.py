from __future__ import annotations

from decimal import Decimal

import psycopg

from src.prices import OPERATION_READ, OPERATION_CHUNK, UNIT_PAGE, PriceBook
from src.processing.recut import RECUT_LOCAL, RECUT_PROVIDER
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_QUESTION_ASKED,
    UsageError,
    priced,
    record_usage,
)

PROVIDER_READER = "google"
PROVIDER_LOCAL = "internal"

MODEL_LOCAL_RECUT = "local"


def record_read(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    document_id: int | None,
    reader_version: str,
    pages: int,
    reason: str,
    duration_ms: Decimal | int | float,
    outcome: str = OUTCOME_SUCCEEDED,
    provider: str = PROVIDER_READER,
) -> int:
    _reject_question_reason(reason, OPERATION_READ)
    return record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_READ,
            reason=reason,
            provider=provider,
            model=reader_version,
            unit=UNIT_PAGE,
            units=pages,
            duration_ms=duration_ms,
            outcome=outcome,
            document_id=document_id,
        ),
        client=client,
    )


def record_recut(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    document_id: int | None,
    reader_version: str,
    pages: int,
    reason: str,
    duration_ms: Decimal | int | float,
    path: str = RECUT_LOCAL,
    outcome: str = OUTCOME_SUCCEEDED,
) -> int:
    _reject_question_reason(reason, OPERATION_CHUNK)
    if path not in (RECUT_LOCAL, RECUT_PROVIDER):
        raise UsageError(f"a re-cut happens locally or at the provider, not by {path!r}")
    provider = PROVIDER_READER if path == RECUT_PROVIDER else PROVIDER_LOCAL
    model = reader_version if path == RECUT_PROVIDER else MODEL_LOCAL_RECUT
    return record_usage(
        conn,
        priced(
            book,
            operation=OPERATION_CHUNK,
            reason=reason,
            provider=provider,
            model=model,
            unit=UNIT_PAGE,
            units=pages,
            duration_ms=duration_ms,
            outcome=outcome,
            document_id=document_id,
        ),
        client=client,
    )


def record_refused_read(
    conn: psycopg.Connection,
    book: PriceBook,
    *,
    client: str,
    document_id: int | None,
    reader_version: str,
    reason: str,
    duration_ms: Decimal | int | float = 0,
) -> int:
    return record_read(
        conn,
        book,
        client=client,
        document_id=document_id,
        reader_version=reader_version,
        pages=0,
        reason=reason,
        duration_ms=duration_ms,
        outcome=OUTCOME_FAILED,
    )


def _reject_question_reason(reason: str, operation: str) -> None:
    if reason == REASON_QUESTION_ASKED:
        raise UsageError(
            f"{operation!r} never happens because a question was asked; that reason belongs "
            "to a search or a rerank, and using it here would make query cost and ingestion "
            "cost indistinguishable"
        )
