from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx
import psycopg

from src.db import DatabaseUnavailable
from src.embedding.batch import BatchError
from src.embedding.client import EmbeddingError
from src.embedding.pinning import ProviderSubstituted
from src.embedding.reembed import ReembedError
from src.models import ChunkError
from src.processing.artifact import ArtifactError, ChunksBeforeArtifact, record_read_failure
from src.processing.pdftext import PdfTextError
from src.processing.recut import RecutError
from src.processing.to_chunks import MappingError
from src.store import StoreError
from src.usage import UsageError
from src.versions import UnknownStageVersion
from migrations import load as load_migration

_migration = load_migration("006_ingest")

FAILURE_PERMANENT: str = _migration.FAILURE_PERMANENT
FAILURE_TRANSIENT: str = _migration.FAILURE_TRANSIENT
FAILURE_KINDS: tuple[str, ...] = _migration.FAILURE_KINDS

# Which step was running. Recorded on the failure because "what are we failing
# at" is the first question, and an exception type answers it only by accident.
STAGE_FETCH = "fetch"
STAGE_READ = "read"
STAGE_CUT = "cut"
STAGE_EMBED = "embed"
STAGE_STORE = "store"

STAGES: tuple[str, ...] = (STAGE_FETCH, STAGE_READ, STAGE_CUT, STAGE_EMBED, STAGE_STORE)


class ReadRefused(Exception):
    pass


class NothingToIndex(Exception):
    pass


# Nothing another attempt can change. Listed before the transient ones because
# ``ProviderSubstituted`` is an ``EmbeddingError``, and the first match wins.
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (
    # The reader's own answer about this document, and a document that produced
    # nothing to index. Both are about the document, not about the moment.
    ReadRefused,
    NothingToIndex,
    # The router served a model other than the one that was pinned. Retrying
    # asks the same router the same question; somebody has to look.
    ProviderSubstituted,
    # The reader's output could not be turned into a citable piece, or pieces
    # were about to be derived with no kept artifact (principle P-009).
    ChunksBeforeArtifact,
    MappingError,
    ChunkError,
    StoreError,
    RecutError,
    BatchError,
    ReembedError,
    UsageError,
    # Data written under a version this build no longer declares (ASM-053).
    UnknownStageVersion,
    PdfTextError,
)

# Worth another run, on its own, at no cost when nothing is wrong.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    DatabaseUnavailable,
    ArtifactError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)

# Status codes that mean "later", not "no".
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FailureRecord:
    external_id: str
    stage: str
    kind: str
    reason: str
    document_id: int | None = None

    @property
    def retryable(self) -> bool:
        return self.kind == FAILURE_TRANSIENT


@dataclass
class FailureLog:
    failures: list[FailureRecord] = field(default_factory=list)

    def add(self, failure: FailureRecord) -> FailureRecord:
        self.failures.append(failure)
        return failure

    @property
    def count(self) -> int:
        return len(self.failures)

    @property
    def documents(self) -> tuple[str, ...]:
        return tuple(failure.external_id for failure in self.failures)

    @property
    def permanent(self) -> tuple[FailureRecord, ...]:
        return tuple(failure for failure in self.failures if not failure.retryable)

    @property
    def transient(self) -> tuple[FailureRecord, ...]:
        return tuple(failure for failure in self.failures if failure.retryable)


def classify(error: BaseException) -> str:
    if isinstance(error, PERMANENT_ERRORS):
        return FAILURE_PERMANENT
    if isinstance(error, TRANSIENT_ERRORS):
        return FAILURE_TRANSIENT
    if isinstance(error, httpx.HTTPStatusError):
        return _classify_status(error.response.status_code)
    if hasattr(error, "status"):
        reported = error.status
        return _classify_status(reported) if isinstance(reported, int) else FAILURE_TRANSIENT
    status = _status_in(str(error))
    if status is not None:
        return _classify_status(status)
    return FAILURE_TRANSIENT


def describe(error: BaseException, *, stage: str) -> str:
    message = str(error).strip() or error.__class__.__name__
    return f"{stage}: {message} ({error.__class__.__name__})"


def mark_failed(
    conn: psycopg.Connection,
    *,
    external_id: str,
    stage: str,
    error: BaseException,
    document_id: int | None = None,
    reader: str | None = None,
    reader_version: str | None = None,
) -> FailureRecord:
    failure = failure_of(
        external_id=external_id, stage=stage, error=error, document_id=document_id
    )
    if stage == STAGE_READ and document_id is not None and reader and reader_version:
        record_read_failure(
            conn,
            document_id=document_id,
            reason=failure.reason,
            reader=reader,
            reader_version=reader_version,
        )
    return failure


@contextmanager
def isolate(
    log: FailureLog,
    *,
    external_id: str,
    stage: str,
    on_failure: Callable[[FailureRecord], None] | None = None,
) -> Iterator[None]:
    try:
        yield
    except Exception as error:  # noqa: BLE001 — converting to a record is the job
        failure = log.add(failure_of(external_id=external_id, stage=stage, error=error))
        if on_failure is not None:
            on_failure(failure)


def failure_of(
    *,
    external_id: str,
    stage: str,
    error: BaseException,
    document_id: int | None = None,
) -> FailureRecord:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
    return FailureRecord(
        external_id=external_id,
        stage=stage,
        kind=classify(error),
        reason=describe(error, stage=stage),
        document_id=document_id,
    )


def _classify_status(status: int) -> str:
    if status in RETRYABLE_STATUS_CODES:
        return FAILURE_TRANSIENT
    if 400 <= status < 500:
        return FAILURE_PERMANENT
    return FAILURE_TRANSIENT


def _status_in(message: str) -> int | None:
    for token in message.replace(",", " ").replace(":", " ").split():
        if token.isdigit() and len(token) == 3 and 400 <= int(token) < 600:
            return int(token)
    return None
