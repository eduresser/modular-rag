from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import psycopg

from src.processing.artifact import record_read_failure
from src.processing.docai import BATCH_PAGE_LIMIT

BATCH_BYTE_LIMIT = 1_073_741_824

OUTCOME_READ = "read"
OUTCOME_REFUSED = "refused"

VERDICT_OUTCOMES: tuple[str, ...] = (OUTCOME_READ, OUTCOME_REFUSED)

SCANNED_MEDIA_TYPES: tuple[str, ...] = (
    "application/pdf",
    "image/gif",
    "image/tiff",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
)

OFFICE_MEDIA_TYPES: tuple[str, ...] = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
)

MEDIA_HTML = "text/html"

SUPPORTED_MEDIA_TYPES: tuple[str, ...] = SCANNED_MEDIA_TYPES + OFFICE_MEDIA_TYPES + (MEDIA_HTML,)

UNPAGINATED_MEDIA_TYPES: frozenset[str] = frozenset({OFFICE_MEDIA_TYPES[0]})

UNREADABLE_TEXT_MEDIA_TYPES: tuple[str, ...] = ("text/plain", "text/markdown", "text/csv")

PAGE_EQUIVALENTS: dict[str, str] = {
    OFFICE_MEDIA_TYPES[0]: "3,000 characters",
    OFFICE_MEDIA_TYPES[1]: "one tab",
    OFFICE_MEDIA_TYPES[2]: "one slide",
}

DOCAI_TEXT_REFUSAL = (
    "is {media_type}, and the reader does not accept text files — a surprising rule, and "
    "the provider's own: HTML and the office formats are supported and plain text is not. "
    "Convert it, or index it by a route that does not go through the reader"
)


@dataclass(frozen=True, slots=True)
class Candidate:

    document_id: int
    external_id: str
    media_type: str
    byte_size: int
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class Verdict:

    candidate: Candidate
    outcome: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in VERDICT_OUTCOMES:
            raise ValueError(
                f"outcome {self.outcome!r} is not one of {', '.join(VERDICT_OUTCOMES)}"
            )
        if self.outcome == OUTCOME_REFUSED and not self.reason.strip():
            raise ValueError("a refusal has to name the limit it broke (AC-035)")

    @property
    def accepted(self) -> bool:
        return self.outcome == OUTCOME_READ

    @property
    def refused(self) -> bool:
        return self.outcome == OUTCOME_REFUSED


@dataclass(frozen=True, slots=True)
class Screening:

    accepted: tuple[Candidate, ...]
    rejected: tuple[Verdict, ...]

    @property
    def clean(self) -> bool:
        return not self.rejected


@dataclass(frozen=True, slots=True)
class Limits:

    media_types: tuple[str, ...]
    pages: int
    byte_size: int
    refusals: Mapping[str, str] = field(default_factory=dict)
    page_equivalents: Mapping[str, str] = field(default_factory=dict)

    def accepts(self, media_type: str) -> bool:
        return media_type in self.media_types

    def page_ceiling(self, configured: int | None = None) -> int:
        return min(configured, self.pages) if configured is not None else self.pages

    def byte_ceiling(self, configured: int | None = None) -> int:
        return min(configured, self.byte_size) if configured is not None else self.byte_size

    def narrowed(self, *, pages: int | None = None, byte_size: int | None = None) -> "Limits":
        return replace(
            self,
            pages=self.page_ceiling(pages),
            byte_size=self.byte_ceiling(byte_size),
        )


DOCAI_LIMITS = Limits(
    media_types=SUPPORTED_MEDIA_TYPES,
    pages=BATCH_PAGE_LIMIT,
    byte_size=BATCH_BYTE_LIMIT,
    refusals={
        media_type: DOCAI_TEXT_REFUSAL.format(media_type=media_type)
        for media_type in UNREADABLE_TEXT_MEDIA_TYPES
    },
    page_equivalents=PAGE_EQUIVALENTS,
)


def page_limit_for(*, configured: int | None = None) -> int:
    return min(configured, BATCH_PAGE_LIMIT) if configured is not None else BATCH_PAGE_LIMIT


def byte_limit_for(*, configured: int | None = None) -> int:
    return min(configured, BATCH_BYTE_LIMIT) if configured is not None else BATCH_BYTE_LIMIT


def preflight(
    candidate: Candidate,
    *,
    limits: Limits = DOCAI_LIMITS,
    page_limit: int | None = None,
    byte_limit: int | None = None,
) -> Verdict:
    pages = limits.page_ceiling(page_limit)
    size = limits.byte_ceiling(byte_limit)
    own_refusal = limits.refusals.get(candidate.media_type)
    if own_refusal:
        return Verdict(candidate, OUTCOME_REFUSED, own_refusal)
    if not limits.accepts(candidate.media_type):
        return Verdict(
            candidate,
            OUTCOME_REFUSED,
            f"is {candidate.media_type}, which this deployment's reader does not accept; "
            f"it accepts {', '.join(limits.media_types)}",
        )
    if candidate.page_count is not None and candidate.page_count > pages:
        equivalent = limits.page_equivalents.get(candidate.media_type)
        counted = f" (a page here is {equivalent})" if equivalent else ""
        return Verdict(
            candidate,
            OUTCOME_REFUSED,
            f"has {candidate.page_count} pages{counted}, above the {pages}-page limit the "
            "reader accepts; it is not sent, and no part of it is processed",
        )
    if candidate.byte_size > size:
        return Verdict(
            candidate,
            OUTCOME_REFUSED,
            f"is {candidate.byte_size} bytes, above the {size}-byte limit the reader "
            "accepts; it is not sent, and no part of it is processed",
        )
    return Verdict(candidate, OUTCOME_READ)


def screen(
    candidates: Sequence[Candidate],
    *,
    limits: Limits = DOCAI_LIMITS,
    page_limit: int | None = None,
    byte_limit: int | None = None,
) -> Screening:
    verdicts = [
        preflight(candidate, limits=limits, page_limit=page_limit, byte_limit=byte_limit)
        for candidate in candidates
    ]
    return Screening(
        accepted=tuple(v.candidate for v in verdicts if v.accepted),
        rejected=tuple(v for v in verdicts if v.refused),
    )


def record_refusals(
    conn: psycopg.Connection,
    screening: Screening,
    *,
    reader: str,
    reader_version: str,
) -> int:
    for verdict in screening.rejected:
        record_read_failure(
            conn,
            document_id=verdict.candidate.document_id,
            reason=verdict.reason,
            reader=reader,
            reader_version=reader_version,
            page_count=verdict.candidate.page_count,
            byte_size=verdict.candidate.byte_size,
        )
    return len(screening.rejected)
