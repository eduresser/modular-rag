from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from src.config import SOURCE_GDRIVE

SOURCE_SLUG_GDRIVE = SOURCE_GDRIVE


class SourceError(Exception):
    ...


class DocumentUnavailable(Exception):

    def __init__(
        self,
        external_id: str,
        reason: str,
        *,
        denied: bool = False,
        trashed: bool = False,
    ) -> None:
        super().__init__(f"{external_id}: {reason}")
        self.external_id = external_id
        self.reason = reason
        self.denied = denied
        self.trashed = trashed


@dataclass(frozen=True, slots=True)
class SourceDocument:

    external_id: str
    name: str
    kind: str
    revision_marker: str | None = None
    modified_at: datetime | None = None
    size_bytes: int | None = None
    parents: tuple[str, ...] = ()
    link: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.external_id:
            raise SourceError("a source document must carry the source's own identifier")
        if not self.kind:
            raise SourceError(
                f"document {self.external_id!r} arrived with no type, so nothing can decide "
                "how to export it; the source's field projection is the usual cause (ASM-014)"
            )


@dataclass(frozen=True, slots=True)
class FetchedDocument:

    external_id: str
    content: bytes
    media_type: str
    content_digest: str

    def __post_init__(self) -> None:
        if not self.content_digest:
            raise SourceError(
                f"document {self.external_id!r} was fetched without a content digest; "
                "without it a moved watermark cannot be told from a real edit (AC-018)"
            )


@dataclass(frozen=True, slots=True)
class ChangePage:

    documents: tuple[SourceDocument, ...]
    removed: tuple[str, ...]
    next_position: str
    exhausted: bool

    @property
    def is_empty(self) -> bool:
        return not self.documents and not self.removed


class DeletionKind(Enum):

    TRASHED = "trashed"
    PERMANENTLY_DELETED = "permanently_deleted"


@dataclass(frozen=True, slots=True)
class DeletionEvent:

    external_id: str
    kind: DeletionKind
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class MoveEvent:

    external_id: str
    removed_parents: tuple[str, ...]
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class PermissionChangeEvent:

    external_id: str
    removed_permissions: tuple[str, ...]
    occurred_at: datetime | None


ActivityEvent = DeletionEvent | MoveEvent | PermissionChangeEvent


def group_by_document(
    events: "Iterable[ActivityEvent]",
) -> dict[str, tuple[ActivityEvent, ...]]:
    grouped: dict[str, list[ActivityEvent]] = {}
    for event in events:
        grouped.setdefault(event.external_id, []).append(event)
    return {external_id: tuple(page) for external_id, page in grouped.items()}


@runtime_checkable
class Source(Protocol):

    def sweep(self) -> Iterator[SourceDocument]:
        ...

    def changes_since(self, position: str | None) -> ChangePage:
        ...

    def start_position(self) -> str:
        ...

    def fetch(self, document: SourceDocument) -> FetchedDocument:
        ...

    def probe(self, external_id: str) -> SourceDocument:
        ...


def registered_slugs() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def source_class(slug: str) -> type:
    if slug not in _REGISTRY:
        raise SourceError(
            f"source module {slug!r} is not registered; registered modules are "
            f"{', '.join(registered_slugs())}"
        )
    return _REGISTRY[slug]()


def _gdrive_class() -> type:
    from src.sources.gdrive import GoogleDriveSource

    return GoogleDriveSource


_REGISTRY: Mapping[str, Callable[[], type]] = {SOURCE_SLUG_GDRIVE: _gdrive_class}
