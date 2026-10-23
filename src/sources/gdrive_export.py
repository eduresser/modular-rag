from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

KIND_DOCUMENT = "application/vnd.google-apps.document"
KIND_SPREADSHEET = "application/vnd.google-apps.spreadsheet"
KIND_PRESENTATION = "application/vnd.google-apps.presentation"
KIND_SHORTCUT = "application/vnd.google-apps.shortcut"
KIND_FOLDER = "application/vnd.google-apps.folder"

MEDIA_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MEDIA_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MEDIA_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
MEDIA_PDF = "application/pdf"

EXPORT_SIZE_LIMIT_BYTES = 10 * 1024 * 1024


class Retrieval(Enum):

    EXPORT = "export"
    DOWNLOAD = "download"


@dataclass(frozen=True, slots=True)
class ExportPlan:

    retrieval: Retrieval
    media_type: str
    rationale: str

    @property
    def is_export(self) -> bool:
        return self.retrieval is Retrieval.EXPORT


class UnsupportedKind(Exception):
    ...


class ExportTooLarge(Exception):
    ...


_EXPORTS: Mapping[str, ExportPlan] = {
    KIND_DOCUMENT: ExportPlan(
        retrieval=Retrieval.EXPORT,
        media_type=MEDIA_PDF,
        rationale=(
            "the source renders it to PDF itself, free and by the same renderer that produced "
            "what the author saw; the reader returns a page and a heading for every block of a "
            "PDF and returned neither for the word-processor export it replaces, measured "
            "2026-09-13 on the document this corpus actually needed it for (AC-210)"
        ),
    ),
    KIND_SPREADSHEET: ExportPlan(
        retrieval=Retrieval.EXPORT,
        media_type=MEDIA_XLSX,
        rationale=(
            "the workbook keeps every tab; the comma-separated export carries only the "
            "first one, silently, so a spreadsheet with a tab per matter would arrive as "
            "one matter and look valid (AC-017)"
        ),
    ),
    KIND_PRESENTATION: ExportPlan(
        retrieval=Retrieval.EXPORT,
        media_type=MEDIA_PPTX,
        rationale=(
            "the presentation format keeps slide boundaries and speaker notes, which a PDF "
            "render flattens"
        ),
    ),
}

NOT_DOCUMENTS: frozenset[str] = frozenset({KIND_FOLDER, KIND_SHORTCUT})


def is_document(kind: str) -> bool:
    return kind not in NOT_DOCUMENTS


def export_plan(kind: str) -> ExportPlan:
    if kind in NOT_DOCUMENTS:
        raise UnsupportedKind(
            f"{kind!r} is not a document — a folder and a shortcut are structure, and a "
            "shortcut is resolved to its target instead of being indexed (AC-015)"
        )
    if kind in _EXPORTS:
        return _EXPORTS[kind]
    if kind.startswith("application/vnd.google-apps."):
        raise UnsupportedKind(
            f"{kind!r} is a document in the source's own editor with no export mapping; "
            f"mapped kinds are {', '.join(sorted(_EXPORTS))}. Falling back to plain text "
            "would index its title as its contents, which reads as coverage the archive "
            "does not have"
        )
    return ExportPlan(
        retrieval=Retrieval.DOWNLOAD,
        media_type=kind,
        rationale="uploaded rather than authored in the source, so the original bytes are the document",
    )


def reject_oversized(external_id: str, size_bytes: int | None) -> None:
    if size_bytes is not None and size_bytes > EXPORT_SIZE_LIMIT_BYTES:
        raise ExportTooLarge(
            f"document {external_id!r} is {size_bytes} bytes, over the source's "
            f"{EXPORT_SIZE_LIMIT_BYTES}-byte export ceiling, and the source offers no "
            "partial download (ASM-011)"
        )


def reject_oversized_content(external_id: str, content: bytes) -> bytes:
    if len(content) > EXPORT_SIZE_LIMIT_BYTES:
        raise ExportTooLarge(
            f"document {external_id!r} exported to {len(content)} bytes, over the source's "
            f"{EXPORT_SIZE_LIMIT_BYTES}-byte ceiling (ASM-011)"
        )
    return content


def content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
