from __future__ import annotations

import io
from functools import cache
from importlib import metadata

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from src.models import VersionStamps
from src.processing.preflight import Limits
from src.processing.render import MEDIA_PDF
from src.processing.to_chunks import (
    Block,
    CutPiece,
    DocumentIdentity,
    LayoutIndex,
    MappedDocument,
    MappingError,
    pieces_from,
    storable,
)
from src.sources.gdrive_export import MEDIA_DOCX

READER_NAME = "pdftext"

EXTRACTOR = "pypdf"

BLOCK_PAGE = "page"

PAGE_BLOCK_ID = "p{page}"

NO_TEXT_REASON = "refused for want of a text layer"

SUPPORTED_MEDIA_TYPES: tuple[str, ...] = (MEDIA_PDF, MEDIA_DOCX)

PAGE_LIMIT = 5_000
BYTE_LIMIT = 536_870_912

LIMITS = Limits(
    media_types=SUPPORTED_MEDIA_TYPES,
    pages=PAGE_LIMIT,
    byte_size=BYTE_LIMIT,
    refusals={},
    page_equivalents={},
)


class PdfTextError(Exception):
    ...


class NoTextLayer(PdfTextError):
    ...


@cache
def extractor_version() -> str:
    return f"{EXTRACTOR}-{metadata.version(EXTRACTOR)}"


def rendering_for(media_type: str) -> str:
    if media_type == MEDIA_PDF:
        return ""
    return MEDIA_PDF if media_type in SUPPORTED_MEDIA_TYPES else ""


def extract(content: bytes, *, external_id: str = "") -> dict:
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [
            {"page": number, "text": storable(page.extract_text() or "").strip()}
            for number, page in enumerate(reader.pages, start=1)
        ]
    except (PyPdfError, ValueError) as exc:
        raise PdfTextError(
            f"document {external_id!r} could not be opened as a PDF: {exc}"
        ) from exc
    if not pages:
        raise PdfTextError(f"document {external_id!r} opened as a PDF holding no pages at all")
    if not any(page["text"] for page in pages):
        raise NoTextLayer(
            f"{NO_TEXT_REASON}: all {len(pages)} pages of document {external_id!r} carry no "
            "extractable text, which is what a scan looks like to this module; reading it "
            "needs optical character recognition, which this route does not buy"
        )
    return {"extractor": EXTRACTOR, "extractorVersion": extractor_version(), "pages": pages}


def pages_in(payload: dict) -> int:
    return len(_pages_of(payload))


def index_of(
    payload: dict,
    *,
    external_id: str = "",
    filename: str = "",
    media_type: str = "",
    content_digest: str = "",
) -> LayoutIndex:
    blocks: dict[str, Block] = {}
    order: list[str] = []
    for entry in _pages_of(payload):
        number = int(entry.get("page", 0))
        block = Block(
            block_id=PAGE_BLOCK_ID.format(page=number),
            kinds=(BLOCK_PAGE,),
            text=storable(str(entry.get("text", ""))),
            page_start=number,
            page_end=number,
        )
        blocks[block.block_id] = block
        order.append(block.block_id)
    return LayoutIndex(
        blocks=blocks,
        order=tuple(order),
        identity=identity_of(
            external_id=external_id,
            filename=filename,
            media_type=media_type,
            content_digest=content_digest,
        ),
    )


def identity_of(
    *, external_id: str, filename: str, media_type: str, content_digest: str
) -> DocumentIdentity:
    if not (external_id and filename and media_type):
        return DocumentIdentity()
    return DocumentIdentity(
        name=filename,
        external_id=external_id,
        filename=filename,
        media_type=media_type,
        content_digest=content_digest,
    )


def cut(index: LayoutIndex) -> tuple[CutPiece, ...]:
    return tuple(
        CutPiece(
            text=block.text,
            blocks=(block,),
            source_block_ids=(block.block_id,),
            page_start=block.page_start,
            page_end=block.page_end,
        )
        for block in index.in_order()
        if block.text.strip()
    )


def chunks_from_document(
    payload: dict,
    *,
    external_id: str,
    stamps: VersionStamps,
    filename: str = "",
    media_type: str = "",
    content_digest: str = "",
) -> MappedDocument:
    index = index_of(
        payload,
        external_id=external_id,
        filename=filename,
        media_type=media_type,
        content_digest=content_digest,
    )
    pieces = cut(index)
    if not pieces:
        raise MappingError(
            f"{NO_TEXT_REASON}: document {external_id!r} was read and none of its "
            f"{len(index.order)} pages carries text"
        )
    return MappedDocument(
        chunks=pieces_from(pieces, external_id=external_id, stamps=stamps),
        index=index,
    )


def _pages_of(payload: dict) -> list[dict]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise MappingError(
            "this payload holds no pages: expected the shape this module keeps, with a "
            "'pages' list of {page, text}"
        )
    return [entry for entry in pages if isinstance(entry, dict)]
