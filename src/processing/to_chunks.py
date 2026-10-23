from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.models import Chunk, ReaderCutChunk, VersionStamps

DESCRIPTION_PREFIX = "> "

NUL = "\x00"


def storable(text: str) -> str:
    return text.replace(NUL, "") if text else text


class MappingError(Exception):
    ...


NO_LOCATOR_REASON = "refused for want of a locator"


class NoLocatorError(MappingError):
    ...


@dataclass(frozen=True, slots=True)
class Box:

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class DocumentIdentity:

    name: str = "document"
    external_id: str = ""
    filename: str = ""
    media_type: str = ""
    content_digest: str = ""

    @property
    def uri(self) -> str:
        return f"source:{self.external_id}" if self.external_id else ""


@dataclass(frozen=True, slots=True)
class Block:

    block_id: str
    kinds: tuple[str, ...] = ()
    text: str = ""
    description: str = ""
    caption: str = ""
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    own_path: tuple[str, ...] = ()
    contained: bool = False
    box: Box | None = None

    @property
    def is_heading(self) -> bool:
        return self.own_path != self.heading_path

    @property
    def citable_text(self) -> str:
        notes = [
            f"{DESCRIPTION_PREFIX}{note}"
            for note in (self.caption, self.description)
            if note and note not in self.text
        ]
        if not notes:
            return self.text
        return "\n\n".join([self.text, *notes]) if self.text else "\n\n".join(notes)


@dataclass(frozen=True, slots=True)
class LayoutIndex:

    blocks: dict[str, Block] = field(default_factory=dict)
    order: tuple[str, ...] = ()
    unrecognised: tuple[str, ...] = ()
    furniture: dict[str, Block] = field(default_factory=dict)
    identity: DocumentIdentity = field(default_factory=DocumentIdentity)

    def __getitem__(self, block_id: str) -> Block:
        return self.blocks[block_id]

    def get(self, block_id: str) -> Block | None:
        return self.blocks.get(block_id)

    def in_order(self) -> tuple[Block, ...]:
        return tuple(self.blocks[block_id] for block_id in self.order)


@dataclass(frozen=True, slots=True)
class ReaderChunk:

    chunk_id: str
    content: str = ""
    source_block_ids: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    page_headers: tuple[str, ...] = ()
    page_footers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CutPiece:

    text: str
    blocks: tuple[Block, ...] = ()
    headings: tuple[str, ...] = ()
    source_block_ids: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str = ""
    page_headers: tuple[str, ...] = ()
    page_footers: tuple[str, ...] = ()

    @property
    def kinds(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for block in self.blocks:
            for kind in block.kinds:
                seen.setdefault(kind, None)
        return tuple(seen)

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(block.box for block in self.blocks if block.box is not None)

    @property
    def pages(self) -> tuple[int | None, int | None]:
        if self.page_start is not None or self.page_end is not None:
            return (self.page_start, self.page_end)
        starts = [block.page_start for block in self.blocks if block.page_start is not None]
        ends = [block.page_end for block in self.blocks if block.page_end is not None]
        if not starts and not ends:
            return (None, None)
        return (min(starts) if starts else None, max(ends) if ends else None)


@dataclass(frozen=True, slots=True)
class MappedDocument:

    chunks: tuple[Chunk, ...]
    index: LayoutIndex

    @property
    def unrecognised_blocks(self) -> tuple[str, ...]:
        return self.index.unrecognised


@dataclass(frozen=True, slots=True)
class Cell:

    text: str
    row: int
    column: int
    row_span: int = 1
    col_span: int = 1
    header: bool = False


@dataclass(frozen=True, slots=True)
class Grid:

    rows: int = 0
    columns: int = 0
    cells: tuple[Cell, ...] = ()


def render_grid(grid: Grid) -> str:
    if grid.columns == 0 or not grid.cells:
        return ""
    rows: dict[int, dict[int, str]] = {}
    header_rows: set[int] = set()
    for cell in grid.cells:
        rows.setdefault(cell.row, {})[cell.column] = cell.text
        if cell.header:
            header_rows.add(cell.row)
    lines: list[str] = []
    for index in sorted(rows):
        cells = [rows[index].get(column, "") for column in range(grid.columns)]
        lines.append(_markdown_row(cells, grid.columns))
        if index in header_rows and (index + 1) not in header_rows:
            lines.append(_markdown_row(["---"] * grid.columns, grid.columns))
    return "\n".join(lines)


def list_lines(entries: Iterable[str], *, ordered: bool) -> str:
    lines = []
    for position, text in enumerate(entries, start=1):
        marker = f"{position}." if ordered else "-"
        lines.append(f"{marker} {text}".rstrip())
    return "\n".join(lines)


def _markdown_row(cells: Sequence[str], width: int) -> str:
    padded = list(cells) + [""] * (width - len(cells))
    return "| " + " | ".join(cell.strip() for cell in padded) + " |"


def pieces_from(
    cut: Iterable[CutPiece],
    *,
    external_id: str,
    stamps: VersionStamps,
    record: type[Chunk] = Chunk,
) -> tuple[Chunk, ...]:
    pieces: list[Chunk] = []
    reader_cut = issubclass(record, ReaderCutChunk)
    for ordinal, piece in enumerate(cut):
        page_start, page_end = piece.pages
        fields: dict[str, Any] = {
            "document_external_id": external_id,
            "ordinal": ordinal,
            "content": piece.text,
            "stamps": stamps,
            "page_start": page_start,
            "page_end": page_end,
            "heading_path": tuple(piece.headings),
            "source_block_ids": tuple(piece.source_block_ids),
            "block_kinds": piece.kinds,
        }
        if reader_cut:
            fields |= {
                "chunk_id": piece.chunk_id,
                "page_headers": tuple(piece.page_headers),
                "page_footers": tuple(piece.page_footers),
            }
        pieces.append(record(**fields))
    return tuple(pieces)


def cut_reader_chunks(
    index: LayoutIndex,
    reader_chunks: Sequence[ReaderChunk],
) -> tuple[CutPiece, ...]:
    cut: list[CutPiece] = []
    unplaceable: list[tuple[str, MappingError]] = []
    for ordinal, reader_chunk in enumerate(reader_chunks):
        try:
            cut.append(place(reader_chunk, index))
        except MappingError as exc:
            unplaceable.append((str(reader_chunk.chunk_id or ordinal), exc))
    if unplaceable:
        if (
            reader_chunks
            and len(unplaceable) == len(reader_chunks)
            and all(isinstance(exc, NoLocatorError) for _, exc in unplaceable)
        ):
            raise NoLocatorError(
                f"{NO_LOCATOR_REASON}: none of its {len(reader_chunks)} pieces carry a "
                "page or a heading (principle P-002)"
            )
        raise MappingError(
            f"{len(unplaceable)} of {len(reader_chunks)} chunks the reader produced could "
            "not be placed, and dropping them would make the archive quietly smaller than "
            "the document — "
            + "; ".join(f"{chunk_id}: {exc}" for chunk_id, exc in unplaceable)
        )
    return tuple(cut)


def pieces_from_reader_cut(
    index: LayoutIndex,
    reader_chunks: Sequence[ReaderChunk],
    *,
    external_id: str,
    stamps: VersionStamps,
) -> MappedDocument:
    try:
        cut = cut_reader_chunks(index, reader_chunks)
    except MappingError as exc:
        wrapped = NoLocatorError if isinstance(exc, NoLocatorError) else MappingError
        raise wrapped(f"document {external_id!r}: {exc}") from exc
    return MappedDocument(
        chunks=pieces_from(
            cut, external_id=external_id, stamps=stamps, record=ReaderCutChunk
        ),
        index=index,
    )


def place(reader_chunk: ReaderChunk, index: LayoutIndex) -> CutPiece:
    source_ids = reader_chunk.source_block_ids
    sources = tuple(
        block for block in (index.get(block_id) for block_id in source_ids) if block is not None
    )
    content = _content_of(reader_chunk, sources)
    if not content.strip():
        raise MappingError("the chunk carries no text and its source blocks carry none either")
    page_start, page_end = _pages_of(reader_chunk, sources)
    heading_path = _heading_path_of(sources)
    if page_start is None and page_end is None and not heading_path:
        if sources:
            raise NoLocatorError(
                "no page range and no heading path could be resolved, so nothing that "
                "returned it could be cited (principle P-002)"
            )
        raise MappingError(
            "no page range and no heading path could be resolved, so nothing that returned "
            "it could be cited (principle P-002)"
        )
    return CutPiece(
        text=content,
        blocks=sources,
        headings=heading_path,
        source_block_ids=source_ids,
        page_start=page_start,
        page_end=page_end,
        chunk_id=reader_chunk.chunk_id,
        page_headers=reader_chunk.page_headers,
        page_footers=reader_chunk.page_footers,
    )


def _content_of(reader_chunk: ReaderChunk, sources: Sequence[Block]) -> str:
    content = reader_chunk.content.strip()
    if not content:
        return "\n\n".join(
            block.citable_text for block in sources if block.citable_text
        ).strip()
    notes = [
        f"{DESCRIPTION_PREFIX}{note}"
        for block in sources
        for note in (block.caption, block.description)
        if note and note not in content
    ]
    return "\n\n".join([content, *_unique(notes)]) if notes else content


def _unique(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _pages_of(reader_chunk: ReaderChunk, sources: Sequence[Block]) -> tuple[int | None, int | None]:
    starts = [block.page_start for block in sources if block.page_start is not None]
    ends = [block.page_end for block in sources if block.page_end is not None]
    if starts or ends:
        return (min(starts) if starts else None, max(ends) if ends else None)
    start, end = reader_chunk.page_start, reader_chunk.page_end
    return (start, end if end is not None else start)


def _heading_path_of(sources: Sequence[Block]) -> tuple[str, ...]:
    for block in sources:
        if not block.is_heading:
            return block.heading_path
    return max((block.own_path for block in sources), key=len, default=())
