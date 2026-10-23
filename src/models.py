from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.store import Piece

METADATA_SOURCE_BLOCK_IDS = "source_block_ids"
METADATA_READER_VERSION = "reader_version"
METADATA_FINGERPRINT = "fingerprint"
METADATA_BLOCK_KINDS = "block_kinds"


class ChunkError(Exception):
    """A chunk was built that could never be cited, naming what is missing."""


@dataclass(frozen=True, slots=True)
class VersionStamps:
    """Which reader produced the text, and under which configuration it was cut.

    Two stamps and not one, because they move independently and cost different
    amounts to react to. A new ``reader_version`` means the document has to be
    read again — the expensive half. A new ``fingerprint`` means only that the
    cutting or the embedding configuration changed, which is re-derivable from
    the artifact already kept (AC-034), and is the whole argument for principle
    P-009.
    """

    reader_version: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.reader_version.strip():
            raise ChunkError(
                "a chunk has to name the reader version that produced it (AC-036); "
                "without it, nobody can tell which pieces have to be read again "
                "when the reader is upgraded"
            )
        if not self.fingerprint.strip():
            raise ChunkError(
                "a chunk has to name the indexing fingerprint it was cut under, or a "
                "configuration change cannot be told from a document change"
            )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable fragment of one document, with everything a citation needs.

    **The base carries what the next stage needs and what a citation needs to
    exist**, and nothing that belongs to whoever produced it: the text, which
    document, where in the cut, the pages, the heading path and the stamps. The
    guard below holds for any module, which is the reason it lives here.

    ``source_block_ids`` and ``block_kinds`` are here rather than on a variant
    because they are *what this piece was cut from*, which any module that cuts
    from blocks can answer — a module cutting a page at a time names the page —
    and because they are the two keys every stored row carries either way.
    ``source_block_ids`` is carried and not resolved away (ASM-023); it is what
    the page range was derived from.

    What only a cut arriving **from a reader** has — the identifier that reader
    called the chunk, and the page furniture it reported beside it — is on
    :class:`ReaderCutChunk`, and none of it reaches a row. The seam was implicit
    while there was one module; naming it is what keeps a second one from having
    to report values nobody produced (AC-002's rule, one layer down).

    ``document_external_id`` is the source's own identifier rather than the
    archive's row id: it is what survives a document being removed and coming
    back on the row it always had (AC-014), and the row id is not known until
    the write.

    ``page_start`` and ``page_end`` are inclusive and may be equal. A source
    with no pages at all leaves both ``None``, in which case the heading path
    is what makes the piece citable — one of the two has to be there.
    """

    document_external_id: str
    ordinal: int
    content: str
    stamps: VersionStamps
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    source_block_ids: tuple[str, ...] = ()
    block_kinds: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_external_id.strip():
            raise ChunkError("a chunk has to name the document it came from (principle P-002)")
        if self.ordinal < 0:
            raise ChunkError(f"a chunk's position cannot be negative, got {self.ordinal}")
        if not self.content.strip():
            raise ChunkError(
                f"chunk {self.ordinal} of document {self.document_external_id!r} carries no "
                "text; an empty piece is retrievable, uncitable and always wrong"
            )
        if "\x00" in self.content:
            raise ChunkError(
                f"chunk {self.ordinal} of document {self.document_external_id!r} carries a NUL "
                "byte, which no text column can hold; the read module that produced it has to "
                "drop it (src.processing.to_chunks.storable) — a glyph with no Unicode mapping "
                "is an absence, not a character"
            )
        if self.page_start is None and self.page_end is None and not self.heading_path:
            raise ChunkError(
                f"chunk {self.ordinal} of document {self.document_external_id!r} has no page "
                "range and no heading path, so nothing that returned it could be cited "
                "(principle P-002)"
            )
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ChunkError(
                f"chunk {self.ordinal} of document {self.document_external_id!r} spans pages "
                f"{self.page_start}..{self.page_end}, which runs backwards"
            )

    @property
    def pages(self) -> tuple[int, int] | None:
        if self.page_start is None and self.page_end is None:
            return None
        start = self.page_start if self.page_start is not None else self.page_end
        end = self.page_end if self.page_end is not None else self.page_start
        return (int(start), int(end))  # type: ignore[arg-type]

    def to_piece(self, embedding: Sequence[float]) -> Piece:
        return Piece(
            ordinal=self.ordinal,
            content=self.content,
            embedding=embedding,
            page_start=self.page_start,
            page_end=self.page_end,
            heading_path=self.heading_path,
            metadata={
                **dict(self.metadata),
                **self.provenance_metadata(),
                METADATA_READER_VERSION: self.stamps.reader_version,
                METADATA_FINGERPRINT: self.stamps.fingerprint,
            },
        )

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            METADATA_SOURCE_BLOCK_IDS: list(self.source_block_ids),
            METADATA_BLOCK_KINDS: list(self.block_kinds),
        }


@dataclass(frozen=True, slots=True)
class ReaderCutChunk(Chunk):
    """A piece of a cut a reader handed over, with what that reader also said.

    Three fields, and each is an answer only a module whose reader cut the
    document has. A module that cuts here names no ``chunk_id`` and reports no
    page furniture; giving it those fields would make it report values nobody
    produced.

    ``page_headers`` and ``page_footers`` are what the reader reported as the
    page's running furniture for this chunk. They are carried and read by
    nothing: they are already blocks of their own, and concatenating them
    would answer a question with a page header (AC-202). **None of the three
    reaches a stored row** — adding one to ``provenance_metadata`` adds a key
    to every row, which is a migration in everything but name (Q-116).
    """

    chunk_id: str = ""
    page_headers: tuple[str, ...] = ()
    page_footers: tuple[str, ...] = ()
