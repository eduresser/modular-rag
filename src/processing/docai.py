from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import httpx

from src.models import VersionStamps
from src.processing.confusables import CONFUSABLES
from src.processing.to_chunks import (
    Block,
    Box,
    Cell,
    DocumentIdentity,
    Grid,
    LayoutIndex,
    MappedDocument,
    MappingError,
    ReaderChunk,
    list_lines,
    pieces_from_reader_cut,
    render_grid,
    storable,
)

READER_NAME = "docai-layout-parser"

API_HOST = "https://{region}-documentai.googleapis.com/v1"
SCOPE_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"

BATCH_PAGE_LIMIT = 500
MAX_DOCUMENTS_PER_BATCH = 1_000
MAX_CONCURRENT_BATCHES = 5

POLL_INTERVAL_SECONDS = 5.0
POLL_ROUND_LIMIT = 2_880


class ReaderError(Exception):
    ...


@dataclass(frozen=True, slots=True)
class ChunkingOptions:

    size: int
    cutter: str
    include_ancestor_headings: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ReaderError(f"a chunk size has to be positive, got {self.size}")


@dataclass(frozen=True, slots=True)
class InputDocument:

    external_id: str
    uri: str
    media_type: str

    def __post_init__(self) -> None:
        if not self.uri.startswith("gs://"):
            raise ReaderError(
                f"document {self.external_id!r} is at {self.uri!r}; the batch path reads "
                "from object storage only (ASM-020)"
            )


@dataclass(frozen=True, slots=True)
class Batch:

    documents: tuple[InputDocument, ...]
    output_prefix: str

    def __post_init__(self) -> None:
        if not self.documents:
            raise ReaderError("an empty batch is a request nobody has to make")
        if len(self.documents) > MAX_DOCUMENTS_PER_BATCH:
            raise ReaderError(
                f"a batch of {len(self.documents)} documents exceeds the provider's ceiling "
                f"of {MAX_DOCUMENTS_PER_BATCH} files per batch"
            )
        if not self.output_prefix.startswith("gs://"):
            raise ReaderError(
                f"batch output prefix {self.output_prefix!r} is not an object-storage "
                "location; the reader writes its output there and nowhere else"
            )


@dataclass(frozen=True, slots=True)
class DocumentOutcome:

    external_id: str
    uri: str
    succeeded: bool
    output_uri: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BatchOutcome:

    operation: str
    succeeded: bool
    documents: tuple[DocumentOutcome, ...] = ()
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RunOutcome:

    batches: tuple[BatchOutcome, ...]
    peak_in_flight: int

    @property
    def succeeded(self) -> bool:
        return all(batch.succeeded for batch in self.batches)

    def documents(self) -> tuple[DocumentOutcome, ...]:
        return tuple(document for batch in self.batches for document in batch.documents)


def plan_batches(
    documents: Sequence[InputDocument],
    *,
    output_prefix: str,
    max_documents_per_batch: int = MAX_DOCUMENTS_PER_BATCH,
) -> tuple[Batch, ...]:
    if max_documents_per_batch < 1:
        raise ReaderError(f"a batch has to hold at least one document, got {max_documents_per_batch}")
    size = min(max_documents_per_batch, MAX_DOCUMENTS_PER_BATCH)
    prefix = output_prefix.rstrip("/")
    return tuple(
        Batch(
            documents=tuple(documents[start : start + size]),
            output_prefix=f"{prefix}/batch-{start // size:04d}/",
        )
        for start in range(0, len(documents), size)
    )


@dataclass
class LayoutParser:

    processor_id: str
    processor_version: str
    http: httpx.Client
    token: Callable[[], str]
    chunking: ChunkingOptions
    region: str = "us"
    enable_table_annotation: bool = True
    enable_image_annotation: bool = True
    max_concurrent_batches: int = MAX_CONCURRENT_BATCHES
    sleeper: Callable[[float], None] = field(default=lambda seconds: None)
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    poll_round_limit: int = POLL_ROUND_LIMIT

    def __post_init__(self) -> None:
        if not self.processor_id.strip():
            raise ReaderError("a reader has to name the processor it sends documents to")
        if not self.processor_version.strip():
            raise ReaderError(
                "a reader has to name an explicit processor version (AC-036); an "
                "unversioned request resolves against the provider's current default, "
                "which moves without a deploy and does not return the same fields"
            )
        if not 1 <= self.max_concurrent_batches <= MAX_CONCURRENT_BATCHES:
            raise ReaderError(
                f"concurrency {self.max_concurrent_batches} is outside "
                f"1..{MAX_CONCURRENT_BATCHES}; the provider allows at most "
                f"{MAX_CONCURRENT_BATCHES} batch operations at once per project"
            )


    @property
    def root(self) -> str:
        return API_HOST.format(region=self.region)

    @property
    def version_name(self) -> str:
        return f"{self.processor_id}/processorVersions/{self.processor_version}"

    def batch_url(self) -> str:
        return f"{self.root}/{self.version_name}:batchProcess"

    def process_url(self) -> str:
        return f"{self.root}/{self.version_name}:process"

    def operation_url(self, operation: str) -> str:
        return f"{self.root}/{operation}"


    def layout_config(self) -> dict:
        return {
            "layoutConfig": {
                "enableTableAnnotation": self.enable_table_annotation,
                "enableImageAnnotation": self.enable_image_annotation,
                "returnBoundingBoxes": True,
                "chunkingConfig": {
                    "chunkSize": self.chunking.size,
                    "includeAncestorHeadings": self.chunking.include_ancestor_headings,
                },
            }
        }

    def batch_body(self, batch: Batch) -> dict:
        return {
            "inputDocuments": {
                "gcsDocuments": {
                    "documents": [
                        {"gcsUri": document.uri, "mimeType": document.media_type}
                        for document in batch.documents
                    ]
                }
            },
            "documentOutputConfig": {"gcsOutputConfig": {"gcsUri": batch.output_prefix}},
            "processOptions": self.layout_config(),
            "skipHumanReview": True,
        }

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"}


    def submit(self, batch: Batch) -> str:
        response = self.http.post(self.batch_url(), json=self.batch_body(batch), headers=self._headers())
        if response.status_code >= 400:
            raise ReaderError(f"the reader refused a batch of {len(batch.documents)}: {_message(response)}")
        name = response.json().get("name", "")
        if not name:
            raise ReaderError("the reader accepted a batch without naming an operation to poll")
        return str(name)

    def poll(self, operation: str, batch: Batch) -> BatchOutcome | None:
        response = self.http.get(self.operation_url(operation), headers=self._headers())
        if response.status_code >= 400:
            raise ReaderError(f"could not read operation {operation}: {_message(response)}")
        body = response.json()
        if not body.get("done"):
            return None
        error = body.get("error")
        return BatchOutcome(
            operation=operation,
            succeeded=error is None,
            documents=_document_outcomes(body, batch),
            detail="" if error is None else str(error.get("message", "")),
        )

    def run(self, batches: Sequence[Batch]) -> RunOutcome:
        queue = list(enumerate(batches))
        in_flight: dict[str, tuple[int, Batch]] = {}
        outcomes: list[BatchOutcome | None] = [None] * len(batches)
        peak = 0
        rounds = 0
        while queue or in_flight:
            while queue and len(in_flight) < self.max_concurrent_batches:
                index, batch = queue.pop(0)
                in_flight[self.submit(batch)] = (index, batch)
            peak = max(peak, len(in_flight))
            finished = []
            for operation, (index, batch) in in_flight.items():
                outcome = self.poll(operation, batch)
                if outcome is not None:
                    outcomes[index] = outcome
                    finished.append(operation)
            for operation in finished:
                del in_flight[operation]
            if in_flight and not finished:
                rounds += 1
                if rounds > self.poll_round_limit:
                    raise ReaderError(
                        f"{len(in_flight)} batch operation(s) never reported done after "
                        f"{self.poll_round_limit} polls; the run is abandoned rather than "
                        "left polling unattended"
                    )
                self.sleeper(self.poll_interval_seconds)
        return RunOutcome(batches=tuple(outcome for outcome in outcomes if outcome is not None), peak_in_flight=peak)


    def recut(self, document: dict, chunking: ChunkingOptions) -> dict:
        body = {
            "inlineDocument": document,
            "processOptions": {
                "layoutConfig": {
                    "enableTableAnnotation": self.enable_table_annotation,
                    "enableImageAnnotation": self.enable_image_annotation,
                    "chunkingConfig": {
                        "chunkSize": chunking.size,
                        "includeAncestorHeadings": chunking.include_ancestor_headings,
                    },
                }
            },
            "skipHumanReview": True,
        }
        response = self.http.post(self.process_url(), json=body, headers=self._headers())
        if response.status_code >= 400:
            raise ReaderError(f"the reader refused to re-cut a stored document: {_message(response)}")
        return response.json()


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", response.text))
    except ValueError:
        return response.text


def _document_outcomes(body: dict, batch: Batch) -> tuple[DocumentOutcome, ...]:
    statuses = {
        status.get("inputGcsSource", ""): status
        for status in body.get("metadata", {}).get("individualProcessStatuses", [])
    }
    outcomes = []
    for document in batch.documents:
        status = statuses.get(document.uri)
        if status is None:
            outcomes.append(
                DocumentOutcome(
                    external_id=document.external_id,
                    uri=document.uri,
                    succeeded=False,
                    detail="the reader reported no status for this document",
                )
            )
            continue
        code = status.get("status", {}).get("code", 0)
        succeeded = code == 0
        outcomes.append(
            DocumentOutcome(
                external_id=document.external_id,
                uri=document.uri,
                succeeded=succeeded,
                output_uri=status.get("outputGcsDestination") if succeeded else None,
                detail=str(status.get("status", {}).get("message", "")),
            )
        )
    return tuple(outcomes)


HEADING_LEVELS: dict[str, int] = {
    "title": 0,
    "subtitle": 1,
    "heading-1": 1,
    "heading-2": 2,
    "heading-3": 3,
    "heading-4": 4,
    "heading-5": 5,
    "heading-6": 6,
}

HEADING_TYPES: frozenset[str] = frozenset(HEADING_LEVELS)

FURNITURE_TYPES: frozenset[str] = frozenset({"header", "footer"})

BLOCK_TEXT = "textBlock"
BLOCK_LIST = "listBlock"
BLOCK_TABLE = "tableBlock"
BLOCK_IMAGE = "imageBlock"

KNOWN_BLOCK_KEYS: tuple[str, ...] = (BLOCK_TEXT, BLOCK_LIST, BLOCK_TABLE, BLOCK_IMAGE)

BLOCK_KEY_SUFFIX = "Block"

ANONYMOUS_BLOCK = "#/blocks/{position}"


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _repair_word(word: str) -> str:
    foreign = [char for char in word if char.isalpha() and not _is_latin(char)]
    if not foreign or any(char not in CONFUSABLES for char in foreign):
        return word
    return "".join(CONFUSABLES.get(char, char) for char in word)


def _is_latin(letter: str) -> bool:
    return "LATIN" in unicodedata.name(letter, "")


def _repair_homoglyphs(text: str) -> str:
    if not text:
        return text
    return storable(_WORD.sub(lambda match: _repair_word(match.group()), text))


def document_of(payload: dict) -> dict:
    if "documentLayout" in payload or "chunkedDocument" in payload:
        return payload
    document = payload.get("document")
    if isinstance(document, dict):
        return document
    raise MappingError(
        "this payload holds no document: expected a 'document' object, or the document "
        "itself with 'documentLayout' and 'chunkedDocument'"
    )


def identity_of(
    *, external_id: str, filename: str, media_type: str, content_digest: str
) -> DocumentIdentity:
    if not (external_id and filename and media_type):
        return DocumentIdentity()
    digest_source = content_digest or external_id
    return DocumentIdentity(
        name=filename,
        external_id=external_id,
        filename=filename,
        media_type=media_type,
        content_digest=hashlib.sha256(digest_source.encode()).hexdigest(),
    )


def index_of(
    payload: dict,
    *,
    external_id: str = "",
    filename: str = "",
    media_type: str = "",
    content_digest: str = "",
) -> LayoutIndex:
    document = document_of(payload)
    blocks: dict[str, Block] = {}
    order: list[str] = []
    unrecognised: list[str] = []
    furniture: dict[str, Block] = {}
    stack: list[tuple[int, str]] = []
    position = 0

    def path() -> tuple[str, ...]:
        return tuple(text for _, text in stack)

    def register(block: Block, aliases: Sequence[str] = ()) -> None:
        if block.block_id in blocks:
            return
        blocks[block.block_id] = block
        order.append(block.block_id)
        for alias in aliases:
            if alias and alias != block.block_id:
                blocks.setdefault(alias, block)

    def walk(raw_blocks: Iterable[dict], *, contained: bool) -> None:
        nonlocal position
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            key = _block_key(raw)
            if key is None:
                continue
            inner = raw[key] if isinstance(raw.get(key), dict) else {}
            block_id = str(raw.get("blockId", "")) or ANONYMOUS_BLOCK.format(position=position)
            position += 1
            common = {
                "block_id": block_id,
                "kinds": _kinds_of(key, inner),
                "page_start": _span_of(raw)[0],
                "page_end": _span_of(raw)[1],
                "contained": contained,
                "box": _box_of(raw),
            }
            if key == BLOCK_TEXT:
                text = _repair_homoglyphs(str(inner.get("text", "")))
                block_type = str(inner.get("type", ""))
                ancestors = path()
                own = ancestors
                if block_type in HEADING_TYPES and text.strip():
                    level = HEADING_LEVELS[block_type]
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    ancestors = path()
                    stack.append((level, text.strip()))
                    own = path()
                block = Block(
                    **common,
                    text=text,
                    description=_description_of(raw, inner),
                    heading_path=ancestors,
                    own_path=own,
                )
                if block_type in FURNITURE_TYPES:
                    furniture.setdefault(block.block_id, block)
                else:
                    register(block)
                walk(inner.get("blocks", []), contained=contained)
            elif key == BLOCK_LIST:
                register(
                    Block(
                        **common,
                        text=render_list(inner),
                        description=_description_of(raw, inner),
                        heading_path=path(),
                        own_path=path(),
                    )
                )
                for entry in inner.get("listEntries", []):
                    if isinstance(entry, dict):
                        walk(entry.get("blocks", []), contained=True)
            elif key == BLOCK_TABLE:
                register(
                    Block(
                        **common,
                        text=render_table(inner),
                        description=_table_annotation_of(raw, inner),
                        caption=_table_caption_of(raw, inner),
                        heading_path=path(),
                        own_path=path(),
                    ),
                    aliases=_identifiers_in(_cell_blocks(inner)),
                )
            elif key == BLOCK_IMAGE:
                register(
                    Block(
                        **common,
                        description=_image_description_of(raw, inner),
                        heading_path=path(),
                        own_path=path(),
                    )
                )
            else:
                block = Block(
                    **common,
                    text=_repair_homoglyphs(_any_text(inner)),
                    description=_description_of(raw, inner),
                    heading_path=path(),
                    own_path=path(),
                )
                register(block)
                unrecognised.append(block.block_id)

    walk(document.get("documentLayout", {}).get("blocks", []), contained=False)
    return LayoutIndex(
        blocks=blocks,
        order=tuple(order),
        unrecognised=tuple(unrecognised),
        furniture=furniture,
        identity=identity_of(
            external_id=external_id,
            filename=filename,
            media_type=media_type,
            content_digest=content_digest,
        ),
    )


def _block_key(raw: dict) -> str | None:
    for key in KNOWN_BLOCK_KEYS:
        if key in raw:
            return key
    for key in raw:
        if key.endswith(BLOCK_KEY_SUFFIX):
            return key
    return None


def _kinds_of(key: str, inner: dict) -> tuple[str, ...]:
    block_type = str(inner.get("type", "")).strip()
    return (key, block_type) if block_type else (key,)


NOT_TEXT_KEYS: frozenset[str] = frozenset({"type", "mimeType", "dataUri", "uri"})


def _any_text(node: object) -> str:
    found: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                if name.endswith("Id") or name in NOT_TEXT_KEYS:
                    continue
                if isinstance(child, str):
                    if child.strip():
                        found.append(child.strip())
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(node)
    return "\n\n".join(found)


def reader_chunks_of(payload: dict) -> tuple[ReaderChunk, ...]:
    document = document_of(payload)
    return tuple(
        _reader_chunk(raw)
        for raw in document.get("chunkedDocument", {}).get("chunks", [])
        if isinstance(raw, dict)
    )


def _reader_chunk(raw: dict) -> ReaderChunk:
    start, end = _span_of(raw)
    return ReaderChunk(
        chunk_id=str(raw.get("chunkId", "")),
        content=_repair_homoglyphs(str(raw.get("content", ""))),
        source_block_ids=tuple(str(block_id) for block_id in raw.get("sourceBlockIds", [])),
        page_start=start,
        page_end=end,
        page_headers=_furniture_texts(raw.get("pageHeaders")),
        page_footers=_furniture_texts(raw.get("pageFooters")),
    )


def _furniture_texts(entries: object) -> tuple[str, ...]:
    if not isinstance(entries, list):
        return ()
    return tuple(
        _repair_homoglyphs(str(entry.get("text", "")).strip())
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("text", "")).strip()
    )


def _box_of(raw: dict) -> Box | None:
    box = raw.get("boundingBox")
    if not isinstance(box, dict):
        return None
    try:
        return Box(
            left=float(box["left"]),
            top=float(box["top"]),
            right=float(box["right"]),
            bottom=float(box["bottom"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _identifiers_in(blocks: Iterable[dict]) -> tuple[str, ...]:
    found: list[str] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block_id = str(raw.get("blockId", ""))
        if block_id:
            found.append(block_id)
        for key in (BLOCK_TEXT, BLOCK_LIST, BLOCK_TABLE):
            inner = raw.get(key)
            if not isinstance(inner, dict):
                continue
            found.extend(_identifiers_in(inner.get("blocks", [])))
            for entry in inner.get("listEntries", []):
                if isinstance(entry, dict):
                    found.extend(_identifiers_in(entry.get("blocks", [])))
            found.extend(_identifiers_in(_cell_blocks(inner)))
    return tuple(found)


def _cell_blocks(table_block: dict) -> list[dict]:
    return [
        block
        for row in _rows_of(table_block)
        for cell in row.get("cells", [])
        for block in cell.get("blocks", [])
    ]


def _is_ordered(list_block: dict) -> bool:
    return str(list_block.get("type", "")).lower().startswith("ordered")


def _grid_of(table_block: dict) -> Grid:
    header = list(table_block.get("headerRows", []))
    rows = header + list(table_block.get("bodyRows", []))
    cells: list[Cell] = []
    occupied_until: dict[int, int] = {}
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row.get("cells", []):
            while occupied_until.get(column, 0) > row_index:
                column += 1
            row_span = max(1, int(cell.get("rowSpan", 1) or 1))
            col_span = max(1, int(cell.get("colSpan", 1) or 1))
            cells.append(
                Cell(
                    text=_joined_text(cell.get("blocks", [])),
                    row=row_index,
                    column=column,
                    row_span=row_span,
                    col_span=col_span,
                    header=row_index < len(header),
                )
            )
            for occupied_column in range(column, column + col_span):
                occupied_until[occupied_column] = max(
                    occupied_until.get(occupied_column, 0), row_index + row_span
                )
            column += col_span
    width = max((cell.column + cell.col_span for cell in cells), default=0)
    return Grid(rows=len(rows), columns=width, cells=tuple(cells))


def _span_of(raw: dict) -> tuple[int | None, int | None]:
    span = raw.get("pageSpan") or {}
    start = span.get("pageStart")
    end = span.get("pageEnd", start)
    return (
        int(start) if start is not None else None,
        int(end) if end is not None else None,
    )


def _description_of(raw: dict, inner: dict) -> str:
    for holder in (inner, raw):
        annotations = holder.get("annotations")
        if isinstance(annotations, dict):
            description = str(annotations.get("description", "")).strip()
            if description:
                return description
        caption = str(holder.get("caption", "")).strip()
        if caption:
            return caption
    return ""


def _image_text_of(raw: dict, inner: dict) -> str:
    for holder in (inner, raw):
        text = _repair_homoglyphs(str(holder.get("imageText", ""))).strip()
        if text:
            return text
    return ""


def _image_description_of(raw: dict, inner: dict) -> str:
    description = _description_of(raw, inner)
    image_text = _image_text_of(raw, inner)
    if not image_text or image_text in description:
        return description
    if not description:
        return image_text
    return f"{description}\n\n{image_text}"


def _table_caption_of(raw: dict, inner: dict) -> str:
    for holder in (inner, raw):
        caption = str(holder.get("caption", "")).strip()
        if caption:
            return caption
    return ""


def _table_annotation_of(raw: dict, inner: dict) -> str:
    for holder in (inner, raw):
        annotations = holder.get("annotations")
        if isinstance(annotations, dict):
            description = str(annotations.get("description", "")).strip()
            if description:
                return description
    return ""


def _rows_of(table_block: dict) -> list[dict]:
    return list(table_block.get("headerRows", [])) + list(table_block.get("bodyRows", []))


def _joined_text(blocks: Iterable[dict]) -> str:
    return " ".join(part for part in (_text_of(block) for block in blocks) if part)


def _text_of(raw: dict) -> str:
    if BLOCK_TEXT in raw:
        text_block = raw[BLOCK_TEXT]
        nested = _joined_text(text_block.get("blocks", []))
        own = _repair_homoglyphs(str(text_block.get("text", ""))).strip()
        return f"{own} {nested}".strip()
    if BLOCK_LIST in raw:
        return render_list(raw[BLOCK_LIST])
    if BLOCK_TABLE in raw:
        return render_table(raw[BLOCK_TABLE])
    if BLOCK_IMAGE in raw:
        return _description_of(raw, raw[BLOCK_IMAGE])
    key = _block_key(raw)
    inner = raw.get(key) if key else None
    return _repair_homoglyphs(_any_text(inner)) if isinstance(inner, dict) else ""


def render_list(list_block: dict) -> str:
    return list_lines(
        (
            _joined_text(entry.get("blocks", []))
            for entry in list_block.get("listEntries", [])
            if isinstance(entry, dict)
        ),
        ordered=_is_ordered(list_block),
    )


def render_table(table_block: dict) -> str:
    return render_grid(_grid_of(table_block))


def index_blocks(payload: dict) -> LayoutIndex:
    return index_of(payload)


def chunks_from_document(
    payload: dict,
    *,
    external_id: str,
    stamps: VersionStamps,
    filename: str = "",
    media_type: str = "",
    content_digest: str = "",
) -> MappedDocument:
    return pieces_from_reader_cut(
        index_of(
            payload,
            external_id=external_id,
            filename=filename,
            media_type=media_type,
            content_digest=content_digest,
        ),
        reader_chunks_of(payload),
        external_id=external_id,
        stamps=stamps,
    )
