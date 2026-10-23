from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import psycopg

from src.models import Chunk, VersionStamps
from src.processing.artifact import (
    ArtifactStore,
    ParsedArtifact,
    load_parsed_artifact,
    require_parsed_artifact,
)
from src.config import CHUNK_DOCAI, PROCESS_PDFTEXT, READER_DOCAI
from src.processing import docai, pdftext
from src.processing.docai import ChunkingOptions, LayoutParser
from src.processing.to_chunks import MappedDocument

RECUT_LOCAL = "local"
RECUT_PROVIDER = "provider"

CUTTER_DOCAI = CHUNK_DOCAI

class RecutError(Exception):
    ...


@dataclass(frozen=True, slots=True)
class RecutResult:

    chunks: tuple[Chunk, ...]
    path: str
    artifact: ParsedArtifact
    used_reader: bool

    @property
    def pages(self) -> int | None:
        return self.artifact.page_count


Cut = Callable[..., MappedDocument]


def cut_payload(
    payload: dict,
    *,
    cut: Cut,
    external_id: str,
    stamps: VersionStamps,
    filename: str = "",
    media_type: str = "",
    content_digest: str = "",
) -> tuple[Chunk, ...]:
    return cut(
        payload,
        external_id=external_id,
        stamps=stamps,
        filename=filename,
        media_type=media_type,
        content_digest=content_digest,
    ).chunks


CUT_SAMPLES = Path(__file__).with_name("cut_samples")

SAMPLE_READERS: tuple[tuple[str, Cut], ...] = (
    (READER_DOCAI, docai.chunks_from_document),
    (PROCESS_PDFTEXT, pdftext.chunks_from_document),
)


SAMPLE_STAMPS = VersionStamps(reader_version="sample", fingerprint="sample")

# Allowlist: a field joins the cut version only when changing it must re-cut stored documents
PIECE_FIELDS: tuple[str, ...] = (
    "ordinal",
    "content",
    "page_start",
    "page_end",
    "heading_path",
    "source_block_ids",
    "block_kinds",
)


@cache
# Digest of the pieces, not of code: ast/bytecode differ between Python 3.12 and 3.13
def cut_version(module: str) -> str:
    return _digest_of(sample_pieces(module))


def sample_pieces(module: str | None = None) -> dict[str, list[dict[str, Any]]]:
    cut: dict[str, list[dict[str, Any]]] = {}
    for reader, module_cut in SAMPLE_READERS:
        if module is not None and reader != module:
            continue
        directory = CUT_SAMPLES / reader
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise RecutError(
                f"the read module {reader!r} has no cut samples in {directory}; without them "
                "the cut version is a constant, and a changed cut would go unnoticed by "
                "every deployment that trusts it"
            )
        for path in paths:
            pieces = cut_payload(
                json.loads(path.read_text(encoding="utf-8")),
                cut=module_cut,
                external_id=path.stem,
                stamps=SAMPLE_STAMPS,
            )
            cut[f"{reader}/{path.name}"] = [
                {field: _plain(getattr(piece, field)) for field in PIECE_FIELDS}
                for piece in pieces
            ]
    return cut


def _plain(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _digest_of(pieces: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(pieces), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def recut_document(
    conn: psycopg.Connection,
    store: ArtifactStore,
    *,
    document_id: int,
    chunking: ChunkingOptions,
    stamps: VersionStamps,
    cut: Cut,
    reader: LayoutParser | None = None,
) -> RecutResult:
    artifact = require_parsed_artifact(conn, document_id)
    payload = load_parsed_artifact(conn, store, artifact)
    if reader is not None:
        payload = reader.recut(payload, chunking)
    return RecutResult(
        chunks=cut_payload(
            payload,
            cut=cut,
            external_id=artifact.external_id,
            stamps=stamps,
        ),
        path=RECUT_PROVIDER if reader is not None else RECUT_LOCAL,
        artifact=artifact,
        used_reader=reader is not None,
    )
