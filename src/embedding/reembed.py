from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psycopg

from src.embedding.batch import EmbeddingRun, Group, embed_batches, plan_batches
from src.embedding.client import Embedder
from src.store import ModelStamp, Piece, upsert_pieces
from src.store_versions import activate_version, pieces_under


class ReembedError(Exception):
    ...


@dataclass(frozen=True, slots=True)
class StoredPiece:

    document_id: int
    ordinal: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    metadata: dict | None = None

    def with_vector(self, embedding: Sequence[float]) -> Piece:
        return Piece(
            ordinal=self.ordinal,
            content=self.content,
            embedding=embedding,
            page_start=self.page_start,
            page_end=self.page_end,
            heading_path=self.heading_path,
            metadata=dict(self.metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class ReembedOutcome:

    run: EmbeddingRun
    documents_written: tuple[int, ...]
    pieces_written: int
    failed_documents: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return not self.failed_documents


def stored_pieces(
    conn: psycopg.Connection,
    stamp: ModelStamp,
    *,
    document_ids: Sequence[int] | None = None,
) -> tuple[StoredPiece, ...]:
    query = """
        SELECT document_id, ordinal, content, page_start, page_end, heading_path, metadata
          FROM chunks
         WHERE embed_model = %(model)s AND embed_dimension = %(dimension)s
    """
    params: dict = {"model": stamp.model, "dimension": stamp.dimension}
    if document_ids is not None:
        query += " AND document_id = ANY(%(documents)s)"
        params["documents"] = list(document_ids)
    query += " ORDER BY document_id, ordinal"
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    return tuple(
        StoredPiece(
            document_id=int(row[0]),
            ordinal=int(row[1]),
            content=row[2],
            page_start=row[3],
            page_end=row[4],
            heading_path=tuple(row[5]),
            metadata=dict(row[6] or {}),
        )
        for row in rows
    )


def group_by_document(pieces: Sequence[StoredPiece]) -> tuple[Group, ...]:
    grouped: dict[int, list[StoredPiece]] = {}
    for piece in pieces:
        grouped.setdefault(piece.document_id, []).append(piece)
    return tuple(
        Group(key=str(document_id), texts=tuple(piece.content for piece in members))
        for document_id, members in grouped.items()
    )


def reembed(
    conn: psycopg.Connection,
    *,
    source: ModelStamp,
    target: ModelStamp,
    embedder: Embedder,
    batch_size: int,
    max_inputs: int,
    document_ids: Sequence[int] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReembedOutcome:
    if source == target:
        raise ReembedError(
            f"re-embedding {source.model} at {source.dimension} into itself would "
            "delete each document's pieces and write them back unchanged, at the price "
            "of embedding the whole corpus"
        )
    if embedder.dimension != target.dimension:
        raise ReembedError(
            f"the embedder returns {embedder.dimension} dimensions and the target stamp "
            f"says {target.dimension}; one of the two is not what the store's column holds"
        )
    if embedder.model != target.model:
        raise ReembedError(
            f"the embedder calls {embedder.model!r} and the target stamp says "
            f"{target.model!r}; a row stamped with a model that did not produce its "
            "vector is the corruption this stamp exists to prevent"
        )
    pieces = stored_pieces(conn, source, document_ids=document_ids)
    if not pieces:
        raise ReembedError(
            f"nothing is stored under {source.model} at {source.dimension} dimensions, "
            "so there is nothing to embed again"
        )
    by_document: dict[int, list[StoredPiece]] = {}
    for piece in pieces:
        by_document.setdefault(piece.document_id, []).append(piece)
    run = embed_batches(
        plan_batches(group_by_document(pieces), batch_size=batch_size, max_inputs=max_inputs),
        embedder,
        clock=clock,
    )
    vectors = run.vectors()
    written: list[int] = []
    pieces_written = 0
    for document_id, members in by_document.items():
        produced = vectors.get(str(document_id))
        if produced is None:
            continue
        if len(produced) != len(members):
            raise ReembedError(
                f"document {document_id} sent {len(members)} piece(s) and {len(produced)} "
                "vector(s) came back; writing them paired by position would give every "
                "piece after the gap its neighbour's meaning"
            )
        pieces_written += upsert_pieces(
            conn,
            document_id=document_id,
            stamp=target,
            pieces=[piece.with_vector(vector) for piece, vector in zip(members, produced)],
        )
        written.append(document_id)
    return ReembedOutcome(
        run=run,
        documents_written=tuple(written),
        pieces_written=pieces_written,
        failed_documents=tuple(int(key) for key in run.failed_keys),
    )


def cutover(conn: psycopg.Connection, target: ModelStamp) -> ModelStamp | None:
    if pieces_under(conn, target) == 0:
        raise ReembedError(
            f"no pieces are stored under {target.model} at {target.dimension} dimensions; "
            "activating it would leave every search returning nothing while every row is "
            "still in the table"
        )
    return activate_version(conn, target)
