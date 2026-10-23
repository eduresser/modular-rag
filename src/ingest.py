from __future__ import annotations

import time
from datetime import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Protocol

import psycopg

from src.config import AppConfig, DocaiChunkSettings
from src.embedding.batch import Group, embed_batches, plan_batches
from src.embedding.client import Embedder, EmbeddingError
from src.embedding.cost import provider_of, record_run
from src.embedding.reembed import StoredPiece, stored_pieces
from src.embedding.stamp import stamp_from_config
from src.fingerprint import StageFingerprints, stage_fingerprints
from src.ingest_errors import (
    STAGE_CUT,
    STAGE_EMBED,
    STAGE_FETCH,
    STAGE_READ,
    STAGE_STORE,
    FailureLog,
    FailureRecord,
    NothingToIndex,
    ReadRefused,
    failure_of,
)
from src.ingest_hooks import NO_HOOKS, StageHooks
from src.ingest_idempotency import (
    APP_VERSION,
    CLAIM_STATE_DONE,
    ClaimOutcome,
    claim,
    fail,
    settle,
)
from src.models import Chunk, VersionStamps
from src.prices import PriceBook
from src.processing.artifact import (
    ArchiveStore,
    ArtifactStore,
    ObjectStore,
    read_batch_output,
    record_read_failure,
    store_parsed_artifact,
)
from src.processing.cost import PROVIDER_LOCAL, PROVIDER_READER, record_read, record_recut
from src.processing.docai import ChunkingOptions, InputDocument, LayoutParser
from src.processing.docai import plan_batches as plan_read_batches
from src.processing.render import docai_rendering_for, render
from src.processing.preflight import DOCAI_LIMITS, Candidate, Limits, record_refusals, screen
from src.processing.recut import RECUT_LOCAL, cut_payload, cut_version, recut_document
from src.processing.to_chunks import MappedDocument
from src.processing import pdftext
from src.processing.docai import READER_NAME, index_blocks
from src.processing.docai import chunks_from_document as docai_cut
from src.reprocess import (
    ACTION_READ,
    ACTION_RECUT,
    ACTION_REEMBED,
    ACTION_TOUCH,
    IngestState,
    Reprocessing,
    decide,
    record_state,
    state_by_external_id,
    touch,
)
from src.sources.base import FetchedDocument, Source, SourceDocument
from src.staleness import record_generation
from src.store import ModelStamp, Piece, upsert_document, upsert_pieces
from src.store_versions import activate_version, current_version
from src.usage import OUTCOME_FAILED, REASON_OPERATOR_REQUESTED

READ_OUTPUT_PREFIX = "reader-output"


@dataclass(frozen=True, slots=True)
class ReadRequest:
    document: SourceDocument
    document_id: int
    fetched: FetchedDocument
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class ReadOutcome:
    payloads: Mapping[str, dict]
    pages: Mapping[str, int]
    failures: Mapping[str, str]
    duration_ms: Decimal = Decimal(0)


class ReaderStep(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...

    @property
    def provider(self) -> str:
        ...

    @property
    def limits(self) -> Limits:
        ...

    def rendering_for(self, media_type: str) -> str:
        ...

    def read(self, conn: psycopg.Connection, requests: Sequence[ReadRequest]) -> ReadOutcome:
        ...

    def cut(
        self,
        payload: dict,
        *,
        external_id: str,
        stamps: VersionStamps,
        filename: str = "",
        media_type: str = "",
        content_digest: str = "",
    ) -> MappedDocument:
        ...


@dataclass(frozen=True, slots=True)
class IngestResult:
    external_id: str
    action: str
    document_id: int | None = None
    reason: str | None = None
    pieces: int = 0
    duplicate: bool = False
    failure: FailureRecord | None = None
    cut_off: bool = False

    @property
    def indexed(self) -> bool:
        return self.failure is None and not self.duplicate


@dataclass
class IngestRun:
    results: list[IngestResult] = field(default_factory=list)
    failures: FailureLog = field(default_factory=FailureLog)
    stamp: ModelStamp | None = None

    @property
    def indexed(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.indexed and result.action != ACTION_TOUCH)

    @property
    def touched(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.action == ACTION_TOUCH and result.indexed)

    @property
    def duplicates(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.duplicate)

    @property
    def cut_off(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.cut_off)

    @property
    def errored(self) -> int:
        return self.failures.count

    def by_action(self, action: str) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.action == action)

    def result_for(self, external_id: str) -> IngestResult | None:
        for result in self.results:
            if result.external_id == external_id:
                return result
        return None


@dataclass(frozen=True, slots=True)
class Pipeline:
    config: AppConfig
    source: Source
    reader: ReaderStep
    store: ArtifactStore
    embedder: Embedder
    prices: PriceBook
    hooks: StageHooks = NO_HOOKS
    clock: Callable[[], float] = time.monotonic
    resume: "Resume | None" = None
    activity: "Callable[[datetime | None], Iterable[ActivityEvent]] | None" = None
    checkpoint: Callable[[], None] = lambda: None

    @property
    def client(self) -> str:
        return self.config.telemetry.client

    @property
    def stamp(self) -> ModelStamp:
        return stamp_from_config(self.config)

    @property
    def fingerprints(self) -> StageFingerprints:
        return stage_fingerprints(self.config)

    @property
    def chunking(self) -> ChunkingOptions | None:
        chunk = self.config.chunk
        if not isinstance(chunk, DocaiChunkSettings):
            return None
        return ChunkingOptions(
            size=chunk.size,
            include_ancestor_headings=chunk.include_headings,
            cutter=self.config.process.type,
        )

    @property
    def cut_version(self) -> str:
        return cut_version(self.config.process.type)


@dataclass
class _Work:
    document: SourceDocument
    document_id: int
    decision: Reprocessing
    content_digest: str
    stored: IngestState | None
    claim_id: int | None = None
    fetched: FetchedDocument | None = None
    chunks: tuple[Chunk, ...] = ()
    reader_version: str = ""
    pages: int = 0
    done: bool = False


def commit_document(
    conn: psycopg.Connection,
    *,
    document_id: int,
    stamp: ModelStamp,
    pieces: Sequence[Piece],
    content_digest: str,
    fingerprints: StageFingerprints,
    reader_version: str,
    cut_version: str,
    revision_marker: str | None = None,
    process_module: str = "",
) -> int:
    with conn.transaction():
        written = upsert_pieces(conn, document_id=document_id, stamp=stamp, pieces=pieces)
        record_state(
            conn,
            document_id=document_id,
            content_digest=content_digest,
            fingerprints=fingerprints,
            reader_version=reader_version,
            cut_version=cut_version,
            piece_count=written,
            revision_marker=revision_marker,
            process_module=process_module,
        )
    return written


def ingest_documents(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    documents: Sequence[SourceDocument],
    *,
    run_id: int | None = None,
    operator_requested: bool = False,
    app_version: str = APP_VERSION,
) -> IngestRun:
    run = IngestRun(stamp=pipeline.stamp)
    record_generation(conn, pipeline.config)
    work = _decide_all(
        conn, pipeline, documents, run, run_id=run_id,
        operator_requested=operator_requested, app_version=app_version,
    )
    _read_phase(conn, pipeline, work, run, run_id=run_id)
    pipeline.checkpoint()
    _cut_phase(conn, pipeline, work, run)
    _cutoff_phase(conn, pipeline, work, run)
    _embed_and_commit(conn, pipeline, work, run)
    _activate_first_generation(conn, pipeline, run)
    pipeline.checkpoint()
    return run


def _decide_all(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    documents: Sequence[SourceDocument],
    run: IngestRun,
    *,
    run_id: int | None,
    operator_requested: bool,
    app_version: str,
) -> list[_Work]:
    fingerprints = pipeline.fingerprints
    pending: list[_Work] = []
    claimed: set[int] = set()
    for document in documents:
        stored = state_by_external_id(conn, document.external_id)
        needs_bytes = (
            stored is None
            or operator_requested
            or document.revision_marker is None
            or document.revision_marker != stored.revision_marker
        )
        document_id = upsert_document(
            conn,
            external_id=document.external_id,
            title=document.name,
            metadata={"parents": list(document.parents)} if document.parents else None,
            media_type=document.kind,
            link=document.link,
            location=list(document.parents),
            modified_at=document.modified_at,
            size_bytes=document.size_bytes,
        )
        fetched: FetchedDocument | None = None

        def _fetch() -> FetchedDocument | None:
            try:
                return pipeline.hooks.apply_before_send(document, pipeline.source.fetch(document))
            except Exception as error:  # noqa: BLE001 — one document, not the run
                run.results.append(
                    _record_failure(
                        conn, pipeline, run, external_id=document.external_id,
                        stage=STAGE_FETCH, error=error, document_id=document_id, claim_id=None,
                    )
                )
                return None

        if needs_bytes:
            fetched = _fetch()
            if fetched is None:
                continue
            digest = fetched.content_digest
        else:
            digest = stored.content_digest  # type: ignore[union-attr]
        decision = decide(
            stored=stored,
            content_digest=digest,
            fingerprints=fingerprints,
            marker_moved=needs_bytes and stored is not None,
            operator_requested=operator_requested,
        )
        if decision.action == ACTION_READ and fetched is None:
            fetched = _fetch()
            if fetched is None:
                continue

        if decision.action == ACTION_TOUCH:
            touch(conn, document_id=document_id, revision_marker=document.revision_marker)
            run.results.append(
                IngestResult(
                    external_id=document.external_id,
                    action=ACTION_TOUCH,
                    document_id=document_id,
                    reason=decision.reason,
                    pieces=stored.piece_count if stored else 0,
                )
            )
            continue
        outcome = claim(
            conn,
            external_id=document.external_id,
            content_digest=digest,
            fingerprint=fingerprints.whole,
            document_id=document_id,
            run_id=run_id,
            app_version=app_version,
        )
        if outcome.claim.id in claimed or _settled_and_still_true(outcome, stored):
            touch(conn, document_id=document_id, revision_marker=document.revision_marker)
            run.results.append(
                IngestResult(
                    external_id=document.external_id,
                    action=outcome.claim.action or decision.action,
                    document_id=document_id,
                    reason=None,
                    duplicate=True,
                )
            )
            continue
        claimed.add(outcome.claim.id)
        pending.append(
            _Work(
                document=document,
                document_id=document_id,
                decision=decision,
                content_digest=digest,
                stored=stored,
                claim_id=outcome.claim.id,
                fetched=fetched,
                reader_version=stored.reader_version if stored else "",
            )
        )
    return pending


def _settled_and_still_true(outcome: ClaimOutcome, stored: IngestState | None) -> bool:
    if outcome.should_run:
        return False
    if outcome.claim.state != CLAIM_STATE_DONE:
        return True
    return stored is not None and stored.fingerprint == outcome.claim.fingerprint


def _rendered(
    conn: psycopg.Connection,
    pipeline: "Pipeline",
    run: "_Run",
    reading: "list[_Work]",
) -> "list[_Work]":
    kept: list[_Work] = []
    for item in reading:
        fetched = item.fetched
        if fetched is None or not pipeline.reader.rendering_for(fetched.media_type):
            kept.append(item)
            continue
        try:
            rendered = render(
                item.document.external_id,
                fetched.media_type,
                fetched.content,
                to=pipeline.reader.rendering_for(fetched.media_type),
            )
        except Exception as error:  # noqa: BLE001 — one document, not the run
            item.done = True
            run.results.append(
                _record_failure(
                    conn, pipeline, run, external_id=item.document.external_id,
                    stage=STAGE_READ, error=error,
                    document_id=item.document_id, claim_id=item.claim_id,
                    reason=item.decision.reason, action=item.decision.action,
                )
            )
            continue
        item.fetched = replace(
            fetched, content=rendered.content, media_type=rendered.media_type
        )
        kept.append(item)
    return kept


def _read_phase(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    work: Sequence[_Work],
    run: IngestRun,
    *,
    run_id: int | None,
) -> None:
    reading = [item for item in work if item.decision.action == ACTION_READ and not item.done]
    if not reading:
        return
    reading = _rendered(conn, pipeline, run, reading)
    if not reading:
        return
    by_id = {item.document.external_id: item for item in reading}
    screening = screen(
        [
            Candidate(
                document_id=item.document_id,
                external_id=item.document.external_id,
                media_type=item.fetched.media_type,  # type: ignore[union-attr]
                byte_size=len(item.fetched.content),  # type: ignore[union-attr]
                page_count=None,
            )
            for item in reading
        ],
        limits=pipeline.reader.limits,
    )
    record_refusals(
        conn, screening, reader=pipeline.reader.name, reader_version=pipeline.reader.version
    )
    for verdict in screening.rejected:
        item = by_id[verdict.candidate.external_id]
        item.done = True
        run.results.append(
            _record_failure(
                conn, pipeline, run, external_id=item.document.external_id,
                stage=STAGE_READ, error=ReadRefused(verdict.reason),
                document_id=item.document_id, claim_id=item.claim_id,
                reason=item.decision.reason, action=item.decision.action,
            )
        )
    accepted = [by_id[candidate.external_id] for candidate in screening.accepted]
    if not accepted:
        return
    try:
        outcome = pipeline.reader.read(
            conn,
            [
                ReadRequest(
                    document=item.document,
                    document_id=item.document_id,
                    fetched=item.fetched,  # type: ignore[arg-type]
                )
                for item in accepted
            ],
        )
    except Exception as error:  # noqa: BLE001 — the reader itself, not one document
        for item in accepted:
            item.done = True
            run.results.append(
                _record_failure(
                    conn, pipeline, run, external_id=item.document.external_id,
                    stage=STAGE_READ, error=error, document_id=item.document_id,
                    claim_id=item.claim_id, reason=item.decision.reason,
                    action=item.decision.action,
                )
            )
        return
    for item in accepted:
        external_id = item.document.external_id
        reason = outcome.failures.get(external_id)
        if reason is not None or external_id not in outcome.payloads:
            item.done = True
            run.results.append(
                _record_failure(
                    conn, pipeline, run, external_id=external_id, stage=STAGE_READ,
                    error=ReadRefused(reason or "the reader returned no output for it"),
                    document_id=item.document_id, claim_id=item.claim_id,
                    reason=item.decision.reason, pages=outcome.pages.get(external_id, 0),
                    action=item.decision.action,
                )
            )
            continue
        item.pages = outcome.pages.get(external_id, 0)
        item.reader_version = pipeline.reader.version
        # The read is billed whether or not the cut that follows succeeds: the
        # pages were read and the invoice will say so (AC-160).
        record_read(
            conn,
            pipeline.prices,
            client=pipeline.client,
            document_id=item.document_id,
            reader_version=pipeline.reader.version,
            pages=item.pages,
            reason=item.decision.reason or REASON_OPERATOR_REQUESTED,
            duration_ms=outcome.duration_ms,
            provider=pipeline.reader.provider,
        )
        payload = pipeline.hooks.apply_after_read(external_id, outcome.payloads[external_id])
        try:
            item.chunks = cut_payload(
                payload,
                cut=pipeline.reader.cut,
                external_id=external_id,
                filename=item.document.name,
                media_type=item.fetched.media_type if item.fetched else "",
                content_digest=item.content_digest,
                stamps=VersionStamps(
                    reader_version=pipeline.reader.version,
                    fingerprint=pipeline.fingerprints.whole,
                ),
            )
        except Exception as error:  # noqa: BLE001 — one document, not the run
            item.done = True
            run.results.append(
                _record_failure(
                    conn, pipeline, run, external_id=external_id, stage=STAGE_CUT,
                    error=error, document_id=item.document_id, claim_id=item.claim_id,
                    reason=item.decision.reason, pages=item.pages,
                    action=item.decision.action,
                )
            )
            continue


def _cut_phase(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    work: Sequence[_Work],
    run: IngestRun,
) -> None:
    for item in work:
        if item.done or item.decision.action != ACTION_RECUT:
            continue
        started = pipeline.clock()
        try:
            result = recut_document(
                conn,
                pipeline.store,
                document_id=item.document_id,
                chunking=pipeline.chunking,
                cut=pipeline.reader.cut,
                stamps=VersionStamps(
                    reader_version=item.stored.reader_version if item.stored else pipeline.reader.version,
                    fingerprint=pipeline.fingerprints.whole,
                ),
            )
        except Exception as error:  # noqa: BLE001 — one document, not the run
            item.done = True
            run.results.append(
                _record_failure(
                    conn, pipeline, run, external_id=item.document.external_id,
                    stage=STAGE_CUT, error=error, document_id=item.document_id,
                    claim_id=item.claim_id, reason=item.decision.reason,
                    action=item.decision.action,
                )
            )
            continue
        item.chunks = result.chunks
        item.reader_version = result.artifact.reader_version
        item.pages = result.pages or 0
        # A local cut is billed to nobody and still writes a row: "free" and
        # "unpriced" are different answers, and only one of them is true here.
        record_recut(
            conn,
            pipeline.prices,
            client=pipeline.client,
            document_id=item.document_id,
            reader_version=result.artifact.reader_version,
            pages=item.pages,
            reason=item.decision.reason or REASON_OPERATOR_REQUESTED,
            duration_ms=_elapsed_ms(started, pipeline.clock()),
            path=RECUT_LOCAL if not result.used_reader else result.path,
        )


def _cutoff_phase(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    work: Sequence[_Work],
    run: IngestRun,
) -> None:
    for item in work:
        if item.done or item.decision.action == ACTION_REEMBED or not item.chunks:
            continue
        stored = stored_pieces(conn, pipeline.stamp, document_ids=[item.document_id])
        if not _same_pieces(item.chunks, stored):
            continue
        # The pieces are already right, so only the bookkeeping is missing.
        record_state(
            conn,
            document_id=item.document_id,
            content_digest=item.content_digest,
            fingerprints=pipeline.fingerprints,
            reader_version=item.reader_version or pipeline.reader.version,
            cut_version=pipeline.cut_version,
            piece_count=len(stored),
            process_module=pipeline.config.process.type,
        )
        item.done = True
        run.results.append(
            IngestResult(
                external_id=item.document.external_id,
                action=item.decision.action,
                document_id=item.document_id,
                reason=item.decision.reason,
                pieces=len(stored),
                cut_off=True,
            )
        )


def _same_pieces(cut: Sequence[Chunk], stored: Sequence[StoredPiece]) -> bool:
    if len(cut) != len(stored):
        return False
    return all(
        piece.ordinal == chunk.ordinal
        and piece.content == chunk.content
        and piece.page_start == chunk.page_start
        and piece.page_end == chunk.page_end
        and tuple(piece.heading_path) == tuple(chunk.heading_path)
        for chunk, piece in zip(cut, stored)
    )


def _embed_and_commit(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    work: Sequence[_Work],
    run: IngestRun,
) -> None:
    pending = [item for item in work if not item.done]
    if not pending:
        return
    stamp = pipeline.stamp
    provider = provider_of(pipeline.config.embed.base_url)
    for reason in sorted({item.decision.reason or REASON_OPERATOR_REQUESTED for item in pending}):
        batch = [
            item
            for item in pending
            if (item.decision.reason or REASON_OPERATOR_REQUESTED) == reason
        ]
        texts: dict[str, tuple[str, ...]] = {}
        carriers: dict[str, list[Chunk | StoredPiece]] = {}
        for item in batch:
            pieces = _texts_of(conn, item, stamp)
            if not pieces:
                item.done = True
                run.results.append(
                    _record_failure(
                        conn, pipeline, run, external_id=item.document.external_id,
                        stage=STAGE_CUT,
                        error=NothingToIndex(
                            f"{item.document.external_id} produced no piece to index; a "
                            "document with no citable text is not indexed rather than "
                            "indexed empty"
                        ),
                        document_id=item.document_id, claim_id=item.claim_id, reason=reason,
                        action=item.decision.action,
                    )
                )
                continue
            carriers[item.document.external_id] = list(pieces)
            texts[item.document.external_id] = tuple(piece.content for piece in pieces)
        if not texts:
            continue
        groups = tuple(Group(key=key, texts=value) for key, value in texts.items())
        embedding = embed_batches(
            plan_batches(
                groups,
                batch_size=pipeline.config.embed.batch_size,
                max_inputs=pipeline.config.embed.max_inputs_per_request,
            ),
            pipeline.embedder,
            clock=pipeline.clock,
        )
        record_run(
            conn,
            pipeline.prices,
            client=pipeline.client,
            run=embedding,
            model=pipeline.config.embed.model,
            provider=provider,
            reason=reason,
            document_ids={
                item.document.external_id: item.document_id
                for item in batch
                if item.document.external_id in texts
            },
        )
        vectors = embedding.vectors()
        for item in batch:
            if item.done:
                continue
            external_id = item.document.external_id
            produced = vectors.get(external_id)
            if produced is None or len(produced) != len(carriers[external_id]):
                item.done = True
                run.results.append(
                    _record_failure(
                        conn, pipeline, run, external_id=external_id, stage=STAGE_EMBED,
                        error=_not_embedded(
                            external_id,
                            sent=len(carriers[external_id]),
                            back=0 if produced is None else len(produced),
                            refusal=embedding.refusal_for(external_id),
                            status=embedding.refusal_status_for(external_id),
                        ),
                        document_id=item.document_id, claim_id=item.claim_id, reason=reason,
                        action=item.decision.action,
                    )
                )
                continue
            _commit(conn, pipeline, item, run, carriers[external_id], produced, stamp)


def _not_embedded(
    external_id: str, *, sent: int, back: int, refusal: str, status: int | None = None
) -> Exception:
    missing = (
        f"{external_id} sent {sent} piece(s) and {back} vector(s) came back; "
        "a document is embedded whole or not at all (AC-057)"
    )
    if refusal:
        return EmbeddingError(f"{missing} — the provider said: {refusal}", status=status)
    return NothingToIndex(missing)


def _commit(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    item: _Work,
    run: IngestRun,
    carriers: Sequence[Chunk | StoredPiece],
    vectors: Sequence[Sequence[float]],
    stamp: ModelStamp,
) -> None:
    pieces = [
        carrier.to_piece(vector) if isinstance(carrier, Chunk) else carrier.with_vector(vector)
        for carrier, vector in zip(carriers, vectors)
    ]
    try:
        written = commit_document(
            conn,
            document_id=item.document_id,
            stamp=stamp,
            pieces=pieces,
            content_digest=item.content_digest,
            fingerprints=pipeline.fingerprints,
            reader_version=item.reader_version or pipeline.reader.version,
            cut_version=pipeline.cut_version,
            revision_marker=item.document.revision_marker,
            process_module=pipeline.config.process.type,
        )
    except Exception as error:  # noqa: BLE001 — one document, not the run
        item.done = True
        run.results.append(
            _record_failure(
                conn, pipeline, run, external_id=item.document.external_id,
                stage=STAGE_STORE, error=error, document_id=item.document_id,
                claim_id=item.claim_id, reason=item.decision.reason,
            )
        )
        return
    item.done = True
    if item.claim_id is not None:
        settle(conn, item.claim_id, action=item.decision.action, document_id=item.document_id)
    run.results.append(
        IngestResult(
            external_id=item.document.external_id,
            action=item.decision.action,
            document_id=item.document_id,
            reason=item.decision.reason,
            pieces=written,
        )
    )
    pipeline.checkpoint()


def _activate_first_generation(
    conn: psycopg.Connection, pipeline: Pipeline, run: IngestRun
) -> None:
    if current_version(conn) is not None or not run.indexed:
        return
    activate_version(conn, pipeline.stamp)


def _texts_of(
    conn: psycopg.Connection, item: _Work, stamp: ModelStamp
) -> tuple[Chunk | StoredPiece, ...]:
    if item.decision.action != ACTION_REEMBED:
        return item.chunks
    source = _stamp_of_document(conn, item.document_id)
    if source is None:
        return ()
    return stored_pieces(conn, source, document_ids=[item.document_id])


def _stamp_of_document(conn: psycopg.Connection, document_id: int) -> ModelStamp | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT embed_model, embed_dimension
              FROM chunks
             WHERE document_id = %s
             ORDER BY ordinal
             LIMIT 1
            """,
            (document_id,),
        )
        row = cursor.fetchone()
    return ModelStamp(model=row[0], dimension=int(row[1])) if row is not None else None


def _record_failure(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    run: IngestRun,
    *,
    external_id: str,
    stage: str,
    error: BaseException,
    document_id: int | None,
    claim_id: int | None,
    reason: str | None = None,
    pages: int = 0,
    action: str = ACTION_TOUCH,
) -> IngestResult:
    failure = run.failures.add(
        failure_of(external_id=external_id, stage=stage, error=error, document_id=document_id)
    )
    if claim_id is not None:
        fail(conn, claim_id, reason=failure.reason, kind=failure.kind, document_id=document_id)
    if stage == STAGE_READ:
        record_read(
            conn,
            pipeline.prices,
            client=pipeline.client,
            document_id=document_id,
            reader_version=pipeline.reader.version,
            pages=pages,
            reason=reason or REASON_OPERATOR_REQUESTED,
            duration_ms=0,
            outcome=OUTCOME_FAILED,
        )
    return IngestResult(
        external_id=external_id,
        action=action,
        document_id=document_id,
        reason=reason,
        failure=failure,
    )


def _elapsed_ms(started: float, finished: float) -> Decimal:
    return Decimal(str(round(max(finished - started, 0.0) * 1000, 3)))


@dataclass
class DocumentAIReader:
    reader: LayoutParser
    store: ObjectStore
    max_documents_per_batch: int = 1_000
    clock: Callable[[], float] = time.monotonic
    input_prefix: str = "source"
    page_limit: int | None = None
    max_document_bytes: int | None = None

    @property
    def name(self) -> str:
        return READER_NAME

    @property
    def version(self) -> str:
        return self.reader.processor_version

    @property
    def provider(self) -> str:
        return PROVIDER_READER

    @property
    def limits(self) -> Limits:
        return DOCAI_LIMITS.narrowed(
            pages=self.page_limit, byte_size=self.max_document_bytes
        )

    def rendering_for(self, media_type: str) -> str:
        return docai_rendering_for(media_type)

    def cut(
        self,
        payload: dict,
        *,
        external_id: str,
        stamps: VersionStamps,
        filename: str = "",
        media_type: str = "",
        content_digest: str = "",
    ) -> MappedDocument:
        return docai_cut(
            payload,
            external_id=external_id,
            stamps=stamps,
            filename=filename,
            media_type=media_type,
            content_digest=content_digest,
        )

    def read(self, conn: psycopg.Connection, requests: Sequence[ReadRequest]) -> ReadOutcome:
        if not requests:
            return ReadOutcome(payloads={}, pages={}, failures={})
        by_uri: dict[str, ReadRequest] = {}
        inputs: list[InputDocument] = []
        for request in requests:
            name = f"{self.input_prefix}/{request.document.external_id}"
            uri = self.store.put(
                conn, name, request.fetched.content, media_type=request.fetched.media_type
            )
            by_uri[uri] = request
            inputs.append(
                InputDocument(
                    external_id=request.document.external_id,
                    uri=uri,
                    media_type=request.fetched.media_type,
                )
            )
        started = self.clock()
        outcome = self.reader.run(
            plan_read_batches(
                inputs,
                output_prefix=self.store.uri_for(READ_OUTPUT_PREFIX),
                max_documents_per_batch=self.max_documents_per_batch,
            )
        )
        duration = _elapsed_ms(started, self.clock())
        payloads: dict[str, dict] = {}
        pages: dict[str, int] = {}
        failures: dict[str, str] = {}
        for document in outcome.documents():
            request = by_uri.get(document.uri)
            if request is None:
                continue
            if not document.succeeded or not document.output_uri:
                failures[document.external_id] = (
                    document.detail or "the reader reported no output for this document"
                )
                continue
            payload = read_batch_output(conn, self.store, document.output_uri)
            page_count = pages_in(payload)
            store_parsed_artifact(
                conn,
                self.store,
                document_id=request.document_id,
                external_id=document.external_id,
                payload=payload,
                reader=READER_NAME,
                reader_version=self.version,
                page_count=page_count,
            )
            payloads[document.external_id] = payload
            pages[document.external_id] = page_count
        return ReadOutcome(
            payloads=payloads, pages=pages, failures=failures, duration_ms=duration
        )


def _extract_pdf_text(content: bytes, external_id: str) -> dict:
    return pdftext.extract(content, external_id=external_id)


def pages_in(payload: dict) -> int:
    blocks = index_blocks(payload).in_order()
    return max((block.page_end or 0) for block in blocks) if blocks else 0


@dataclass(frozen=True, slots=True)
class LocalTextReader:
    #: The archive itself. This module reaches no provider, so it keeps what it
    #: read where the rest of the deployment already is (AC-272).
    store: ArchiveStore = field(default_factory=ArchiveStore)
    extract: Callable[[bytes, str], dict] = _extract_pdf_text

    @property
    def name(self) -> str:
        return pdftext.READER_NAME

    @property
    def version(self) -> str:
        return pdftext.extractor_version()

    @property
    def provider(self) -> str:
        return PROVIDER_LOCAL

    @property
    def limits(self) -> Limits:
        return pdftext.LIMITS

    def rendering_for(self, media_type: str) -> str:
        return pdftext.rendering_for(media_type)

    def cut(
        self,
        payload: dict,
        *,
        external_id: str,
        stamps: VersionStamps,
        filename: str = "",
        media_type: str = "",
        content_digest: str = "",
    ) -> MappedDocument:
        return pdftext.chunks_from_document(
            payload,
            external_id=external_id,
            stamps=stamps,
            filename=filename,
            media_type=media_type,
            content_digest=content_digest,
        )

    def read(self, conn: psycopg.Connection, requests: Sequence[ReadRequest]) -> ReadOutcome:
        payloads: dict[str, dict] = {}
        pages: dict[str, int] = {}
        failures: dict[str, str] = {}
        for request in requests:
            external_id = request.document.external_id
            if request.fetched is None:  # pragma: no cover — the pipeline always fetches first
                failures[external_id] = "no bytes were fetched for this document"
                continue
            try:
                payload = self.extract(request.fetched.content, external_id)
            except pdftext.PdfTextError as error:
                failures[external_id] = str(error)
                record_read_failure(
                    conn,
                    document_id=request.document_id,
                    reason=str(error),
                    reader=self.name,
                    reader_version=self.version,
                    byte_size=len(request.fetched.content),
                )
                continue
            page_count = pdftext.pages_in(payload)
            store_parsed_artifact(
                conn,
                self.store,
                document_id=request.document_id,
                external_id=external_id,
                payload=payload,
                reader=self.name,
                reader_version=self.version,
                page_count=page_count,
            )
            payloads[external_id] = payload
            pages[external_id] = page_count
        return ReadOutcome(payloads=payloads, pages=pages, failures=failures)
