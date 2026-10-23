from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

from src.fingerprint import STAGE_CUT, STAGE_EMBED, STAGE_READ, STAGES, StageFingerprints
from src.ingest_idempotency import CLAIM_STATE_FAILED, FAILURE_TRANSIENT
from src.usage import (
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_OPERATOR_REQUESTED,
    REASON_WATERMARK_MOVED,
)
from migrations import load as load_migration

_migration = load_migration("006_ingest")

ACTION_TOUCH: str = _migration.ACTION_TOUCH
ACTION_REEMBED: str = _migration.ACTION_REEMBED
ACTION_RECUT: str = _migration.ACTION_RECUT
ACTION_READ: str = _migration.ACTION_READ
ACTIONS: tuple[str, ...] = _migration.ACTIONS

_STAGE_ACTIONS: tuple[tuple[str, str], ...] = (
    (STAGE_READ, ACTION_READ),
    (STAGE_CUT, ACTION_RECUT),
    (STAGE_EMBED, ACTION_REEMBED),
)


@dataclass(frozen=True, slots=True)
class IngestState:
    document_id: int
    content_digest: str
    fingerprint: str
    read_fingerprint: str
    cut_fingerprint: str
    embed_fingerprint: str
    reader_version: str
    cut_version: str
    revision_marker: str | None = None
    piece_count: int = 0
    indexed_at: datetime | None = None
    last_seen_at: datetime | None = None

    def __getitem__(self, stage: str) -> str:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
        return {
            STAGE_READ: self.read_fingerprint,
            STAGE_CUT: self.cut_fingerprint,
            STAGE_EMBED: self.embed_fingerprint,
        }[stage]


@dataclass(frozen=True, slots=True)
class Reprocessing:
    action: str
    reason: str | None
    explanation: str
    changed_stages: tuple[str, ...] = ()

    @property
    def costs_anything(self) -> bool:
        return self.action != ACTION_TOUCH

    @property
    def reads_the_source(self) -> bool:
        return self.action == ACTION_READ

    @property
    def cuts(self) -> bool:
        return self.action in (ACTION_READ, ACTION_RECUT)

    @property
    def embeds(self) -> bool:
        return self.action in (ACTION_READ, ACTION_RECUT, ACTION_REEMBED)


def decide(
    *,
    stored: IngestState | None,
    content_digest: str,
    fingerprints: StageFingerprints,
    marker_moved: bool = False,
    operator_requested: bool = False,
) -> Reprocessing:
    if stored is None:
        return Reprocessing(
            action=ACTION_READ,
            reason=REASON_NEW_DOCUMENT,
            explanation="not indexed before, so there is nothing to reuse",
            changed_stages=STAGES,
        )
    if operator_requested:
        return Reprocessing(
            action=ACTION_READ,
            reason=REASON_OPERATOR_REQUESTED,
            explanation=(
                "an operator asked for this document to be reprocessed, and the point of "
                "asking is to distrust what is stored"
            ),
            changed_stages=STAGES,
        )
    if content_digest != stored.content_digest:
        return Reprocessing(
            action=ACTION_READ,
            reason=REASON_CONTENT_CHANGED,
            explanation="the exported bytes differ from the ones indexed",
            changed_stages=STAGES,
        )
    if fingerprints.whole == stored.fingerprint:
        return Reprocessing(
            action=ACTION_TOUCH,
            reason=REASON_WATERMARK_MOVED if marker_moved else None,
            explanation=(
                "the source's revision marker moved but the exported bytes are identical and "
                "the configuration did not change, so nothing is read, cut or embedded (AC-018)"
                if marker_moved
                else "the content and the indexing configuration both match what is stored"
            ),
        )
    changed = tuple(stage for stage in STAGES if fingerprints[stage] != stored[stage])
    for stage, action in _STAGE_ACTIONS:
        if stage in changed:
            return Reprocessing(
                action=action,
                reason=REASON_CONFIGURATION_CHANGED,
                explanation=_EXPLANATIONS[action],
                changed_stages=changed,
            )
    return Reprocessing(
        action=ACTION_READ,
        reason=REASON_CONFIGURATION_CHANGED,
        explanation=(
            "the indexing fingerprint moved but no single stage's did, so which step became "
            "stale cannot be told; the document is reprocessed from the source rather than "
            "guessed at cheaply"
        ),
        changed_stages=STAGES,
    )


_EXPLANATIONS: dict[str, str] = {
    ACTION_READ: (
        "the reader's configuration moved, so the kept output was produced by something "
        "this deployment no longer asks for and the document is read again"
    ),
    ACTION_RECUT: (
        "the bytes are identical and only the cutting configuration moved, so the pieces "
        "are cut again from the reader's kept output and nothing is read (AC-102)"
    ),
    ACTION_REEMBED: (
        "the bytes and the cut are identical and only the embedding configuration moved, so "
        "the stored pieces are embedded again and nothing is read or cut (AC-103)"
    ),
}


def stored_state(conn: psycopg.Connection, document_id: int) -> IngestState | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, content_digest, fingerprint, read_fingerprint, cut_fingerprint,
                   embed_fingerprint, reader_version, cut_version, revision_marker, piece_count,
                   indexed_at, last_seen_at
              FROM document_ingest
             WHERE document_id = %s
            """,
            (document_id,),
        )
        row = cursor.fetchone()
    return _state(row) if row is not None else None


def state_by_external_id(conn: psycopg.Connection, external_id: str) -> IngestState | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.document_id, i.content_digest, i.fingerprint, i.read_fingerprint,
                   i.cut_fingerprint, i.embed_fingerprint, i.reader_version, i.cut_version,
                   i.revision_marker, i.piece_count, i.indexed_at, i.last_seen_at
              FROM document_ingest i
              JOIN documents d ON d.id = i.document_id
             WHERE d.external_id = %s
            """,
            (external_id,),
        )
        row = cursor.fetchone()
    return _state(row) if row is not None else None


def record_state(
    conn: psycopg.Connection,
    *,
    document_id: int,
    content_digest: str,
    fingerprints: StageFingerprints,
    reader_version: str,
    cut_version: str,
    piece_count: int,
    revision_marker: str | None = None,
    process_module: str = "",
) -> IngestState:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO document_ingest (
                document_id, content_digest, fingerprint, read_fingerprint, cut_fingerprint,
                embed_fingerprint, reader_version, cut_version, revision_marker, piece_count,
                process_module
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE
                    SET content_digest    = EXCLUDED.content_digest,
                        fingerprint       = EXCLUDED.fingerprint,
                        read_fingerprint  = EXCLUDED.read_fingerprint,
                        cut_fingerprint   = EXCLUDED.cut_fingerprint,
                        embed_fingerprint = EXCLUDED.embed_fingerprint,
                        reader_version    = EXCLUDED.reader_version,
                        cut_version       = EXCLUDED.cut_version,
                        revision_marker   = EXCLUDED.revision_marker,
                        piece_count       = EXCLUDED.piece_count,
                        process_module    = EXCLUDED.process_module,
                        indexed_at        = now(),
                        last_seen_at      = clock_timestamp()
              RETURNING document_id, content_digest, fingerprint, read_fingerprint,
                        cut_fingerprint, embed_fingerprint, reader_version, cut_version,
                        revision_marker, piece_count, indexed_at, last_seen_at
            """,
            (
                document_id,
                content_digest,
                fingerprints.whole,
                fingerprints.read,
                fingerprints.cut,
                fingerprints.embed,
                reader_version,
                cut_version,
                revision_marker,
                piece_count,
                process_module or None,
            ),
        )
        return _state(cursor.fetchone())


def touch(
    conn: psycopg.Connection,
    *,
    document_id: int,
    revision_marker: str | None = None,
) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE document_ingest
               SET last_seen_at    = clock_timestamp(),
                   revision_marker = COALESCE(%s, revision_marker)
             WHERE document_id = %s
            """,
            (revision_marker, document_id),
        )
        return cursor.rowcount == 1


def documents_at(conn: psycopg.Connection, fingerprint: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM document_ingest WHERE fingerprint = %s", (fingerprint,)
        )
        (count,) = cursor.fetchone()
    return int(count)


def documents_not_at(conn: psycopg.Connection, fingerprint: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM document_ingest WHERE fingerprint <> %s", (fingerprint,)
        )
        (count,) = cursor.fetchone()
    return int(count)


def stale_states(conn: psycopg.Connection, fingerprint: str) -> tuple[IngestState, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, content_digest, fingerprint, read_fingerprint, cut_fingerprint,
                   embed_fingerprint, reader_version, cut_version, revision_marker, piece_count,
                   indexed_at, last_seen_at
              FROM document_ingest
             WHERE fingerprint <> %s
             ORDER BY indexed_at, document_id
            """,
            (fingerprint,),
        )
        return tuple(_state(row) for row in cursor.fetchall())


def recorded_fingerprints(conn: psycopg.Connection) -> tuple[str, ...]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT fingerprint FROM document_ingest ORDER BY fingerprint")
        return tuple(row[0] for row in cursor.fetchall())


def _state(row: tuple) -> IngestState:
    return IngestState(
        document_id=int(row[0]),
        content_digest=row[1],
        fingerprint=row[2],
        read_fingerprint=row[3],
        cut_fingerprint=row[4],
        embed_fingerprint=row[5],
        reader_version=row[6],
        cut_version=row[7],
        revision_marker=row[8],
        piece_count=int(row[9]),
        indexed_at=row[10],
        last_seen_at=row[11],
    )


@dataclass(frozen=True, slots=True)
class Retirement:
    considered: int = 0
    retired: tuple[str, ...] = ()
    spared: tuple[str, ...] = ()
    refusal: str = ""
    detail: str = ""

    @property
    def clean(self) -> bool:
        return not self.refusal


def foreign_documents(conn: psycopg.Connection, module: str) -> tuple[tuple[int, str], ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT i.document_id, d.external_id
              FROM document_ingest i
              JOIN documents d ON d.id = i.document_id
             WHERE i.process_module IS DISTINCT FROM %s
             ORDER BY d.external_id
            """,
            (module,),
        )
        return tuple((int(row[0]), row[1]) for row in cursor.fetchall())


def unfinished_documents(conn: psycopg.Connection) -> frozenset[int]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT document_id
              FROM ingest_claims
             WHERE document_id IS NOT NULL
               AND state = %s
               AND failure_kind = %s
            """,
            (CLAIM_STATE_FAILED, FAILURE_TRANSIENT),
        )
        return frozenset(int(row[0]) for row in cursor.fetchall())


def retire_foreign_pieces(
    conn: psycopg.Connection,
    *,
    module: str,
    may_remove: bool,
) -> Retirement:
    found = foreign_documents(conn, module)
    unfinished = unfinished_documents(conn)
    spared = tuple(external for document_id, external in found if document_id in unfinished)
    foreign = [(document_id, external) for document_id, external in found
               if document_id not in unfinished]
    if not found:
        return Retirement(detail=f"every indexed document was produced by {module!r}")
    if not may_remove:
        return Retirement(
            considered=len(found),
            spared=spared,
            refusal="sweep_did_not_complete",
            detail=(
                f"{len(found)} document(s) still carry another module's pieces and this run "
                "is not a completed sweep, so it never finished looking (principle P-003); "
                "they stay indexed and the next completed sweep decides"
            ),
        )
    if not foreign:
        return Retirement(
            considered=len(found),
            spared=spared,
            detail=(
                f"all {len(spared)} document(s) still carrying another module's pieces failed "
                "in a way the next run retries, so the switch is not finished for them and "
                "nothing is removed"
            ),
        )
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM document_ingest WHERE process_module = %s", (module,)
        )
        produced = int(cursor.fetchone()[0])
    if produced == 0:
        return Retirement(
            considered=len(found),
            spared=spared,
            refusal="module_produced_nothing",
            detail=(
                f"not one document was indexed by {module!r}, so retiring the "
                f"{len(foreign)} the previous module produced would empty the archive "
                "without emptying a table — every search returning nothing while every row "
                "is still there. Nothing is removed; the reason each document was refused "
                "is on document_reads"
            ),
        )
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunks WHERE document_id = ANY(%s)",
                ([document_id for document_id, _external in foreign],),
            )
            cursor.execute(
                "DELETE FROM document_ingest WHERE document_id = ANY(%s)",
                ([document_id for document_id, _external in foreign],),
            )
    kept = (
        f"; {len(spared)} more failed in a way the next run retries and keep what they have"
        if spared
        else ""
    )
    return Retirement(
        considered=len(found),
        retired=tuple(external for _document_id, external in foreign),
        spared=spared,
        detail=(
            f"{len(foreign)} document(s) carried pieces made by a module this deployment no "
            f"longer runs; {produced} were produced by {module!r} and are what the archive "
            f"now holds{kept}"
        ),
    )
