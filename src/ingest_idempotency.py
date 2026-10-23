from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

from migrations import load as load_migration

_migration = load_migration("006_ingest")

CLAIM_STATE_CLAIMED: str = _migration.CLAIM_STATE_CLAIMED
CLAIM_STATE_DONE: str = _migration.CLAIM_STATE_DONE
CLAIM_STATE_FAILED: str = _migration.CLAIM_STATE_FAILED
CLAIM_STATES: tuple[str, ...] = _migration.CLAIM_STATES

ACTIONS: tuple[str, ...] = _migration.ACTIONS

FAILURE_PERMANENT: str = _migration.FAILURE_PERMANENT
FAILURE_TRANSIENT: str = _migration.FAILURE_TRANSIENT
FAILURE_KINDS: tuple[str, ...] = _migration.FAILURE_KINDS

APP_VERSION = "0.1.0"


class ClaimError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Claim:
    id: int
    external_id: str
    content_digest: str
    fingerprint: str
    document_id: int | None
    run_id: int | None
    app_version: str
    state: str
    action: str | None
    deliveries: int
    failure_kind: str | None
    failure_reason: str | None
    claimed_at: datetime
    settled_at: datetime | None

    @property
    def settled(self) -> bool:
        return self.state != CLAIM_STATE_CLAIMED

    @property
    def retryable(self) -> bool:
        return self.state != CLAIM_STATE_FAILED or self.failure_kind == FAILURE_TRANSIENT


@dataclass(frozen=True, slots=True)
class ClaimOutcome:
    claim: Claim
    fresh: bool

    @property
    def duplicate(self) -> bool:
        return not self.fresh and self.claim.state == CLAIM_STATE_DONE

    @property
    def resumed(self) -> bool:
        return not self.fresh and self.claim.state == CLAIM_STATE_CLAIMED

    @property
    def blocked(self) -> bool:
        return not self.fresh and self.claim.state == CLAIM_STATE_FAILED and not self.claim.retryable

    @property
    def should_run(self) -> bool:
        return self.fresh or self.resumed or (self.claim.state == CLAIM_STATE_FAILED and self.claim.retryable)


_COLUMNS = """
    id, external_id, content_digest, fingerprint, document_id, run_id, app_version,
    state, action, deliveries, failure_kind, failure_reason, claimed_at, settled_at
"""


def claim(
    conn: psycopg.Connection,
    *,
    external_id: str,
    content_digest: str,
    fingerprint: str,
    document_id: int | None = None,
    run_id: int | None = None,
    app_version: str = APP_VERSION,
) -> ClaimOutcome:
    if not external_id.strip() or not content_digest.strip() or not fingerprint.strip():
        raise ClaimError(
            "a unit of work is the document's identifier, the digest of its bytes and the "
            "indexing fingerprint together; none of the three may be blank, or two "
            "different pieces of work would share one claim"
        )
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO ingest_claims (
                external_id, content_digest, fingerprint, document_id, run_id, app_version
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id, content_digest, fingerprint) DO UPDATE
                    SET deliveries  = ingest_claims.deliveries + 1,
                        document_id = COALESCE(EXCLUDED.document_id, ingest_claims.document_id),
                        run_id      = COALESCE(EXCLUDED.run_id, ingest_claims.run_id)
              RETURNING {_COLUMNS}, (xmax = 0) AS inserted
            """,
            (external_id, content_digest, fingerprint, document_id, run_id, app_version),
        )
        row = cursor.fetchone()
    return ClaimOutcome(claim=_claim(row), fresh=bool(row[-1]))


def settle(
    conn: psycopg.Connection,
    claim_id: int,
    *,
    action: str,
    document_id: int | None = None,
) -> Claim:
    if action not in ACTIONS:
        raise ClaimError(f"action {action!r} is not one of {', '.join(ACTIONS)}")
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE ingest_claims
               SET state          = %s,
                   action         = %s,
                   document_id    = COALESCE(%s, document_id),
                   failure_kind   = NULL,
                   failure_reason = NULL,
                   settled_at     = now()
             WHERE id = %s
         RETURNING {_COLUMNS}
            """,
            (CLAIM_STATE_DONE, action, document_id, claim_id),
        )
        row = cursor.fetchone()
    if row is None:
        raise ClaimError(f"no claim {claim_id} to settle")
    return _claim(row)


def fail(
    conn: psycopg.Connection,
    claim_id: int,
    *,
    reason: str,
    kind: str,
    document_id: int | None = None,
) -> Claim:
    if kind not in FAILURE_KINDS:
        raise ClaimError(
            f"failure kind {kind!r} is not one of {', '.join(FAILURE_KINDS)}; a failure that "
            "does not say whether trying again could work turns a retry budget into a bill"
        )
    if not reason.strip():
        raise ClaimError(
            f"claim {claim_id} was failed with no reason; a failure nobody can act on is the "
            "one AC-106 exists to prevent"
        )
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE ingest_claims
               SET state          = %s,
                   failure_kind   = %s,
                   failure_reason = %s,
                   document_id    = COALESCE(%s, document_id),
                   settled_at     = now()
             WHERE id = %s
         RETURNING {_COLUMNS}
            """,
            (CLAIM_STATE_FAILED, kind, reason, document_id, claim_id),
        )
        row = cursor.fetchone()
    if row is None:
        raise ClaimError(f"no claim {claim_id} to fail")
    return _claim(row)


def read_claim(
    conn: psycopg.Connection,
    *,
    external_id: str,
    content_digest: str,
    fingerprint: str,
) -> Claim | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_COLUMNS}
              FROM ingest_claims
             WHERE external_id = %s AND content_digest = %s AND fingerprint = %s
            """,
            (external_id, content_digest, fingerprint),
        )
        row = cursor.fetchone()
    return _claim(row) if row is not None else None


def in_flight(
    conn: psycopg.Connection,
    *,
    app_version: str = APP_VERSION,
    run_id: int | None = None,
) -> tuple[Claim, ...]:
    query = f"""
        SELECT {_COLUMNS}
          FROM ingest_claims
         WHERE state = %(state)s AND app_version = %(app_version)s
    """
    params: dict = {"state": CLAIM_STATE_CLAIMED, "app_version": app_version}
    if run_id is not None:
        query += " AND run_id = %(run_id)s"
        params["run_id"] = run_id
    query += " ORDER BY claimed_at, id"
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        return tuple(_claim(row) for row in cursor.fetchall())


def claims_for(conn: psycopg.Connection, external_id: str) -> tuple[Claim, ...]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {_COLUMNS} FROM ingest_claims WHERE external_id = %s ORDER BY id",
            (external_id,),
        )
        return tuple(_claim(row) for row in cursor.fetchall())


def _claim(row: tuple) -> Claim:
    return Claim(
        id=int(row[0]),
        external_id=row[1],
        content_digest=row[2],
        fingerprint=row[3],
        document_id=int(row[4]) if row[4] is not None else None,
        run_id=int(row[5]) if row[5] is not None else None,
        app_version=row[6],
        state=row[7],
        action=row[8],
        deliveries=int(row[9]),
        failure_kind=row[10],
        failure_reason=row[11],
        claimed_at=row[12],
        settled_at=row[13],
    )
