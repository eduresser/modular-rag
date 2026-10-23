"""Unit-level coverage for src/ingest_idempotency.py.

AC-105 is proven in tests/test_spec_ingest_pipeline.py, against the whole
pipeline. This file covers the ledger underneath it: what the unit of work is,
what the constraint refuses, and what a claim says about whether another
attempt could ever work.

The archive is real PostgreSQL because the guarantee under test *is* the unique
constraint. A stand-in that checked in Python would prove the check, not the
property — and the property is that two workers on two machines cannot both
win, which no application-level check has.
"""

import psycopg
import pytest

from src.ingest_idempotency import (
    APP_VERSION,
    CLAIM_STATE_CLAIMED,
    CLAIM_STATE_DONE,
    CLAIM_STATE_FAILED,
    FAILURE_PERMANENT,
    FAILURE_TRANSIENT,
    ClaimError,
    claim,
    claims_for,
    fail,
    in_flight,
    read_claim,
    settle,
)
from src.reprocess import ACTION_READ, ACTION_RECUT
from src.store import upsert_document

UNIT = dict(external_id="doc-1", content_digest="digest-1", fingerprint="fingerprint-1")


def test_the_first_delivery_opens_the_claim(conn):
    outcome = claim(conn, **UNIT)

    assert outcome.fresh is True
    assert outcome.should_run is True
    assert outcome.duplicate is False
    assert outcome.claim.state == CLAIM_STATE_CLAIMED
    assert outcome.claim.deliveries == 1
    assert outcome.claim.app_version == APP_VERSION
    assert outcome.claim.settled_at is None


def test_a_second_delivery_of_settled_work_is_a_duplicate_and_is_counted(conn):
    first = claim(conn, **UNIT)
    settle(conn, first.claim.id, action=ACTION_READ)

    again = claim(conn, **UNIT)

    assert again.fresh is False
    assert again.duplicate is True
    assert again.should_run is False
    assert again.claim.deliveries == 2
    assert again.claim.action == ACTION_READ
    # One row, not two: the unit of work is the triple, and the constraint is
    # what makes that true rather than a check somebody wrote.
    assert len(claims_for(conn, "doc-1")) == 1


def test_the_same_bytes_under_a_new_configuration_are_new_work(conn):
    # ASM-051 named the identifier and the digest. The fingerprint has to join
    # them, or the configuration change that AC-102 and AC-103 require to be
    # *done* would be refused as a duplicate and nothing would be re-indexed.
    first = claim(conn, **UNIT)
    settle(conn, first.claim.id, action=ACTION_READ)

    reconfigured = claim(conn, external_id="doc-1", content_digest="digest-1", fingerprint="fingerprint-2")

    assert reconfigured.fresh is True
    assert reconfigured.should_run is True
    assert len(claims_for(conn, "doc-1")) == 2


def test_the_constraint_and_not_a_check_is_what_refuses_the_second_insert(conn):
    claim(conn, **UNIT)

    with conn.cursor() as cursor, pytest.raises(psycopg.errors.UniqueViolation):
        cursor.execute(
            """
            INSERT INTO ingest_claims (external_id, content_digest, fingerprint, app_version)
                 VALUES (%s, %s, %s, %s)
            """,
            (UNIT["external_id"], UNIT["content_digest"], UNIT["fingerprint"], APP_VERSION),
        )


def test_a_delivery_of_work_still_in_flight_is_work_to_resume(conn):
    claim(conn, **UNIT)

    again = claim(conn, **UNIT)

    assert again.resumed is True
    assert again.should_run is True
    assert again.duplicate is False


def test_a_claim_carries_the_document_and_the_run_once_they_are_known(conn):
    document_id = upsert_document(conn, external_id="doc-1")
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO sync_runs (client, kind) VALUES ('example', 'sweep') RETURNING id"
        )
        (run_id,) = cursor.fetchone()

    first = claim(conn, **UNIT)
    again = claim(conn, **UNIT, document_id=document_id, run_id=run_id)

    assert first.claim.document_id is None
    assert again.claim.document_id == document_id
    assert again.claim.run_id == run_id


def test_settling_names_the_route_that_ran(conn):
    outcome = claim(conn, **UNIT)

    settled = settle(conn, outcome.claim.id, action=ACTION_RECUT)

    assert settled.state == CLAIM_STATE_DONE
    assert settled.action == ACTION_RECUT
    assert settled.settled_at is not None
    assert read_claim(conn, **UNIT) == settled


def test_settling_with_an_action_nobody_recognises_is_refused(conn):
    outcome = claim(conn, **UNIT)

    with pytest.raises(ClaimError) as excinfo:
        settle(conn, outcome.claim.id, action="reticulate")

    assert "reticulate" in str(excinfo.value)


def test_a_transient_failure_is_retried_and_a_permanent_one_is_not(conn):
    # The answer to Q-051, and it is neither of the two the question offered:
    # retrying everything hides a systematic problem behind a recurring cost,
    # retrying nothing leaves a gap nobody notices.
    transient = claim(conn, **UNIT)
    fail(conn, transient.claim.id, reason="the provider timed out", kind=FAILURE_TRANSIENT)

    retried = claim(conn, **UNIT)
    assert retried.claim.retryable is True
    assert retried.should_run is True
    assert retried.blocked is False

    permanent_unit = dict(UNIT, content_digest="digest-2")
    permanent = claim(conn, **permanent_unit)
    fail(
        conn, permanent.claim.id,
        reason="is text/plain, and the reader does not accept text files",
        kind=FAILURE_PERMANENT,
    )

    again = claim(conn, **permanent_unit)
    assert again.claim.state == CLAIM_STATE_FAILED
    assert again.claim.retryable is False
    assert again.blocked is True
    assert again.should_run is False


def test_a_failure_has_to_say_what_broke_and_whether_a_retry_could_help(conn):
    outcome = claim(conn, **UNIT)

    with pytest.raises(ClaimError) as excinfo:
        fail(conn, outcome.claim.id, reason="", kind=FAILURE_TRANSIENT)
    assert "no reason" in str(excinfo.value)

    with pytest.raises(ClaimError) as excinfo:
        fail(conn, outcome.claim.id, reason="something went wrong", kind="maybe")
    assert "maybe" in str(excinfo.value)


def test_settling_a_claim_that_failed_clears_the_failure(conn):
    outcome = claim(conn, **UNIT)
    fail(conn, outcome.claim.id, reason="the provider timed out", kind=FAILURE_TRANSIENT)

    settled = settle(conn, outcome.claim.id, action=ACTION_READ)

    assert (settled.state, settled.failure_kind, settled.failure_reason) == (CLAIM_STATE_DONE, None, None)


def test_a_blank_part_of_the_key_is_refused_before_it_reaches_the_database(conn):
    with pytest.raises(ClaimError) as excinfo:
        claim(conn, external_id="doc-1", content_digest="  ", fingerprint="fingerprint-1")

    assert "none of the three may be blank" in str(excinfo.value)


def test_settling_or_failing_a_claim_nobody_made_says_so(conn):
    with pytest.raises(ClaimError):
        settle(conn, 999_999, action=ACTION_READ)
    with pytest.raises(ClaimError):
        fail(conn, 999_999, reason="gone", kind=FAILURE_TRANSIENT)


def test_in_flight_is_scoped_to_the_build_that_opened_the_claim(conn):
    mine = claim(conn, **UNIT, app_version="0.1.0")
    claim(conn, external_id="doc-2", content_digest="d", fingerprint="f", app_version="0.2.0")
    settled = claim(conn, external_id="doc-3", content_digest="d", fingerprint="f", app_version="0.1.0")
    settle(conn, settled.claim.id, action=ACTION_READ)

    # A rolling deploy has two builds running at once; two workers finishing
    # each other's in-flight units is how a redeploy turns into a double bill.
    assert [c.id for c in in_flight(conn, app_version="0.1.0")] == [mine.claim.id]
    assert [c.external_id for c in in_flight(conn, app_version="0.2.0")] == ["doc-2"]


def test_in_flight_can_be_narrowed_to_one_run(conn):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO sync_runs (client, kind) VALUES ('example', 'sweep') RETURNING id")
        (run_id,) = cursor.fetchone()
    claim(conn, **UNIT, run_id=run_id)
    claim(conn, external_id="doc-2", content_digest="d", fingerprint="f")

    assert [c.external_id for c in in_flight(conn, run_id=run_id)] == ["doc-1"]


def test_a_claim_nobody_made_reads_as_nothing(conn):
    assert read_claim(conn, **UNIT) is None
    assert claims_for(conn, "doc-1") == ()


# ── A claim is work attempted, not state reached ──────────────────────────


def _cut_fingerprint(conn: psycopg.Connection, external_id: str) -> str:
    row = conn.execute(
        "SELECT i.cut_fingerprint FROM document_ingest i JOIN documents d ON d.id = i.document_id "
        "WHERE d.external_id = %s",
        (external_id,),
    ).fetchone()
    return "" if row is None else row[0]


def test_going_back_to_a_configuration_reprocesses_instead_of_refusing(monkeypatch, conn):
    """A settled claim stops being evidence once the archive has moved past it.

    Found by reverting a real deployment. The unit of work is the document, its
    digest and the fingerprint together, so a *new* configuration is new work —
    that much ASM-051 got right. What it did not cover is going **back**: the
    claim from the first time is still settled ``done``, so every document is
    refused as a duplicate while ``document_ingest`` still holds the cut it was
    moved to. Nothing converges, and nothing ever would: each later run finds
    the same settled claim and makes the same refusal.

    All fourteen documents of the real corpus were skipped that way, and the
    only way out was deleting claim rows by hand.
    """
    from src.ingest import ingest_documents
    from tests.conftest import contract_payload, ingest_world, pdf_file

    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=("Cláusula primeira.", "Cláusula segunda."))}

    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    ingest_documents(conn, first.pipeline, list(first.pipeline.source.sweep()))
    original = _cut_fingerprint(conn, "doc-1")

    # A different cut, over the artifact the first run kept.
    moved = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage, RAG_CHUNK_SIZE="400"
    )
    ingest_documents(conn, moved.pipeline, list(moved.pipeline.source.sweep()))
    assert _cut_fingerprint(conn, "doc-1") != original

    # And back again — the configuration the first run used, whose claim is
    # still settled from the first run.
    back = ingest_world(monkeypatch, files=files, outputs=outputs, storage=first.storage)
    run = ingest_documents(conn, back.pipeline, list(back.pipeline.source.sweep()))

    assert run.duplicates == ()
    assert _cut_fingerprint(conn, "doc-1") == original


def test_a_redelivery_of_work_the_archive_still_holds_is_still_a_duplicate(monkeypatch, conn):
    """The other side, so the fix above does not simply turn AC-105 off.

    A claim whose fingerprint is the one the archive is on is evidence, and a
    second delivery of it is charged nothing.
    """
    from src.ingest import ingest_documents
    from tests.conftest import contract_payload, ingest_world, pdf_file

    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=("Cláusula primeira.",))},
    )
    document = next(iter(world.pipeline.source.sweep()))

    run = ingest_documents(conn, world.pipeline, [document, document])

    assert len(run.duplicates) == 1
    assert len(run.indexed) == 1
