import pytest

from src.fingerprint import STAGE_CUT, STAGE_EMBED, STAGE_READ, StageFingerprints
from src.reprocess import (
    ACTION_READ,
    ACTION_RECUT,
    ACTION_REEMBED,
    ACTION_TOUCH,
    IngestState,
    decide,
    documents_at,
    documents_not_at,
    record_state,
    recorded_fingerprints,
    stale_states,
    state_by_external_id,
    stored_state,
    touch,
)
from src.store import upsert_document
from src.usage import (
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_OPERATOR_REQUESTED,
    REASON_WATERMARK_MOVED,
)

DIGEST = "digest-of-the-bytes"


def _fingerprints(read="r1", cut="c1", embed="e1", whole="w1") -> StageFingerprints:
    return StageFingerprints(read=read, cut=cut, embed=embed, whole=whole)


def _stored(**overrides) -> IngestState:
    base = dict(
        document_id=1,
        content_digest=DIGEST,
        fingerprint="w1",
        read_fingerprint="r1",
        cut_fingerprint="c1",
        embed_fingerprint="e1",
        reader_version="pretrained-layout-parser-v1.0-2024-06-03",
        cut_version="1",
        revision_marker="marker-1",
        piece_count=3,
    )
    base.update(overrides)
    return IngestState(**base)


def test_a_document_nobody_has_seen_is_read_from_the_source():
    decision = decide(stored=None, content_digest=DIGEST, fingerprints=_fingerprints())

    assert decision.action == ACTION_READ
    assert decision.reason == REASON_NEW_DOCUMENT
    assert decision.reads_the_source


def test_matching_content_and_configuration_costs_nothing():
    decision = decide(stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints())

    assert decision.action == ACTION_TOUCH
    assert decision.reason is None
    assert not decision.costs_anything
    assert not decision.embeds


def test_a_marker_that_moved_over_identical_bytes_is_attributed_to_the_watermark():
    decision = decide(
        stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints(), marker_moved=True
    )

    assert decision.action == ACTION_TOUCH
    assert decision.reason == REASON_WATERMARK_MOVED


def test_changed_bytes_are_read_again_whatever_else_matches():
    decision = decide(stored=_stored(), content_digest="something-else", fingerprints=_fingerprints())

    assert decision.action == ACTION_READ
    assert decision.reason == REASON_CONTENT_CHANGED


def test_a_changed_reader_configuration_reads_again():
    decision = decide(
        stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints(read="r2", whole="w2")
    )

    assert decision.action == ACTION_READ
    assert decision.reason == REASON_CONFIGURATION_CHANGED
    assert decision.changed_stages == (STAGE_READ,)


def test_a_changed_cut_configuration_recuts_from_the_kept_artifact():
    decision = decide(
        stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints(cut="c2", whole="w2")
    )

    assert decision.action == ACTION_RECUT
    assert decision.reason == REASON_CONFIGURATION_CHANGED
    assert decision.changed_stages == (STAGE_CUT,)
    assert not decision.reads_the_source
    assert decision.cuts and decision.embeds


def test_a_changed_embedding_configuration_only_embeds_again():
    decision = decide(
        stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints(embed="e2", whole="w2")
    )

    assert decision.action == ACTION_REEMBED
    assert decision.changed_stages == (STAGE_EMBED,)
    assert not decision.cuts
    assert decision.embeds


def test_the_dearest_changed_stage_wins():
    decision = decide(
        stored=_stored(),
        content_digest=DIGEST,
        fingerprints=_fingerprints(read="r2", cut="c2", embed="e2", whole="w2"),
    )

    assert decision.action == ACTION_READ
    assert decision.changed_stages == (STAGE_READ, STAGE_CUT, STAGE_EMBED)


def test_a_cut_and_an_embed_together_still_only_cut_and_embed():
    decision = decide(
        stored=_stored(),
        content_digest=DIGEST,
        fingerprints=_fingerprints(cut="c2", embed="e2", whole="w2"),
    )

    assert decision.action == ACTION_RECUT
    assert not decision.reads_the_source


def test_an_operator_request_distrusts_everything_that_is_stored():
    decision = decide(
        stored=_stored(),
        content_digest=DIGEST,
        fingerprints=_fingerprints(),
        operator_requested=True,
    )

    assert decision.action == ACTION_READ
    assert decision.reason == REASON_OPERATOR_REQUESTED


def test_a_fingerprint_that_moved_with_no_stage_moving_is_read_rather_than_guessed():
    decision = decide(stored=_stored(), content_digest=DIGEST, fingerprints=_fingerprints(whole="w2"))

    assert decision.action == ACTION_READ
    assert "cannot be told" in decision.explanation


def test_a_stage_the_state_does_not_know_is_refused_by_name():
    with pytest.raises(KeyError) as excinfo:
        _stored()["rerank"]

    assert "rerank" in str(excinfo.value)


def test_the_state_round_trips_by_row_id_and_by_external_id(conn):
    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")

    written = record_state(
        conn,
        document_id=document_id,
        content_digest=DIGEST,
        fingerprints=_fingerprints(),
        reader_version="pretrained-layout-parser-v1.0-2024-06-03",
        cut_version="1",
        piece_count=4,
        revision_marker="marker-1",
    )

    assert written.piece_count == 4
    assert stored_state(conn, document_id) == written
    assert state_by_external_id(conn, "doc-1") == written
    assert state_by_external_id(conn, "doc-404") is None


def test_recording_the_state_again_replaces_it_rather_than_adding_a_row(conn):
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn, document_id=document_id, content_digest=DIGEST, fingerprints=_fingerprints(),
        reader_version="v1", cut_version="1", piece_count=4,
    )

    record_state(
        conn, document_id=document_id, content_digest="new-digest",
        fingerprints=_fingerprints(whole="w2"), reader_version="v1", cut_version="2",
        piece_count=7,
    )

    state = stored_state(conn, document_id)
    assert (state.content_digest, state.fingerprint, state.cut_version, state.piece_count) == (
        "new-digest", "w2", "2", 7,
    )
    assert recorded_fingerprints(conn) == ("w2",)


def test_touching_moves_only_the_last_seen_time(conn):
    document_id = upsert_document(conn, external_id="doc-1")
    before = record_state(
        conn, document_id=document_id, content_digest=DIGEST, fingerprints=_fingerprints(),
        reader_version="v1", cut_version="1", piece_count=4, revision_marker="marker-1",
    )

    assert touch(conn, document_id=document_id, revision_marker="marker-2") is True

    after = stored_state(conn, document_id)
    assert after.last_seen_at > before.last_seen_at
    assert after.indexed_at == before.indexed_at
    assert after.piece_count == before.piece_count
    assert after.revision_marker == "marker-2"


def test_touching_a_document_nobody_has_indexed_says_so_rather_than_inventing_a_row(conn):
    document_id = upsert_document(conn, external_id="doc-1")

    assert touch(conn, document_id=document_id) is False
    assert stored_state(conn, document_id) is None


def test_touching_without_a_marker_keeps_the_one_that_is_there(conn):
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn, document_id=document_id, content_digest=DIGEST, fingerprints=_fingerprints(),
        reader_version="v1", cut_version="1", piece_count=1, revision_marker="marker-1",
    )

    touch(conn, document_id=document_id)

    assert stored_state(conn, document_id).revision_marker == "marker-1"


def test_counting_and_listing_the_documents_behind_each_fingerprint(conn):
    for external_id, whole in (("doc-1", "w1"), ("doc-2", "w1"), ("doc-3", "w2")):
        document_id = upsert_document(conn, external_id=external_id)
        record_state(
            conn, document_id=document_id, content_digest=DIGEST,
            fingerprints=_fingerprints(whole=whole), reader_version="v1", cut_version="1",
            piece_count=1,
        )

    assert documents_at(conn, "w1") == 2
    assert documents_not_at(conn, "w1") == 1
    assert recorded_fingerprints(conn) == ("w1", "w2")
    assert [state.fingerprint for state in stale_states(conn, "w2")] == ["w1", "w1"]
