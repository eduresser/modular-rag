from src.sources.identity import Action, StoredIdentity, after_fetch, before_fetch
from src.usage import (
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_OPERATOR_REQUESTED,
    REASON_WATERMARK_MOVED,
)

FINGERPRINT = "fp-1"
OTHER_FINGERPRINT = "fp-2"


def _stored(marker: str | None = "7", digest: str = "abc", fingerprint: str = FINGERPRINT):
    return StoredIdentity(
        external_id="doc-1",
        content_digest=digest,
        indexing_fingerprint=fingerprint,
        revision_marker=marker,
    )


def test_a_document_never_seen_is_indexed_without_asking_anything_else():
    decision = before_fetch(stored=None, revision_marker="1", fingerprint=FINGERPRINT)

    assert decision.action is Action.INDEX_FROM_SOURCE
    assert decision.reason == REASON_NEW_DOCUMENT
    assert decision.costs_a_read is True


def test_an_unmoved_marker_and_unchanged_configuration_cost_nothing():
    decision = before_fetch(stored=_stored(), revision_marker="7", fingerprint=FINGERPRINT)

    assert decision.action is Action.TOUCH_ONLY
    assert decision.costs_anything is False
    assert decision.reason is None


def test_a_changed_configuration_re_cuts_without_downloading_anything():
    decision = before_fetch(stored=_stored(), revision_marker="7", fingerprint=OTHER_FINGERPRINT)

    assert decision.action is Action.RECUT_FROM_STORED
    assert decision.reason == REASON_CONFIGURATION_CHANGED
    assert decision.costs_a_read is False
    assert decision.costs_anything is True


def test_a_moved_marker_cannot_be_settled_without_the_bytes():
    assert before_fetch(stored=_stored(), revision_marker="8", fingerprint=FINGERPRINT) is None


def test_a_source_that_offers_no_marker_is_never_assumed_unchanged():
    assert before_fetch(stored=_stored(marker="7"), revision_marker=None, fingerprint=FINGERPRINT) is None
    assert before_fetch(stored=_stored(marker=None), revision_marker=None, fingerprint=FINGERPRINT) is None


def test_an_operator_request_always_pays_for_the_download():
    assert (
        before_fetch(
            stored=_stored(), revision_marker="7", fingerprint=FINGERPRINT, operator_requested=True
        )
        is None
    )


def test_identical_bytes_after_a_moved_marker_are_touched_and_attributed():
    decision = after_fetch(stored=_stored(), content_digest="abc", fingerprint=FINGERPRINT)

    assert decision.action is Action.TOUCH_ONLY
    assert decision.costs_anything is False
    assert decision.reason == REASON_WATERMARK_MOVED
    assert "AC-018" in decision.explanation


def test_different_bytes_are_a_real_edit():
    decision = after_fetch(stored=_stored(), content_digest="different", fingerprint=FINGERPRINT)

    assert decision.action is Action.INDEX_FROM_SOURCE
    assert decision.reason == REASON_CONTENT_CHANGED


def test_identical_bytes_with_a_moved_configuration_re_cut_rather_than_re_read():
    decision = after_fetch(stored=_stored(), content_digest="abc", fingerprint=OTHER_FINGERPRINT)

    assert decision.action is Action.RECUT_FROM_STORED
    assert decision.reason == REASON_CONFIGURATION_CHANGED


def test_content_wins_over_configuration_when_both_moved():
    decision = after_fetch(stored=_stored(), content_digest="different", fingerprint=OTHER_FINGERPRINT)

    assert decision.action is Action.INDEX_FROM_SOURCE
    assert decision.reason == REASON_CONTENT_CHANGED


def test_an_operator_request_reindexes_even_when_nothing_moved():
    decision = after_fetch(
        stored=_stored(), content_digest="abc", fingerprint=FINGERPRINT, operator_requested=True
    )

    assert decision.action is Action.INDEX_FROM_SOURCE
    assert decision.reason == REASON_OPERATOR_REQUESTED


def test_every_decision_explains_itself_in_words_an_operator_can_act_on():
    decisions = [
        before_fetch(stored=None, revision_marker="1", fingerprint=FINGERPRINT),
        before_fetch(stored=_stored(), revision_marker="7", fingerprint=FINGERPRINT),
        before_fetch(stored=_stored(), revision_marker="7", fingerprint=OTHER_FINGERPRINT),
        after_fetch(stored=_stored(), content_digest="abc", fingerprint=FINGERPRINT),
        after_fetch(stored=_stored(), content_digest="x", fingerprint=FINGERPRINT),
    ]

    for decision in decisions:
        assert decision.explanation.strip(), decision
        assert not decision.explanation.endswith("."), "these are clause fragments, not sentences"
