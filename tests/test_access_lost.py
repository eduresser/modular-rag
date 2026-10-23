"""Unit-level coverage for src/sync_access.py.

AC-012 and principle P-005 are proven in tests/test_spec_source_gdrive.py —
see CLAUDE.md for why those proofs live there. This file covers the summary an
operator actually reads, the ordering that makes two identical runs produce
identical records, and how a plain "not found" is resolved once the activity
history is consulted.
"""

from src.sources.base import DocumentUnavailable, SourceDocument
from src.sources.gdrive_activity import DeletionEvent, DeletionKind, MoveEvent, PermissionChangeEvent
from src.sources.gdrive_export import KIND_DOCUMENT
from src.sync_access import ProbeSummary, Verdict, probe_removal, probe_removals


class _Source:
    """A source whose answers a test states outright."""

    def __init__(self, **answers) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def probe(self, external_id: str) -> SourceDocument:
        self.asked.append(external_id)
        answer = self.answers.get(external_id, "missing")
        if answer == "denied":
            raise DocumentUnavailable(external_id, "forbidden", denied=True)
        if answer == "trashed":
            # The fact travels in the field, as the real client sets it. When
            # this was a phrase to be searched for, a fake that reworded the
            # message stopped proving anything without failing.
            raise DocumentUnavailable(
                external_id,
                "the source still answers for it, but marks it trashed — proof of "
                "deletion, not evidence of a permission change",
                denied=False,
                trashed=True,
            )
        if answer == "missing":
            raise DocumentUnavailable(external_id, "notFound")
        return SourceDocument(external_id=external_id, name=external_id, kind=KIND_DOCUMENT)


def test_a_trashed_answer_is_proof_without_any_activity_history():
    result = probe_removal(_Source(a="trashed"), "a")

    assert result.verdict is Verdict.GONE
    assert result.removable is True
    assert "does not exist" in result.detail


def test_a_bare_not_found_is_unproven_with_no_explanation_at_all():
    # The defect this module used to have: a 404 and a lost permission are
    # answered identically by the source, so a bare not-found cannot be read
    # as proof of deletion (principle P-005).
    result = probe_removal(_Source(a="missing"), "a")

    assert result.verdict is Verdict.UNPROVEN
    assert result.removable is False
    assert "absence is never proof" in result.detail


def test_a_denied_answer_leaves_the_document_alone_and_says_why():
    result = probe_removal(_Source(a="denied"), "a")

    assert result.verdict is Verdict.ACCESS_LOST
    assert result.removable is False
    # The asymmetry is the whole point: one of these is recoverable by fixing a
    # share, the other only from a backup.
    assert "not a deletion" in result.detail


def test_a_document_handed_over_means_the_feed_was_premature():
    result = probe_removal(_Source(a="present"), "a")

    assert result.verdict is Verdict.STILL_THERE
    assert result.removable is False
    assert result.document is not None
    assert result.document.external_id == "a"


def test_an_activity_deletion_event_proves_a_bare_not_found():
    event = DeletionEvent(external_id="a", kind=DeletionKind.PERMANENTLY_DELETED, occurred_at=None)

    result = probe_removal(_Source(a="missing"), "a", explained_by=(event,))

    assert result.verdict is Verdict.GONE
    assert result.removable is True
    assert "permanently_deleted" in result.detail


def test_an_activity_move_event_vetoes_a_bare_not_found():
    event = MoveEvent(external_id="a", removed_parents=("root-folder",), occurred_at=None)

    result = probe_removal(_Source(a="missing"), "a", explained_by=(event,))

    assert result.verdict is Verdict.ACCESS_LOST
    assert result.removable is False
    assert "move" in result.detail


def test_an_activity_permission_change_event_vetoes_a_bare_not_found():
    event = PermissionChangeEvent(external_id="a", removed_permissions=("writer",), occurred_at=None)

    result = probe_removal(_Source(a="missing"), "a", explained_by=(event,))

    assert result.verdict is Verdict.ACCESS_LOST
    assert result.removable is False
    assert "permission change" in result.detail


def test_a_denied_probe_never_needs_the_activity_history():
    # Conclusive on its own — an event for a different reason would be a
    # coincidence, not a contradiction, and must not override it.
    event = DeletionEvent(external_id="a", kind=DeletionKind.TRASHED, occurred_at=None)

    result = probe_removal(_Source(a="denied"), "a", explained_by=(event,))

    assert result.verdict is Verdict.ACCESS_LOST


def test_a_successful_probe_never_needs_the_activity_history():
    # The current state wins over a stale event: access may have been restored
    # since.
    event = PermissionChangeEvent(external_id="a", removed_permissions=("writer",), occurred_at=None)

    result = probe_removal(_Source(a="present"), "a", explained_by=(event,))

    assert result.verdict is Verdict.STILL_THERE


def test_the_summary_sorts_the_answers_into_what_may_be_acted_on():
    source = _Source(gone="trashed", hidden="denied", quiet="missing", here="present")

    summary = probe_removals(source, ["gone", "hidden", "quiet", "here"])

    assert summary.gone == ("gone",)
    assert summary.access_lost == ("hidden",)
    assert summary.unproven == ("quiet",)
    assert [r.external_id for r in summary.still_there] == ["here"]
    assert summary.lost_any_access is True


def test_the_summary_passes_each_documents_own_explanations_along():
    source = _Source(a="missing", b="missing")
    explanations = {
        "a": (DeletionEvent(external_id="a", kind=DeletionKind.PERMANENTLY_DELETED, occurred_at=None),),
    }

    summary = probe_removals(source, ["a", "b"], explanations=explanations)

    assert summary.gone == ("a",)
    assert summary.unproven == ("b",)


def test_the_probes_happen_in_a_stable_order():
    # A set's iteration order would make two identical runs produce different
    # records, and a record that changes between identical runs cannot be
    # compared to the last one.
    source = _Source(c="trashed", a="trashed", b="trashed")

    probe_removals(source, ["c", "a", "b", "a"])

    assert source.asked == ["a", "b", "c"]


def test_duplicates_and_blanks_are_not_probed():
    source = _Source(a="trashed")

    probe_removals(source, ["a", "a", "", "a"])

    assert source.asked == ["a"]


def test_nothing_reported_asks_nothing():
    source = _Source()

    summary = probe_removals(source, [])

    assert summary.results == ()
    assert summary.gone == ()
    assert summary.lost_any_access is False
    assert source.asked == []


def test_the_report_is_one_line_an_operator_can_read_off_a_run_record():
    source = _Source(gone="trashed", also_gone="trashed", hidden="denied", quiet="missing", here="present")

    report = probe_removals(source, ["gone", "also_gone", "hidden", "quiet", "here"]).report()

    assert "2 confirmed gone" in report
    # "We lost sight of documents last night" has to be impossible to miss,
    # which is why the sentence is composed here and not at the call site.
    assert "1 left indexed because access was refused or explained otherwise" in report
    assert "1 left indexed because the source offered no proof either way" in report
    assert "1 still present despite being reported removed" in report


def test_a_report_with_nothing_unusual_says_only_what_happened():
    summary = probe_removals(_Source(a="trashed"), ["a"])

    assert summary.report() == "1 confirmed gone"


def test_an_empty_summary_still_reports_a_number():
    assert ProbeSummary(results=()).report() == "0 confirmed gone"
