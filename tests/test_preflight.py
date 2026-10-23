import psycopg
import pytest

from src.processing.artifact import READ_STATE_FAILED, read_record
from src.processing.preflight import (
    BATCH_BYTE_LIMIT,
    BATCH_PAGE_LIMIT,
    MEDIA_HTML,
    OFFICE_MEDIA_TYPES,
    OUTCOME_READ,
    OUTCOME_REFUSED,
    SCANNED_MEDIA_TYPES,
    UNPAGINATED_MEDIA_TYPES,
    UNREADABLE_TEXT_MEDIA_TYPES,
    SUPPORTED_MEDIA_TYPES,
    Candidate,
    Verdict,
    byte_limit_for,
    page_limit_for,
    preflight,
    record_refusals,
    screen,
)
from src.sources.gdrive_export import (
    KIND_DOCUMENT,
    KIND_PRESENTATION,
    KIND_SPREADSHEET,
    export_plan,
)
from src.store import upsert_document
from tests.conftest import DOCAI_PROCESSOR, DOCAI_VERSION

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _candidate(**overrides) -> Candidate:
    base = dict(
        document_id=1,
        external_id="contrato-1",
        media_type="application/pdf",
        byte_size=2_000_000,
        page_count=40,
    )
    base.update(overrides)
    return Candidate(**base)


def test_an_ordinary_pdf_passes():
    assert preflight(_candidate()).accepted is True


def test_a_document_over_the_page_limit_names_the_limit_and_its_own_count():
    verdict = preflight(_candidate(page_count=BATCH_PAGE_LIMIT + 1))

    assert verdict.accepted is False
    assert str(BATCH_PAGE_LIMIT) in verdict.reason
    assert str(BATCH_PAGE_LIMIT + 1) in verdict.reason


def test_exactly_at_the_limit_passes():
    assert preflight(_candidate(page_count=BATCH_PAGE_LIMIT)).accepted is True


def test_a_document_over_the_byte_limit_is_refused_even_with_few_pages():
    verdict = preflight(_candidate(page_count=1, byte_size=BATCH_BYTE_LIMIT + 1))

    assert verdict.accepted is False
    assert str(BATCH_BYTE_LIMIT) in verdict.reason


def test_a_document_whose_page_count_is_unknown_is_still_screened_by_size():
    assert preflight(_candidate(page_count=None)).accepted is True
    assert preflight(_candidate(page_count=None, byte_size=BATCH_BYTE_LIMIT + 1)).accepted is False


def test_the_limits_in_force_are_the_batch_paths_because_it_is_the_only_one():
    assert page_limit_for() == BATCH_PAGE_LIMIT
    assert byte_limit_for() == BATCH_BYTE_LIMIT


def test_a_deployment_may_lower_a_limit_and_may_not_raise_one():
    assert page_limit_for(configured=50) == 50
    assert page_limit_for(configured=BATCH_PAGE_LIMIT + 5_000) == BATCH_PAGE_LIMIT
    assert byte_limit_for(configured=BATCH_BYTE_LIMIT * 4) == BATCH_BYTE_LIMIT
    assert preflight(_candidate(page_count=80), page_limit=50).accepted is False


def test_nothing_the_source_can_hand_over_is_refused_for_its_format():
    from_source = [export_plan(kind).media_type for kind in (KIND_DOCUMENT, KIND_SPREADSHEET, KIND_PRESENTATION)]
    uploaded = [DOCX, "application/pdf", "image/jpeg"]

    for media_type in from_source + uploaded:
        assert preflight(_candidate(media_type=media_type)).refused is False, media_type


def test_a_word_processor_file_goes_to_the_reader():
    assert preflight(_candidate(media_type=DOCX)).outcome == OUTCOME_READ


def test_every_office_format_goes_to_the_reader():
    for media_type in OFFICE_MEDIA_TYPES:
        assert preflight(_candidate(media_type=media_type)).outcome == OUTCOME_READ, media_type


def test_a_word_processor_file_over_the_limit_says_what_a_page_means_here():
    verdict = preflight(_candidate(media_type=DOCX, page_count=BATCH_PAGE_LIMIT + 1))

    assert verdict.refused is True
    assert "3,000 characters" in verdict.reason


def test_html_is_accepted_and_plain_text_is_not():
    assert preflight(_candidate(media_type=MEDIA_HTML)).outcome == OUTCOME_READ

    for media_type in UNREADABLE_TEXT_MEDIA_TYPES:
        verdict = preflight(_candidate(media_type=media_type))
        assert verdict.outcome == OUTCOME_REFUSED, media_type
        assert "does not accept text files" in verdict.reason


def test_a_type_the_reader_does_not_take_is_refused_and_the_reason_lists_what_it_does():
    verdict = preflight(_candidate(media_type="image/svg+xml"))

    assert verdict.outcome == OUTCOME_REFUSED
    assert "image/svg+xml" in verdict.reason
    assert "application/pdf" in verdict.reason


def test_every_supported_type_is_accepted():
    for media_type in SUPPORTED_MEDIA_TYPES:
        assert preflight(_candidate(media_type=media_type)).accepted is True, media_type


def test_the_word_processor_format_is_the_one_measured_as_unpaginated():
    assert UNPAGINATED_MEDIA_TYPES == {OFFICE_MEDIA_TYPES[0]}
    for media_type in SCANNED_MEDIA_TYPES:
        assert media_type not in UNPAGINATED_MEDIA_TYPES, media_type
    for media_type in OFFICE_MEDIA_TYPES[1:]:
        assert media_type not in UNPAGINATED_MEDIA_TYPES, media_type


def test_a_refusal_that_names_no_limit_cannot_be_built():
    with pytest.raises(ValueError, match="AC-035"):
        Verdict(_candidate(), OUTCOME_REFUSED, "  ")


def test_a_verdict_nobody_recognises_cannot_be_built():
    with pytest.raises(ValueError, match="outcome"):
        Verdict(_candidate(), "talvez")


def test_screening_never_raises_and_keeps_the_order_of_what_passed():
    candidates = [
        _candidate(document_id=1, external_id="contrato.pdf"),
        _candidate(document_id=2, external_id="processo.pdf", page_count=BATCH_PAGE_LIMIT + 1),
        _candidate(document_id=3, external_id="parecer.docx", media_type=DOCX),
        _candidate(document_id=4, external_id="minuta-no-editor", media_type=export_plan(KIND_DOCUMENT).media_type),
        _candidate(document_id=5, external_id="logo.svg", media_type="image/svg+xml"),
    ]

    screening = screen(candidates)

    assert [c.external_id for c in screening.accepted] == [
        "contrato.pdf", "parecer.docx", "minuta-no-editor"
    ]
    assert [v.candidate.external_id for v in screening.rejected] == ["processo.pdf", "logo.svg"]
    assert screening.clean is False
    assert screen([_candidate()]).clean is True


def test_the_reader_and_its_version_are_recorded_against_a_refusal(conn: psycopg.Connection):
    document_id = upsert_document(conn, external_id="processo", title="Processo")
    screening = screen([_candidate(document_id=document_id, page_count=BATCH_PAGE_LIMIT + 90)])

    written = record_refusals(
        conn, screening, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION
    )

    assert written == 1
    record = read_record(conn, document_id)
    assert record.state == READ_STATE_FAILED
    assert record.reader_version == DOCAI_VERSION
    assert record.page_count == BATCH_PAGE_LIMIT + 90


def test_recording_nothing_is_not_an_error(conn: psycopg.Connection):
    assert record_refusals(
        conn, screen([_candidate()]), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION
    ) == 0
