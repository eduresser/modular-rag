"""Unit-level coverage for src/sources/gdrive_export.py.

AC-016 and AC-017 are proven in tests/test_spec_source_gdrive.py — see
CLAUDE.md for why the proofs live there. This file covers the rest of the
mapping: what is not a document, what has no mapping, and the export ceiling
that has to be a hard failure rather than a truncation.
"""

import pytest

from src.sources.gdrive_export import (
    EXPORT_SIZE_LIMIT_BYTES,
    KIND_DOCUMENT,
    KIND_FOLDER,
    KIND_PRESENTATION,
    KIND_SHORTCUT,
    KIND_SPREADSHEET,
    MEDIA_DOCX,
    MEDIA_PDF,
    MEDIA_PPTX,
    MEDIA_XLSX,
    ExportTooLarge,
    Retrieval,
    UnsupportedKind,
    content_digest,
    export_plan,
    is_document,
    reject_oversized,
    reject_oversized_content,
)


def test_a_native_document_exports_to_a_paginated_rendering_and_says_why():
    plan = export_plan(KIND_DOCUMENT)

    assert plan.retrieval is Retrieval.EXPORT
    # Not the word-processor format any more (AC-210): the reader returned no
    # page and no heading for a block of it, measured on the one document
    # this corpus actually needed it for.
    assert plan.media_type == MEDIA_PDF
    # The rationale is carried rather than left in a comment because "why is
    # this PDF and not the word-processor format" gets asked again every time.
    assert "AC-210" in plan.rationale


def test_a_native_document_no_longer_needs_local_rendering():
    # The source renders it to PDF itself now, so it never reaches
    # src/processing/render.py at all — that module exists for the one case
    # the source refuses outright: an uploaded word-processor file.
    from src.processing.preflight import UNPAGINATED_MEDIA_TYPES

    assert export_plan(KIND_DOCUMENT).media_type not in UNPAGINATED_MEDIA_TYPES


def test_every_export_target_is_a_format_the_reader_accepts():
    # The rule that was learned the hard way: an export the next step cannot
    # read is not an export. Markdown kept the headings perfectly and the
    # layout parser does not take text files, so a whole corpus of documents
    # authored in Drive would have been exported and then refused.
    from src.processing.preflight import SUPPORTED_MEDIA_TYPES

    for kind in (KIND_DOCUMENT, KIND_SPREADSHEET, KIND_PRESENTATION):
        assert export_plan(kind).media_type in SUPPORTED_MEDIA_TYPES, kind


def test_a_spreadsheet_exports_to_the_workbook_and_never_to_csv():
    plan = export_plan(KIND_SPREADSHEET)

    assert plan.media_type == MEDIA_XLSX
    assert "csv" not in plan.media_type
    assert "only the first one" in plan.rationale


def test_a_presentation_keeps_slide_boundaries():
    assert export_plan(KIND_PRESENTATION).media_type == MEDIA_PPTX


def test_an_uploaded_file_is_taken_as_it_is():
    # There is nothing to convert, and converting would lose whatever the
    # reader could have made of the original.
    for kind in ("application/pdf", "application/msword", "image/png"):
        plan = export_plan(kind)
        assert plan.retrieval is Retrieval.DOWNLOAD
        assert plan.media_type == kind
        assert plan.is_export is False


def test_a_folder_and_a_shortcut_are_structure_not_documents():
    assert is_document(KIND_FOLDER) is False
    assert is_document(KIND_SHORTCUT) is False
    assert is_document(KIND_DOCUMENT) is True


def test_asking_to_export_a_folder_says_it_is_not_a_document():
    # Named explicitly so folders are skipped rather than filling a run record
    # with an error for every folder in the corpus.
    with pytest.raises(UnsupportedKind) as excinfo:
        export_plan(KIND_FOLDER)

    assert "not a document" in str(excinfo.value)


def test_asking_to_export_a_shortcut_points_at_the_resolution_instead():
    with pytest.raises(UnsupportedKind) as excinfo:
        export_plan(KIND_SHORTCUT)

    assert "AC-015" in str(excinfo.value)


def test_an_unmapped_editor_type_fails_by_name_rather_than_falling_back_to_text():
    # A Google Form exported as text yields its title and nothing else, which
    # reads as coverage of a document the archive does not hold.
    with pytest.raises(UnsupportedKind) as excinfo:
        export_plan("application/vnd.google-apps.form")

    message = str(excinfo.value)
    assert "application/vnd.google-apps.form" in message
    assert "reads as coverage" in message
    # And it lists what is mapped, so the fix does not require reading code.
    assert KIND_DOCUMENT in message


def test_a_known_size_over_the_ceiling_is_refused_before_the_request_goes_out():
    with pytest.raises(ExportTooLarge) as excinfo:
        reject_oversized("doc-1", EXPORT_SIZE_LIMIT_BYTES + 1)

    message = str(excinfo.value)
    assert "doc-1" in message
    # The source offers no partial download, so the alternatives are an honest
    # error and a truncated document nobody knows is truncated (ASM-011).
    assert "ASM-011" in message
    assert "no\npartial download" in message.replace(" ", "\n", 1) or "partial download" in message


def test_a_size_at_the_ceiling_is_allowed():
    reject_oversized("doc-1", EXPORT_SIZE_LIMIT_BYTES)  # must not raise


def test_an_unknown_size_is_not_a_refusal():
    # A native editor document's exported size cannot be known in advance, so
    # the check happens again on the bytes.
    reject_oversized("doc-1", None)  # must not raise


def test_bytes_over_the_ceiling_are_refused_when_they_arrive():
    with pytest.raises(ExportTooLarge) as excinfo:
        reject_oversized_content("doc-1", b"x" * (EXPORT_SIZE_LIMIT_BYTES + 1))

    assert "exported to" in str(excinfo.value)


def test_bytes_within_the_ceiling_come_straight_back():
    content = b"# Contrato"

    assert reject_oversized_content("doc-1", content) is content


def test_the_ceiling_is_the_sources_own_ten_megabytes():
    assert EXPORT_SIZE_LIMIT_BYTES == 10 * 1024 * 1024


def test_the_digest_is_over_exactly_the_exported_bytes():
    # Which is why the export format is part of identity: the same document
    # exported two ways is two digests, so changing the format invalidates the
    # corpus the way changing the parser does.
    assert content_digest(b"# Contrato") == content_digest(b"# Contrato")
    assert content_digest(b"# Contrato") != content_digest(b"# Contrato ")
    assert len(content_digest(b"")) == 64
