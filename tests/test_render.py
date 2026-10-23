import io
import zipfile

import pytest

from src.processing.preflight import OFFICE_MEDIA_TYPES, SCANNED_MEDIA_TYPES, UNPAGINATED_MEDIA_TYPES
from src.processing.render import (
    MEDIA_PDF,
    RenderedDocument,
    RenderError,
    docai_rendering_for,
    render,
    render_to_pdf,
)

DOCX = OFFICE_MEDIA_TYPES[0]

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_PACKAGE_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def real_word_document(text: str = "texto da cláusula") -> bytes:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD_NAMESPACE}"><w:body>'
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _PACKAGE_RELS)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def test_the_word_processor_format_and_only_that_one_needs_rendering():
    assert docai_rendering_for(DOCX) == MEDIA_PDF
    assert UNPAGINATED_MEDIA_TYPES == {DOCX}


def test_a_pdf_and_a_scan_already_paginate_and_are_left_alone():
    for media_type in SCANNED_MEDIA_TYPES:
        assert docai_rendering_for(media_type) == "", media_type


def test_a_spreadsheet_and_a_presentation_are_out_of_scope_today():
    for media_type in OFFICE_MEDIA_TYPES[1:]:
        assert docai_rendering_for(media_type) == "", media_type


def test_a_target_the_image_cannot_render_to_is_refused_rather_than_attempted():
    with pytest.raises(RenderError, match="renders to application/pdf and nothing else"):
        render("doc-1", DOCX, b"", to="text/plain")


def test_a_format_that_needs_no_rendering_comes_back_unchanged_and_the_renderer_is_not_called():
    def _fails_if_called(media_type: str, content: bytes) -> bytes:
        raise AssertionError("a format that already paginates must never reach the renderer")

    result = render("doc-1", "application/pdf", b"%PDF-1.4 already paginated", run=_fails_if_called)

    assert result == RenderedDocument(media_type="application/pdf", content=b"%PDF-1.4 already paginated")


def test_rendering_never_happens_twice():
    calls = []

    def _record(media_type: str, content: bytes) -> bytes:
        calls.append(media_type)
        return b"%PDF-1.4 rendered"

    once = render("doc-1", DOCX, b"docx bytes", run=_record, to=docai_rendering_for(DOCX))
    assert once.media_type == MEDIA_PDF
    twice = render("doc-1", once.media_type, once.content, run=_record)

    assert twice.content == once.content
    assert calls == [DOCX]


def test_a_renderer_that_raises_is_reported_as_this_documents_own_failure():
    def _blows_up(media_type: str, content: bytes) -> bytes:
        raise RuntimeError("the office suite crashed")

    with pytest.raises(RenderError) as excinfo:
        render("contrato-7", DOCX, b"docx bytes", run=_blows_up, to=docai_rendering_for(DOCX))

    message = str(excinfo.value)
    assert "contrato-7" in message
    assert "the office suite crashed" in message


def test_a_render_error_raised_by_the_renderer_itself_passes_through_unwrapped():
    def _raises_render_error(media_type: str, content: bytes) -> bytes:
        raise RenderError("already a per-document failure")

    with pytest.raises(RenderError, match="already a per-document failure"):
        render("doc-1", DOCX, b"docx bytes", run=_raises_render_error, to=docai_rendering_for(DOCX))


def test_render_to_pdf_refuses_a_format_it_does_not_recognise():
    with pytest.raises(RenderError) as excinfo:
        render_to_pdf("image/png", b"not office bytes")

    assert "image/png" in str(excinfo.value)


def test_the_real_office_suite_renders_a_word_processor_file_to_pdf():
    rendered = render("contrato-1", DOCX, real_word_document(), to=docai_rendering_for(DOCX))

    assert rendered.media_type == MEDIA_PDF
    assert rendered.content.startswith(b"%PDF")
