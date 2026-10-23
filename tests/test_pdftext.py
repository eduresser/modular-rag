import io

import pytest
from pypdf import PdfWriter

from src.models import Chunk, ReaderCutChunk, VersionStamps
from src.processing import pdftext
from src.processing.render import MEDIA_PDF
from src.processing.to_chunks import NUL, MappingError
from src.sources.gdrive_export import MEDIA_DOCX, MEDIA_XLSX
from tests.conftest import pdf_bytes

STAMPS = VersionStamps(reader_version="pypdf-x", fingerprint="fp-1")


def _payload(*pages: str) -> dict:
    return pdftext.extract(pdf_bytes(list(pages)), external_id="doc-1")


def test_the_kept_payload_names_what_produced_it_and_holds_every_page():
    payload = _payload("Cláusula primeira.", "", "Cláusula terceira.")

    assert payload["extractor"] == "pypdf"
    assert payload["extractorVersion"] == pdftext.extractor_version()
    assert [entry["page"] for entry in payload["pages"]] == [1, 2, 3]
    assert payload["pages"][1]["text"] == ""


def test_the_version_is_measured_off_the_installed_package():
    from importlib import metadata

    assert pdftext.extractor_version() == f"pypdf-{metadata.version('pypdf')}"
    assert pdftext.extractor_version() == pdftext.extractor_version()


def test_a_file_that_is_not_a_pdf_is_one_documents_failure_and_not_an_escape():
    with pytest.raises(pdftext.PdfTextError, match="could not be opened as a PDF"):
        pdftext.extract(b"isto nao e um pdf", external_id="lixo-1")


def test_a_pdf_of_blank_pages_is_refused_under_a_countable_reason():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    with pytest.raises(pdftext.NoTextLayer) as refusal:
        pdftext.extract(buffer.getvalue(), external_id="scan-1")

    assert pdftext.NO_TEXT_REASON in str(refusal.value)
    from src.ingest_errors import FAILURE_PERMANENT, classify

    assert classify(refusal.value) == FAILURE_PERMANENT


def test_a_payload_that_is_not_this_modules_says_so_rather_than_cutting_nothing():
    with pytest.raises(MappingError, match="holds no pages"):
        pdftext.chunks_from_document(
            {"documentLayout": {"blocks": []}}, external_id="doc-1", stamps=STAMPS
        )


def test_every_page_is_a_block_and_a_blank_one_keeps_its_number():
    index = pdftext.index_of(_payload("primeira", "", "terceira"))

    assert index.order == ("p1", "p2", "p3")
    assert index["p2"].text == ""
    assert index["p3"].page_start == 3 and index["p3"].page_end == 3
    assert index["p1"].kinds == ("page",)


def test_a_piece_is_a_page_and_carries_the_page_it_is():
    mapped = pdftext.chunks_from_document(
        _payload("primeira", "", "terceira"), external_id="doc-1", stamps=STAMPS
    )

    assert [(piece.content, piece.pages) for piece in mapped.chunks] == [
        ("primeira", (1, 1)),
        ("terceira", (3, 3)),
    ]
    assert [piece.source_block_ids for piece in mapped.chunks] == [("p1",), ("p3",)]


def test_a_piece_from_this_module_is_a_base_record_and_not_a_readers_cut():
    [piece] = pdftext.chunks_from_document(
        _payload("primeira"), external_id="doc-1", stamps=STAMPS
    ).chunks

    assert isinstance(piece, Chunk)
    assert not isinstance(piece, ReaderCutChunk)
    assert piece.to_piece([0.0]).metadata["source_block_ids"] == ["p1"]
    assert piece.to_piece([0.0]).metadata["block_kinds"] == ["page"]


def test_a_document_whose_pages_all_went_blank_after_the_read_is_refused():
    with pytest.raises(MappingError, match=pdftext.NO_TEXT_REASON):
        pdftext.chunks_from_document(
            {"extractor": "pypdf", "extractorVersion": "x",
             "pages": [{"page": 1, "text": ""}, {"page": 2, "text": "   "}]},
            external_id="doc-1",
            stamps=STAMPS,
        )


def test_a_re_cut_with_no_identity_to_give_keeps_the_placeholder_name():
    index = pdftext.index_of(_payload("primeira"))

    assert index.identity.name == "document"
    assert index.identity.uri == ""


def test_the_identity_is_built_only_when_every_fact_it_names_was_given():
    complete = pdftext.index_of(
        _payload("primeira"),
        external_id="doc-42",
        filename="contrato.pdf",
        media_type=MEDIA_PDF,
        content_digest="ab" * 32,
    ).identity
    partial = pdftext.index_of(_payload("primeira"), external_id="doc-42").identity

    assert (complete.name, complete.uri) == ("contrato.pdf", "source:doc-42")
    assert partial.name == "document"


def test_it_takes_a_pdf_directly_and_a_word_processor_file_through_the_image():
    assert pdftext.rendering_for(MEDIA_PDF) == ""
    assert pdftext.rendering_for(MEDIA_DOCX) == MEDIA_PDF
    assert pdftext.rendering_for(MEDIA_XLSX) == ""
    assert MEDIA_XLSX not in pdftext.SUPPORTED_MEDIA_TYPES


def test_its_ceilings_are_its_own_and_not_a_providers():
    from src.processing.preflight import BATCH_BYTE_LIMIT
    from src.processing.docai import BATCH_PAGE_LIMIT

    assert pdftext.LIMITS.pages == pdftext.PAGE_LIMIT != BATCH_PAGE_LIMIT
    assert pdftext.LIMITS.byte_size == pdftext.BYTE_LIMIT != BATCH_BYTE_LIMIT
    assert pdftext.LIMITS.narrowed(pages=50).pages == 50
    assert pdftext.LIMITS.narrowed(pages=99_999).pages == pdftext.PAGE_LIMIT


def test_a_glyph_with_no_unicode_mapping_is_dropped_rather_than_stored():
    from src.processing.to_chunks import NUL, storable

    formula = f"de acordo com a seguinte fórmula:\n{NUL}\n = \n{NUL}{NUL}\nNP = Nota da Proposta"

    assert NUL not in storable(formula)
    assert "NP = Nota da Proposta" in storable(formula)
    assert storable(formula).count("�") == 0
    assert storable("") == "" and storable("sem nada a tirar") == "sem nada a tirar"


def test_the_module_drops_it_on_the_way_in_and_on_the_way_out():
    novo = pdftext.extract(pdf_bytes(["fórmula: fim"]), external_id="doc-1")
    assert all(NUL not in entry["text"] for entry in novo["pages"])

    antigo = {
        "extractor": "pypdf",
        "extractorVersion": "x",
        "pages": [
            {"page": 1, "text": f"fórmula {NUL}{NUL} antiga"},
            {"page": 2, "text": NUL * 3},
        ],
    }
    mapped = pdftext.chunks_from_document(antigo, external_id="doc-1", stamps=STAMPS)

    assert [piece.content for piece in mapped.chunks] == ["fórmula  antiga"]
    assert len(mapped.chunks) == 1


def test_a_module_that_hands_the_byte_straight_through_is_named_at_the_piece():
    from src.models import Chunk, ChunkError

    with pytest.raises(ChunkError, match="NUL byte"):
        Chunk(
            document_external_id="doc-1",
            ordinal=0,
            content=f"fórmula {NUL} solta",
            stamps=STAMPS,
            page_start=1,
            page_end=1,
        )
