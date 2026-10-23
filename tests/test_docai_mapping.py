"""Unit-level coverage for the read module's own walk that is not itself a criterion.

AC-030, AC-031 and AC-032 are proven in tests/test_spec_processing_docai.py —
see CLAUDE.md. This file covers the block walk itself: the heading path in both
shapes the reader emits, the page range derived through the block identifiers
(ASM-023), and the rendering defects the house's earlier pipeline shipped.
"""

import pytest

from src.models import ReaderCutChunk, VersionStamps
from src.processing.docai import (
    BLOCK_LIST,
    BLOCK_TABLE,
    BLOCK_TEXT,
    chunks_from_document,
    document_of,
    index_blocks,
    index_of,
    reader_chunks_of,
    render_list,
    render_table,
)
from src.processing.to_chunks import Box, CutPiece, MappingError, cut_reader_chunks
from tests.conftest import (
    DOCAI_VERSION,
    list_block,
    reader_chunk,
    reader_payload,
    table_block,
    text_block,
)

STAMPS = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")
EXTERNAL_ID = "doc-1"


def _map(blocks, chunks):
    return chunks_from_document(
        reader_payload(blocks, chunks), external_id=EXTERNAL_ID, stamps=STAMPS
    )


def test_the_heading_path_deepens_through_nested_blocks():
    # The reader nests a section's content inside the heading's own blocks.
    nested = [
        text_block(
            "h1", "CONTRATO", pages=(1, 1), kind="heading-1",
            blocks=[
                text_block(
                    "h2", "Cláusula Primeira", pages=(1, 1), kind="heading-2",
                    blocks=[text_block("p1", "texto da cláusula", pages=(1, 1))],
                )
            ],
        )
    ]
    index = index_blocks(reader_payload(nested, []))

    assert index["p1"].heading_path == ("CONTRATO", "Cláusula Primeira")


def test_the_heading_path_is_carried_along_a_flat_document_too():
    # It also emits long documents flat, heading and content as siblings. A
    # walk that only handled nesting would give every clause of one of those an
    # empty heading path.
    flat = [
        text_block("h1", "CONTRATO", pages=(1, 1), kind="heading-1"),
        text_block("h2", "Cláusula Primeira", pages=(1, 1), kind="heading-2"),
        text_block("p1", "texto da cláusula", pages=(1, 1)),
        text_block("h3", "Cláusula Segunda", pages=(2, 2), kind="heading-2"),
        text_block("p2", "texto da segunda", pages=(2, 2)),
    ]
    index = index_blocks(reader_payload(flat, []))

    assert index["p1"].heading_path == ("CONTRATO", "Cláusula Primeira")
    # The second clause replaces the first at its own level rather than
    # accumulating beneath it.
    assert index["p2"].heading_path == ("CONTRATO", "Cláusula Segunda")


def test_a_heading_four_levels_down_is_a_heading():
    # The earlier pipeline's prefix table stopped at three and silently
    # demoted heading-4, heading-5 and subtitle to paragraphs.
    deep = [
        text_block("h1", "A", pages=(1, 1), kind="heading-1"),
        text_block("h4", "B", pages=(1, 1), kind="heading-4"),
        text_block("p1", "texto", pages=(1, 1)),
    ]
    index = index_blocks(reader_payload(deep, []))

    assert index["p1"].heading_path == ("A", "B")


def test_the_page_range_comes_through_the_block_identifiers():
    # ASM-023: the field the reference documentation marks unused and the
    # provider's own example fills. It is what a page range is derived from.
    blocks = [
        text_block("b1", "início", pages=(4, 4)),
        text_block("b2", "meio", pages=(5, 6)),
    ]
    mapped = _map(blocks, [reader_chunk("c1", "início meio", ["b1", "b2"])])

    assert mapped.chunks[0].pages == (4, 6)
    assert mapped.chunks[0].source_block_ids == ("b1", "b2")


def test_a_blocks_bounding_box_reaches_the_piece_it_was_cut_from():
    # AC-209: where the reader returns geometry, it reaches the piece rather
    # than being discarded — kept from the first read, because collecting it
    # later costs the per-page rate again.
    block = text_block("p1", "texto da cláusula", pages=(3, 3))
    block["boundingBox"] = {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.3}

    index = index_blocks(reader_payload([block], []))

    assert index["p1"].box == Box(left=0.1, top=0.2, right=0.9, bottom=0.3)
    assert index["p1"].page_start == 3
    [piece] = cut_reader_chunks(index, reader_chunks_of(
        reader_payload([block], [reader_chunk("c1", "texto da cláusula", ["p1"])])
    ))
    assert piece.boxes == (Box(left=0.1, top=0.2, right=0.9, bottom=0.3),)


def test_a_block_with_no_bounding_box_still_builds_the_piece_with_the_page_alone():
    # AC-209's third clause: asking costs nothing when the reader answers no.
    block = text_block("p1", "texto da cláusula", pages=(3, 3))

    index = index_blocks(reader_payload([block], []))

    assert index["p1"].box is None
    assert index["p1"].page_start == 3
    mapped = _map([block], [reader_chunk("c1", "texto da cláusula", ["p1"])])
    assert mapped.chunks[0].pages == (3, 3)


def test_a_malformed_bounding_box_is_worth_exactly_as_much_as_none_at_all():
    block = text_block("p1", "texto da cláusula", pages=(3, 3))
    block["boundingBox"] = {"left": 0.1, "top": "acima"}

    assert index_blocks(reader_payload([block], []))["p1"].box is None


def test_a_chunk_whose_blocks_are_unknown_falls_back_to_its_own_span():
    # The regression ASM-023 warns about: if sourceBlockIds ever stops
    # arriving, a piece still carries a page instead of becoming uncitable and
    # being refused by the store.
    mapped = _map([], [reader_chunk("c1", "texto solto", [], pages=(9, 9))])

    assert mapped.chunks[0].pages == (9, 9)


def test_a_chunk_with_no_page_and_no_heading_anywhere_stops_the_document():
    with pytest.raises(MappingError, match="quietly smaller"):
        _map([], [reader_chunk("c1", "texto sem lugar", [])])


def test_a_page_header_or_footer_block_is_kept_apart_rather_than_deleted():
    # AC-201: out of the blocks the cut reads, which is what keeps a running
    # header or footer out of every answer — and still somewhere, so a viewer
    # reading the document finds them where the reader put them.
    blocks = [
        text_block("h1", "Contrato nº 123 — página 3", pages=(3, 3), kind="header"),
        text_block("f1", "Confidencial", pages=(3, 3), kind="footer"),
        text_block("p1", "texto da cláusula", pages=(3, 3)),
    ]
    index = index_blocks(reader_payload(blocks, []))

    assert "h1" not in index.blocks
    assert "f1" not in index.blocks
    assert index["p1"].text == "texto da cláusula"
    assert {block_id: block.text for block_id, block in index.furniture.items()} == {
        "h1": "Contrato nº 123 — página 3",
        "f1": "Confidencial",
    }
    # In the reader's own words, which is what says WHY it is furniture.
    assert index.furniture["h1"].kinds == (BLOCK_TEXT, "header")
    assert index.furniture["f1"].kinds == (BLOCK_TEXT, "footer")


def test_a_chunk_naming_a_furniture_block_does_not_pull_its_text_into_the_piece():
    # AC-202: the reader may name a header or footer block among a chunk's
    # sourceBlockIds — it names the page's furniture, not only its content —
    # and the piece is built from the content blocks only.
    blocks = [
        text_block("h1", "Contrato nº 123 — página 3", pages=(3, 3), kind="header"),
        text_block("p1", "texto da cláusula", pages=(3, 3)),
    ]
    # Content left empty, as some processor versions leave a table chunk: the
    # piece is rebuilt from its source blocks, which is exactly the path a
    # furniture block would leak into if it were still indexed as a paragraph.
    mapped = _map(blocks, [reader_chunk("c1", "", ["h1", "p1"])])

    assert mapped.chunks[0].content == "texto da cláusula"


def test_a_chunks_own_page_headers_and_footers_are_not_read_at_all():
    # AC-202: the reader also carries a chunk's page furniture directly on the
    # chunk (`pageHeaders`, `pageFooters`, `chunkFields`), separately from
    # `sourceBlockIds` and `content`. Nothing here reads the three, which is
    # what keeps a piece from carrying its own running header a second time.
    blocks = [text_block("p1", "texto da cláusula", pages=(3, 3))]
    raw_chunk = {
        "chunkId": "c1",
        "content": "texto da cláusula",
        "sourceBlockIds": ["p1"],
        "pageHeaders": [{"text": "Contrato nº 123 — página 3"}],
        "pageFooters": [{"text": "Confidencial"}],
        "chunkFields": [{"fieldName": "valor"}],
    }
    mapped = _map(blocks, [raw_chunk])

    assert mapped.chunks[0].content == "texto da cláusula"


def test_a_block_kind_nobody_recognises_is_counted_rather_than_skipped():
    # The confirmed failure mode: a walker that quietly ignores a block kind.
    # A new kind must not stop a corpus from being indexed, and must not pass
    # unnoticed either.
    strange = {"blockId": "x1", "pageSpan": {"pageStart": 1, "pageEnd": 1}, "audioBlock": {}}
    index = index_blocks(reader_payload([strange], []))

    assert index.unrecognised == ("x1",)
    assert index["x1"].kinds == ("audioBlock",)
    assert index["x1"].page_start == 1


def test_list_entries_are_indexed_as_blocks_and_are_not_emitted_twice():
    blocks = [list_block("l1", ["primeiro", "segundo"], pages=(3, 3))]
    index = index_blocks(reader_payload(blocks, []))

    assert index["l1"].kinds == (BLOCK_LIST, "ordered")
    assert index["l1"].contained is False
    # Their text is already inside the rendered list; emitting them again would
    # put every clause in the archive twice.
    assert index["l1-1"].contained is True
    assert index["l1-2"].contained is True


def test_an_ordered_list_is_numbered_and_an_unordered_one_is_not():
    ordered = render_list(
        {"type": "ordered", "listEntries": [{"blocks": [text_block("a", "um")]}, {"blocks": [text_block("b", "dois")]}]}
    )
    unordered = render_list(
        {"type": "unordered", "listEntries": [{"blocks": [text_block("a", "um")]}]}
    )

    assert ordered == "1. um\n2. dois"
    assert unordered == "- um"


def test_a_table_with_no_header_row_does_not_promote_a_row_of_data_to_one():
    # A confirmed defect of the earlier pipeline: with headerRows empty it made
    # the first row of data the header, losing a row of values and mislabelling
    # every other one.
    rendered = render_table(
        table_block("t1", [], [["1", "R$ 100,00"], ["2", "R$ 200,00"]])["tableBlock"]
    )

    assert rendered.splitlines() == ["| 1 | R$ 100,00 |", "| 2 | R$ 200,00 |"]
    assert "---" not in rendered


def test_a_table_keeps_its_header_when_it_has_one():
    rendered = render_table(
        table_block("t1", ["Parcela", "Valor"], [["1", "R$ 100,00"]])["tableBlock"]
    )

    assert rendered.splitlines()[0] == "| Parcela | Valor |"
    assert rendered.splitlines()[1] == "| --- | --- |"


def test_a_ragged_table_is_padded_rather_than_shifted():
    rendered = render_table(
        table_block("t1", ["A", "B", "C"], [["1"]])["tableBlock"]
    )

    assert rendered.splitlines()[-1] == "| 1 |  |  |"


def test_a_table_cell_holding_a_nested_list_keeps_the_list():
    cell_with_list = {
        "blockId": "t1",
        "pageSpan": {"pageStart": 1, "pageEnd": 1},
        "tableBlock": {
            "bodyRows": [
                {"cells": [{"blocks": [list_block("l9", ["um", "dois"], pages=(1, 1))]}]}
            ]
        },
    }
    index = index_blocks(reader_payload([cell_with_list], []))

    assert index["t1"].kinds == (BLOCK_TABLE,)
    assert "1. um" in index["t1"].text


def test_a_describing_text_already_inlined_is_not_appended_a_second_time():
    # The reader inlines an image's verbalisation into the content itself, and
    # leaves a table's caption on the block.
    described = table_block("t1", ["A"], [["1"]], pages=(1, 1), annotation="Quadro de valores")
    inline = _map([described], [reader_chunk("c1", "Quadro de valores\n| A |", ["t1"])])
    appended = _map([described], [reader_chunk("c2", "| A |\n| 1 |", ["t1"])])

    assert inline.chunks[0].content.count("Quadro de valores") == 1
    assert appended.chunks[0].content.count("Quadro de valores") == 1
    assert appended.chunks[0].content.endswith("> Quadro de valores")


def test_an_image_reaches_the_piece_as_its_description():
    image = {
        "blockId": "i1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {"annotations": {"description": "Planta baixa do imóvel"}},
    }
    mapped = _map([image], [reader_chunk("c1", "", ["i1"])])

    assert "Planta baixa do imóvel" in mapped.chunks[0].content
    assert mapped.chunks[0].pages == (2, 2)


def test_the_text_inside_an_image_reaches_the_piece_beside_its_description():
    # AC-205: a picture carries both a verbalisation and the words the reader
    # transcribed off the image itself — a signature block, a stamp — and both
    # have to reach the piece.
    image = {
        "blockId": "i1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {
            "annotations": {"description": "Assinatura e carimbo do cartório"},
            "imageText": "Cartório do 5º Ofício de Notas",
        },
    }
    mapped = _map([image], [reader_chunk("c1", "", ["i1"])])

    assert "Assinatura e carimbo do cartório" in mapped.chunks[0].content
    assert "Cartório do 5º Ofício de Notas" in mapped.chunks[0].content


def test_the_image_text_is_not_repeated_when_the_description_already_quotes_it():
    # The reader's own verbalisation sometimes already contains the exact
    # transcription — folding it in again would say the same words twice.
    image = {
        "blockId": "i1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {
            "annotations": {"description": 'Carimbo com o texto "Via original"'},
            "imageText": "Via original",
        },
    }
    mapped = _map([image], [reader_chunk("c1", "", ["i1"])])

    assert mapped.chunks[0].content.count("Via original") == 1


def test_the_images_own_texts_are_not_repeated_when_the_reader_already_inlined_them():
    # The reader may have already put both texts into the chunk's own content —
    # neither may be appended a second time (AC-205).
    image = {
        "blockId": "i1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {
            "annotations": {"description": "Assinatura e carimbo do cartório"},
            "imageText": "Cartório do 5º Ofício de Notas",
        },
    }
    inlined = (
        "Assinatura e carimbo do cartório\n\nCartório do 5º Ofício de Notas"
    )
    mapped = _map([image], [reader_chunk("c1", inlined, ["i1"])])

    assert mapped.chunks[0].content.count("Assinatura e carimbo do cartório") == 1
    assert mapped.chunks[0].content.count("Cartório do 5º Ofício de Notas") == 1


def test_the_payload_is_accepted_wrapped_or_bare():
    bare = reader_payload([text_block("b1", "texto")], [reader_chunk("c1", "texto", ["b1"])], wrapped=False)

    assert document_of(bare) is bare
    assert chunks_from_document(bare, external_id=EXTERNAL_ID, stamps=STAMPS).chunks


def test_a_payload_that_is_not_a_document_says_so():
    with pytest.raises(MappingError, match="no document"):
        document_of({"operation": "projects/x/operations/1"})


def test_a_block_the_reader_gave_no_identifier_is_indexed_under_its_position():
    # It used to be skipped, on the grounds that there was nothing to index it
    # under. There is: where it sits in the walk, which is unique inside the
    # document and is what a chunk naming nothing would have to resolve to
    # anyway. Skipping meant a block with no blockId left the archive with
    # nothing raised.
    anonymous = {"pageSpan": {"pageStart": 1, "pageEnd": 1}, "textBlock": {"text": "solto"}}
    index = index_blocks(reader_payload([anonymous], []))

    assert index.order == ("#/blocks/0",)
    assert index["#/blocks/0"].text == "solto"
    assert index["#/blocks/0"].page_start == 1


def test_the_read_module_hands_back_the_blocks_and_the_cut_it_made():
    # The obligation of a process module: its own provider's output in, the
    # blocks a cut reads and the cut the reader already made out. Everything
    # past the read stage takes those and nothing else, which is what a second
    # module has to satisfy to be pluggable — and none of it is anybody else's
    # document model.
    payload = reader_payload(
        [text_block("b1", "CONTRATO", pages=(1, 1), kind="heading-1")],
        [reader_chunk("c1", "CONTRATO", ["b1"])],
    )

    index = index_of(payload)
    reader_cut = reader_chunks_of(payload)

    assert index["b1"].heading_path == ()
    assert index["b1"].own_path == ("CONTRATO",)
    # The reader's own cut travels beside the blocks rather than inside them:
    # the index describes a document, not somebody's cut of it.
    assert [chunk.chunk_id for chunk in reader_cut] == ["c1"]


def test_a_re_cut_with_no_identity_to_give_keeps_the_placeholder_name():
    # A re-cut reads the stored payload with none of the identity arguments (it
    # has no source to ask) — the document then keeps naming nothing, rather
    # than an origin that is only partly true (AC-206).
    index = index_of(reader_payload([text_block("b1", "CONTRATO")], []))

    assert index.identity.name == "document"
    assert index.identity.external_id == ""
    assert index.identity.uri == ""


def test_an_identity_is_built_only_when_every_fact_it_names_was_given():
    # Two of the three is not two thirds of an origin — it is an origin that
    # says something false about the third.
    index = index_of(
        reader_payload([text_block("b1", "CONTRATO")], []),
        external_id="doc-42",
        filename="contrato.pdf",
    )

    assert index.identity.name == "document"


# ── What a chunk module hands back ────────────────────────────────────────
#
# The seam below the read stage is the blocks; the seam below the cut is a
# piece that POINTS AT the blocks it was cut from rather than copying facts out
# of them, so its pages and its kinds cannot drift from the document's.


def _cut(blocks, chunks):
    payload = reader_payload(blocks, chunks)
    index = index_of(payload)
    return index, list(cut_reader_chunks(index, reader_chunks_of(payload)))


def test_a_chunk_module_hands_back_a_piece_that_points_at_its_blocks():
    index, cut = _cut(
        [
            text_block("h1", "CONTRATO", pages=(1, 1), kind="heading-1"),
            text_block("p1", "Cláusula primeira.", pages=(1, 1)),
        ],
        [reader_chunk("c1", "Cláusula primeira.", ["p1"])],
    )

    assert isinstance(cut[0], CutPiece)
    assert cut[0].text == "Cláusula primeira."
    assert cut[0].headings == ("CONTRATO",)
    # The join back to the reader's identifiers (ASM-023) is kept beside the
    # blocks rather than derived from them: one block may stand for several ids.
    assert cut[0].source_block_ids == ("p1",)
    assert cut[0].blocks == (index["p1"],)
    assert cut[0].chunk_id == "c1"


def test_a_pieces_pages_come_off_the_blocks_it_points_at():
    # The difference pointing buys. A piece that holds a page number can
    # disagree with the document; one that reads it off the blocks cannot.
    _, cut = _cut(
        [
            text_block("b1", "início", pages=(4, 4)),
            text_block("b2", "meio", pages=(5, 6)),
        ],
        [reader_chunk("c1", "início meio", ["b1", "b2"])],
    )

    assert len(cut[0].blocks) == 2
    assert cut[0].pages == (4, 6)


def test_a_piece_naming_a_table_and_one_of_its_cells_points_at_the_table_once():
    # Both identifiers resolve to the same block, and a block counted twice
    # would be counted twice by anything reading the piece's provenance.
    table = table_block("t1", ["Parcela"], [["1"]], pages=(2, 2))
    cell_id = table["tableBlock"]["bodyRows"][0]["cells"][0]["blocks"][0]["blockId"]
    index, cut = _cut([table], [reader_chunk("c1", "| 1 |", ["t1", cell_id])])

    assert cut[0].blocks == (index["t1"], index["t1"])
    assert cut[0].source_block_ids == ("t1", cell_id)
    # Which is one kind, once, however many identifiers named it.
    assert cut[0].kinds == (BLOCK_TABLE,)


def test_a_piece_records_every_kind_it_spans_in_the_readers_own_words():
    # AC-232: a piece covering a heading and the table beneath it is both, and
    # each name is the string the reader itself used.
    table = table_block("t1", ["Parcela"], [["1"]], pages=(1, 1))
    _, cut = _cut(
        [text_block("h1", "CLÁUSULA QUINTA", pages=(1, 1), kind="heading-1"), table],
        [reader_chunk("c1", "CLÁUSULA QUINTA | 1 |", ["h1", "t1"])],
    )

    assert cut[0].kinds == (BLOCK_TEXT, "heading-1", BLOCK_TABLE)


def test_a_list_reports_the_pages_of_the_lines_that_are_on_it():
    index, cut = _cut(
        [list_block("l1", ["primeiro item", "segundo item"], pages=(3, 3))],
        [reader_chunk("c1", "1. primeiro item\n2. segundo item", ["l1"])],
    )

    assert cut[0].pages == (3, 3)
    assert cut[0].kinds == (BLOCK_LIST, "ordered")
    assert index["l1"].text == "1. primeiro item\n2. segundo item"


def test_a_piece_becomes_the_record_the_reader_half_of_which_is_its_own():
    # T-215's seam: the base carries what any module has, the variant carries
    # what THIS reader also gave, and the stored row is the same either way.
    mapped = _map(
        [text_block("p1", "Cláusula primeira.", pages=(1, 1))],
        [reader_chunk("c1", "Cláusula primeira.", ["p1"])],
    )
    [piece] = mapped.chunks

    assert isinstance(piece, ReaderCutChunk)
    assert piece.chunk_id == "c1"
    assert piece.source_block_ids == ("p1",)
    assert piece.to_piece([0.0]).metadata["source_block_ids"] == ["p1"]
    assert piece.to_piece([0.0]).metadata["block_kinds"] == [BLOCK_TEXT, "paragraph"]
