from __future__ import annotations

import pytest

from src.search_result import Provenance


def test_a_piece_that_cannot_name_its_document_is_refused() -> None:
    with pytest.raises(ValueError, match="name the document"):
        Provenance(document_id=1, external_id="", page_start=1)


def test_a_piece_with_no_locator_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no locator"):
        Provenance(document_id=1, external_id="drive-1")


def test_a_heading_path_alone_is_locator_enough() -> None:
    provenance = Provenance(document_id=1, external_id="drive-1", heading_path=("Contrato", "Rescisão"))
    assert provenance.page_label() == ""
    assert "Rescisão" in provenance.citation()


def test_a_page_range_alone_is_locator_enough() -> None:
    provenance = Provenance(document_id=1, external_id="drive-1", page_start=4, page_end=7)
    assert provenance.page_label() == "pp. 4-7"


def test_a_single_page_reads_as_one_page_not_a_range() -> None:
    assert Provenance(document_id=1, external_id="d", page_start=4, page_end=4).page_label() == "p. 4"
    assert Provenance(document_id=1, external_id="d", page_start=4).page_label() == "p. 4"
    assert Provenance(document_id=1, external_id="d", page_end=4).page_label() == "p. 4"


def test_a_page_range_that_runs_backwards_is_refused() -> None:
    with pytest.raises(ValueError, match="backwards"):
        Provenance(document_id=1, external_id="d", page_start=7, page_end=4)


def test_the_citation_names_the_document_then_the_location_inside_it() -> None:
    provenance = Provenance(
        document_id=1,
        external_id="drive-1",
        title="Contrato de locação — Sala 402",
        page_start=4,
        page_end=5,
        heading_path=("Cláusula Quinta", "Rescisão antecipada"),
    )
    assert provenance.citation() == (
        "Contrato de locação — Sala 402, Cláusula Quinta › Rescisão antecipada, pp. 4-5"
    )


def test_a_document_with_no_title_is_cited_by_its_identifier() -> None:
    provenance = Provenance(document_id=1, external_id="drive-99", page_start=1)
    assert provenance.citation().startswith("drive-99")


def test_ordinal_link_and_location_are_absent_until_something_fills_them() -> None:
    provenance = Provenance(document_id=1, external_id="drive-1", page_start=1)
    assert provenance.ordinal is None
    assert provenance.link is None
    assert provenance.location == ()
