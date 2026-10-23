from __future__ import annotations

import pytest

from src.responses import (
    BRIEF_CUT_MARK,
    PIECE_SEPARATOR,
    STATUS_FAILED,
    STATUS_FILTER_MATCHED_NOTHING,
    STATUS_FOUND,
    STATUS_NOTHING_MATCHED,
    document_group_payload,
    document_of,
    document_payload,
    documents_search_payload,
    empty_payload,
    failure_payload,
    filter_echo,
    grouped_results_payload,
    grouped_search_payload,
    not_found_payload,
    passage_payload,
    piece_of,
    result_id,
    shorten_to_sentence,
)
from src.search_events import SEARCH_OUTCOMES
from src.search_result import (
    Failed,
    FilterMatchedNothing,
    Found,
    FusionWeights,
    NothingMatched,
    Provenance,
    SearchDiagnostics,
    SearchHit,
)


def _hit(chunk_id: int = 11, external_id: str = "drive-abc") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content="Cláusula décima: a rescisão exige aviso prévio.",
        provenance=Provenance(
            document_id=3,
            external_id=external_id,
            title="Contrato de locação",
            page_start=4,
            page_end=5,
            heading_path=("Da rescisão",),
        ),
        score=0.031,
        found_by=("dense", "lexical"),
    )


def _diagnostics(
    documents_considered: int | None = None, pieces_considered: int | None = None
) -> SearchDiagnostics:
    return SearchDiagnostics(
        index_search_effort=40,
        weights=FusionWeights(),
        candidate_depth=50,
        result_count=10,
        documents_considered=documents_considered,
        pieces_considered=pieces_considered,
    )


def test_the_three_statuses_are_the_search_records_own_vocabulary():
    assert {STATUS_FOUND, STATUS_NOTHING_MATCHED, STATUS_FAILED} == set(SEARCH_OUTCOMES)


def test_a_failure_carries_no_result_list_at_all():
    failure = failure_payload("the archive could not be reached")

    assert failure["status"] == STATUS_FAILED
    assert failure["error"]
    assert "results" not in failure


def test_an_explicit_empty_carries_a_present_and_empty_list():
    empty = empty_payload()

    assert empty["status"] == STATUS_NOTHING_MATCHED
    assert empty["results"] == []
    assert "error" not in empty


def test_a_hit_carries_every_part_of_its_provenance_and_not_only_the_citation():
    group = document_group_payload((_hit(),))[0]
    payload = group["passages"][0]

    assert group["document"] == "drive-abc"
    assert group["title"] == "Contrato de locação"
    assert payload["page_start"] == 4
    assert payload["page_end"] == 5
    assert payload["heading_path"] == ["Da rescisão"]
    assert payload["citation"] == "Contrato de locação, Da rescisão, pp. 4-5"
    assert payload["found_by"] == ["dense", "lexical"]
    assert payload["truncated"] is False
    assert payload["full_length"] == len(payload["text"])


def test_a_brief_passage_never_exceeds_its_limit_when_no_sentence_fits():
    text = "uma cláusula sem pontuação intermediária " * 20

    shortened = shorten_to_sentence(text, 80)

    assert len(shortened) <= 80
    assert shortened.endswith(BRIEF_CUT_MARK)
    assert text.startswith(shortened[: -len(BRIEF_CUT_MARK)])


def test_a_brief_passage_prefers_the_last_complete_sentence_that_fits():
    text = "Primeira frase completa. Segunda frase que não cabe inteira no limite escolhido."

    shortened = shorten_to_sentence(text, 45)

    assert shortened == "Primeira frase completa." + BRIEF_CUT_MARK


def test_a_hit_with_no_title_falls_back_to_the_identifier_it_does_have():
    untitled = SearchHit(
        chunk_id=1,
        content="texto",
        provenance=Provenance(document_id=1, external_id="drive-sem-titulo", page_start=1),
        score=0.1,
        found_by=("dense",),
    )

    assert document_group_payload((untitled,))[0]["title"] == "drive-sem-titulo"


def test_a_result_identifier_names_its_document_and_its_piece():
    identifier = result_id(7, 42)

    assert identifier == f"7{PIECE_SEPARATOR}42"
    assert document_of(identifier) == "7"
    assert piece_of(identifier) == 42


def test_a_bare_document_identifier_passes_through_the_reader_unchanged():
    assert document_of("drive-abc") == "drive-abc"
    assert piece_of("drive-abc") is None


def test_an_identifier_whose_own_text_contains_the_separator_is_not_truncated():
    assert document_of("drive#abc") == "drive#abc"
    assert document_of("drive#abc#7") == "drive#abc"


def test_piece_of_does_not_read_a_pre_change_identifiers_suffix_as_an_ordinal():
    assert piece_of("drive-abc#123") is None


def test_search_payload_maps_each_outcome_to_its_own_shape():
    assert grouped_search_payload(Found(hits=(_hit(),), diagnostics=_diagnostics()))["status"] == STATUS_FOUND
    assert grouped_search_payload(NothingMatched(diagnostics=_diagnostics()))["results"] == []
    assert grouped_search_payload(Failed(reason="down"))["error"] == "down"


def test_search_payload_refuses_something_that_is_not_an_outcome():
    with pytest.raises(TypeError):
        grouped_search_payload(object())  # type: ignore[arg-type]


def test_results_payload_keeps_the_order_it_was_given():
    hits = (_hit(chunk_id=2), _hit(chunk_id=1))

    payload = grouped_results_payload(hits)

    assert [group["document"] for group in payload["results"]] == ["drive-abc"]
    passages = payload["results"][0]["passages"]
    assert [passage["rank"] for passage in passages] == [1, 2]
    assert [passage["id"] for passage in passages] == ["2", "1"]


def test_a_document_reports_the_whole_count_and_not_the_windows():
    payload = document_payload(
        external_id="drive-abc",
        title="Contrato",
        piece_count=184,
        offset=0,
        pieces=[{"ordinal": 0, "text": "..."}],
    )

    assert payload["piece_count"] == 184
    assert len(payload["pieces"]) == 1
    assert payload["offset"] == 0


def test_a_document_the_archive_does_not_hold_is_an_empty_and_not_a_failure():
    absent = not_found_payload("drive-que-nao-existe")

    assert absent["status"] == STATUS_NOTHING_MATCHED
    assert absent["pieces"] == []
    assert "error" not in absent


def _grouped_hit(
    *,
    document_id: int,
    ordinal: int,
    chunk_id: int,
    external_id: str = "drive-abc",
    title: str = "Contrato de locação",
    link: str | None = "https://drive.google.com/file/d/drive-abc/view",
    location: tuple[str, ...] = ("folder-a",),
    content: str = "texto da cláusula",
    score: float = 0.5,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        content=content,
        provenance=Provenance(
            document_id=document_id,
            external_id=external_id,
            title=title,
            page_start=1,
            page_end=1,
            heading_path=("Da rescisão",),
            ordinal=ordinal,
            link=link,
            location=location,
        ),
        score=score,
        found_by=("dense",),
    )


def test_a_document_answering_several_times_is_named_once():
    hits = (
        _grouped_hit(document_id=3, ordinal=0, chunk_id=10, score=0.9),
        _grouped_hit(document_id=3, ordinal=1, chunk_id=11, score=0.7),
    )

    grouped = document_group_payload(hits)

    assert len(grouped) == 1
    assert grouped[0]["document"] == "drive-abc"
    assert grouped[0]["title"] == "Contrato de locação"
    assert grouped[0]["link"] == "https://drive.google.com/file/d/drive-abc/view"
    assert len(grouped[0]["passages"]) == 2
    for passage in grouped[0]["passages"]:
        assert "title" not in passage
        assert "link" not in passage
        assert "location" not in passage


def test_each_passage_states_its_own_rank_and_the_document_holds_the_first():
    hits = (
        _grouped_hit(document_id=3, ordinal=0, chunk_id=10, score=0.9),
        _grouped_hit(document_id=4, ordinal=0, chunk_id=20, external_id="drive-xyz", score=0.8),
        _grouped_hit(document_id=3, ordinal=1, chunk_id=11, score=0.5),
    )

    grouped = document_group_payload(hits)

    assert [group["document"] for group in grouped] == ["drive-abc", "drive-xyz"]

    first_document_ranks = [passage["rank"] for passage in grouped[0]["passages"]]
    second_document_ranks = [passage["rank"] for passage in grouped[1]["passages"]]
    assert first_document_ranks == [1, 3]
    assert second_document_ranks == [2]


def test_a_passages_identifier_is_built_from_the_document_and_its_ordinal():
    hit = _grouped_hit(document_id=3, ordinal=7, chunk_id=99)

    grouped = document_group_payload((hit,))

    assert grouped[0]["passages"][0]["id"] == result_id(3, 7)


def test_grouped_search_payload_keeps_the_three_outcomes_untouched():
    assert grouped_search_payload(NothingMatched(diagnostics=_diagnostics()))["results"] == []
    assert grouped_search_payload(Failed(reason="down"))["error"] == "down"
    assert "results" not in grouped_search_payload(Failed(reason="down"))

    grouped = grouped_search_payload(
        Found(hits=(_grouped_hit(document_id=3, ordinal=0, chunk_id=10),), diagnostics=_diagnostics())
    )
    assert grouped["status"] == STATUS_FOUND
    assert grouped["results"][0]["document"] == "drive-abc"


def test_passage_payload_falls_back_to_the_chunk_id_when_unenriched():
    hit = _hit(chunk_id=42)

    payload = passage_payload(hit, rank=1)

    assert payload["id"] == "42"
    assert payload["rank"] == 1


def test_filter_echo_is_none_without_a_filter():
    assert filter_echo(_diagnostics()) is None
    assert filter_echo(None) is None


def test_filter_echo_reports_both_counts_when_the_diagnostics_carry_them():
    assert filter_echo(_diagnostics(documents_considered=1, pieces_considered=1)) == {
        "documents_considered": 1,
        "pieces_considered": 1,
    }


def test_a_filtered_found_answer_carries_the_echo_in_both_shapes():
    hits = (_grouped_hit(document_id=3, ordinal=0, chunk_id=10),)
    outcome = Found(hits=hits, diagnostics=_diagnostics(documents_considered=1, pieces_considered=1))

    grouped = grouped_search_payload(outcome)
    documents = documents_search_payload(outcome)

    assert grouped["filter"] == {"documents_considered": 1, "pieces_considered": 1}
    assert documents["filter"] == {"documents_considered": 1, "pieces_considered": 1}


def test_an_unfiltered_found_answer_carries_no_echo_in_either_shape():
    hits = (_grouped_hit(document_id=3, ordinal=0, chunk_id=10),)
    outcome = Found(hits=hits, diagnostics=_diagnostics())

    assert "filter" not in grouped_search_payload(outcome)
    assert "filter" not in documents_search_payload(outcome)


def test_a_filtered_nothing_matched_answer_carries_the_echo_in_both_shapes():
    outcome = NothingMatched(diagnostics=_diagnostics(documents_considered=3, pieces_considered=9))

    grouped = grouped_search_payload(outcome)
    documents = documents_search_payload(outcome)

    assert grouped["status"] == STATUS_NOTHING_MATCHED
    assert grouped["filter"] == {"documents_considered": 3, "pieces_considered": 9}
    assert documents["status"] == STATUS_NOTHING_MATCHED
    assert documents["filter"] == {"documents_considered": 3, "pieces_considered": 9}


def test_an_unfiltered_nothing_matched_answer_carries_no_echo():
    assert "filter" not in empty_payload(_diagnostics())
    assert "filter" not in grouped_search_payload(NothingMatched(diagnostics=_diagnostics()))


def test_a_filter_matched_nothing_outcome_carries_its_clause_and_the_echo():
    outcome = FilterMatchedNothing(
        diagnostics=_diagnostics(documents_considered=0, pieces_considered=0),
        clause="document_id ausente",
    )

    grouped = grouped_search_payload(outcome)
    documents = documents_search_payload(outcome)

    for payload in (grouped, documents):
        assert payload["status"] == STATUS_FILTER_MATCHED_NOTHING
        assert payload["status"] != STATUS_NOTHING_MATCHED
        assert payload["clause"] == "document_id ausente"
        assert payload["results"] == []
        assert payload["filter"] == {"documents_considered": 0, "pieces_considered": 0}


def test_a_filtered_failure_carries_the_echo_when_diagnostics_reached_it():
    unfiltered = failure_payload("down")
    filtered = failure_payload("down", _diagnostics(documents_considered=2, pieces_considered=4))

    assert "filter" not in unfiltered
    assert filtered["filter"] == {"documents_considered": 2, "pieces_considered": 4}
