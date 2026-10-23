"""Unit-level coverage for src/filters.py that is not itself an acceptance criterion.

AC-278, AC-279, AC-284 and AC-285 are proven in tests/test_spec_search_filters.py — see
CLAUDE.md for why the proof lives there. This file covers the vocabulary table itself,
every operator family compilation produces, the two semantics design.md calls out by name
(page overlap, path-of-segments prefix), and the edge cases the acceptance criteria do not
individually name.
"""

from __future__ import annotations

import pytest

from src.filters import (
    COMBINATORS,
    DEFAULT_MAX_FILTER_CLAUSES,
    DEFAULT_MAX_FILTER_DEPTH,
    DEFAULT_MAX_PATTERN_LENGTH,
    FIELDS,
    OPERATORS,
    Clause,
    Combinator,
    FilterError,
    compile_filter,
    validate_filter,
)


# --------------------------------------------------------------------------
# The vocabulary itself
# --------------------------------------------------------------------------

def test_the_vocabulary_is_exactly_fourteen_fields():
    assert len(FIELDS) == 14


def test_every_field_type_has_an_entry_in_operators():
    for spec in FIELDS.values():
        assert spec.type in OPERATORS


def test_media_type_and_process_module_are_closed_enums_with_real_values():
    assert FIELDS["media_type"].values
    assert "application/pdf" in FIELDS["media_type"].values
    assert FIELDS["process_module"].values
    assert "pdftext" in FIELDS["process_module"].values
    assert "docai" in FIELDS["process_module"].values


def test_block_kind_is_an_open_enum_with_no_closed_values():
    assert FIELDS["block_kind"].type == "enum_open"
    assert FIELDS["block_kind"].values is None


# --------------------------------------------------------------------------
# validate_filter: absence, and the happy path
# --------------------------------------------------------------------------

def test_no_filter_validates_to_none():
    assert validate_filter(None) is None


def test_a_bare_leaf_clause_validates_without_a_combinator():
    node = validate_filter({"field": "title", "op": "is", "value": "Aditivo"})
    assert isinstance(node, Clause)
    assert node.field == "title"


def test_a_nested_filter_validates_into_a_combinator_tree():
    node = validate_filter(
        {
            "all": [
                {"field": "title", "op": "glob", "value": "*aditivo*"},
                {"field": "page", "op": "between", "value": [30, 50]},
                {
                    "any": [
                        {"field": "media_type", "op": "is", "value": "application/pdf"},
                        {"field": "process_module", "op": "is", "value": "pdftext"},
                    ]
                },
            ]
        }
    )
    assert isinstance(node, Combinator)
    assert node.kind == "all"
    assert len(node.children) == 3
    assert isinstance(node.children[2], Combinator)
    assert node.children[2].kind == "any"


# --------------------------------------------------------------------------
# AC-278 / AC-279 edge cases beyond the one spec test each gets
# --------------------------------------------------------------------------

def test_unknown_field_names_the_field_and_the_whole_vocabulary():
    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "author", "op": "is", "value": "x"})
    assert "author" in excinfo.value.reason
    assert "document_id" in excinfo.value.reason  # the vocabulary is named too


def test_operator_a_type_does_not_take_names_field_type_and_accepted_operators():
    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "document_id", "op": "gt", "value": 5})
    assert "document_id" in excinfo.value.reason
    assert "text" in excinfo.value.reason
    assert "is" in excinfo.value.reason


@pytest.mark.parametrize(
    ("field", "op", "value"),
    [
        ("size_bytes", "is", "ten"),  # number field, text value
        ("modified_at", "gt", "not-a-date"),  # date field, unparsable value
        ("folder", "has_any", "not-a-list"),  # text_list field, scalar value
        ("page", "between", [30, "fifty"]),  # range field, mixed pair
        ("process_module", "is", "not-a-real-module"),  # closed enum, unknown value
    ],
)
def test_a_value_whose_type_or_membership_does_not_match_the_field_is_refused(field, op, value):
    with pytest.raises(FilterError):
        validate_filter({"field": field, "op": op, "value": value})


# --------------------------------------------------------------------------
# AC-284 edge cases: every way a shape can be wrong
# --------------------------------------------------------------------------

def test_a_combinator_mixed_with_other_keys_is_refused():
    with pytest.raises(FilterError):
        validate_filter({"all": [{"field": "title", "op": "is", "value": "x"}], "field": "title"})


def test_an_empty_combinator_is_refused():
    with pytest.raises(FilterError):
        validate_filter({"all": []})


def test_a_clause_with_an_unknown_key_is_refused():
    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "title", "op": "is", "value": "x", "weight": 2})
    assert "weight" in excinfo.value.reason


def test_the_default_max_clauses_matches_design():
    assert DEFAULT_MAX_FILTER_CLAUSES == 16


def test_the_default_max_depth_matches_design():
    assert DEFAULT_MAX_FILTER_DEPTH == 3


def test_the_default_max_pattern_length_matches_design():
    assert DEFAULT_MAX_PATTERN_LENGTH == 200


def test_nesting_within_the_limit_is_accepted():
    # depth 1 (all) -> depth 2 (any) -> depth 3 (none), still within max_depth=3.
    node = {
        "all": [
            {
                "any": [
                    {"none": [{"field": "title", "op": "is", "value": "x"}]},
                ]
            }
        ]
    }
    validate_filter(node, max_depth=3)


def test_the_offending_clause_position_is_named_in_a_nested_filter():
    node = {
        "all": [
            {"field": "title", "op": "is", "value": "x"},
            {"any": [{"field": "bogus", "op": "is", "value": "x"}]},
        ]
    }
    with pytest.raises(FilterError) as excinfo:
        validate_filter(node)
    assert excinfo.value.path == "<filter>.all[1].any[0]"


# --------------------------------------------------------------------------
# AC-285 edge cases: only glob/regex are patterns, and only the length counts
# --------------------------------------------------------------------------

def test_contains_and_starts_with_are_not_bounded_by_max_pattern_length():
    long_value = "a" * 500
    validate_filter({"field": "title", "op": "contains", "value": long_value}, max_pattern_length=200)
    validate_filter({"field": "title", "op": "starts_with", "value": long_value}, max_pattern_length=200)


def test_a_pattern_at_exactly_the_limit_is_accepted():
    value = "a" * 200
    validate_filter({"field": "content", "op": "regex", "value": value}, max_pattern_length=200)


def test_a_pattern_one_over_the_limit_names_the_limit():
    value = "a" * 201
    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "content", "op": "regex", "value": value}, max_pattern_length=200)
    assert "200" in excinfo.value.reason
    assert "max_pattern_length" in excinfo.value.reason


# --------------------------------------------------------------------------
# compile_filter: no interpolation, and the two decided semantics
# --------------------------------------------------------------------------

def test_compile_filter_of_none_is_none():
    assert compile_filter(None) is None


def test_compile_filter_never_interpolates_a_value_into_the_fragment():
    node = validate_filter({"field": "title", "op": "is", "value": "'; DROP TABLE documents; --"})
    sql, params = compile_filter(node)
    assert "DROP TABLE" not in sql
    assert params == ["'; DROP TABLE documents; --"]
    assert "%s" in sql


def test_page_between_is_overlap_not_containment():
    node = validate_filter({"field": "page", "op": "between", "value": [30, 50]})
    sql, params = compile_filter(node)
    assert "page_end >= %s" in sql
    assert "page_start <= %s" in sql
    assert params == [30, 50]


def test_path_starts_with_is_a_prefix_of_path_segments_not_of_the_string():
    node = validate_filter({"field": "heading_path", "op": "path_starts_with", "value": ["Cláusula 5"]})
    sql, params = compile_filter(node)
    assert "heading_path[1:%s]" in sql
    assert params == [1, ["Cláusula 5"]]


def test_joined_fields_compile_to_a_self_contained_exists():
    node = validate_filter({"field": "process_module", "op": "is", "value": "pdftext"})
    sql, params = compile_filter(node)
    assert "EXISTS" in sql
    assert "document_ingest" in sql
    assert "di.document_id = c.document_id" in sql
    assert params == ["pdftext"]

    node = validate_filter({"field": "reader_version", "op": "is", "value": "docai-layout-parser"})
    sql, params = compile_filter(node)
    assert "document_reads" in sql
    assert params == ["docai-layout-parser"]


def test_block_kind_compiles_to_a_jsonb_membership_test():
    node = validate_filter({"field": "block_kind", "op": "is", "value": "tableBlock"})
    sql, params = compile_filter(node)
    assert "block_kinds" in sql
    assert params == ["tableBlock"]


def test_glob_is_translated_to_an_anchored_regex_never_left_as_a_glob():
    node = validate_filter({"field": "title", "op": "glob", "value": "*aditivo*"})
    sql, params = compile_filter(node)
    assert params == ["^.*aditivo.*$"]
    assert sql == "d.title ~ %s"


def test_all_any_none_combine_children_with_the_right_boolean():
    node = validate_filter(
        {
            "all": [
                {"field": "title", "op": "is", "value": "a"},
                {"any": [
                    {"field": "media_type", "op": "is", "value": "application/pdf"},
                    {"field": "process_module", "op": "is", "value": "pdftext"},
                ]},
            ]
        }
    )
    sql, params = compile_filter(node)
    assert " AND " in sql
    assert " OR " in sql
    assert params == ["a", "application/pdf", "pdftext"]


def test_none_negates_the_disjunction_of_its_children():
    node = validate_filter({"none": [{"field": "media_type", "op": "is", "value": "application/pdf"}]})
    sql, params = compile_filter(node)
    assert sql.startswith("NOT (")
    assert params == ["application/pdf"]
