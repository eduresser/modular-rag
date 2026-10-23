from __future__ import annotations

import anyio
import pytest
from conftest import FakeQueryEmbedder

from src.mcp import CALLER, build_server


@pytest.fixture
def server(api_world, served_corpus):
    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    return build_server(service)


def _tools(server) -> dict:
    return {tool.name: tool for tool in anyio.run(server.list_tools)}


def _call(server, name: str, **arguments):
    return anyio.run(lambda: server.call_tool(name, arguments)).structured_content


def test_the_search_tool_takes_a_question_a_position_and_the_filter(server):
    schema = _tools(server)["search"].input_schema

    assert set(schema["properties"]) == {"query", "offset", "filter"}
    assert schema["required"] == ["query"]


def test_the_fetch_tool_takes_the_identifier_a_result_carried(server):
    schema = _tools(server)["fetch"].input_schema

    assert set(schema["properties"]) == {"id", "offset", "limit", "outline"}
    assert schema["required"] == ["id"]


def test_the_filters_tool_takes_no_argument(server):
    schema = _tools(server)["filters"].input_schema

    assert not schema.get("properties")
    assert schema.get("required", []) == []


def test_the_filters_tool_answers_the_same_fields_the_route_publishes(server, api_world):
    _, service = api_world()
    from src.db import connect
    from src.usecases.vocabulary import published_vocabulary

    payload = _call(server, "filters")

    with connect(service.config.db.url) as conn:
        published = published_vocabulary(conn)

    assert payload["status"] == "found"
    assert [field["name"] for field in payload["fields"]] == [field.name for field in published]


def test_each_description_says_when_to_reach_for_it_and_names_the_link_between_them(server):
    tools = _tools(server)

    assert "fetch" in tools["search"].description
    assert "preview" in tools["search"].description
    assert "truncated" in tools["search"].description
    assert "search result" in tools["fetch"].description
    assert "piece_count" in tools["fetch"].description


def test_the_two_tools_answer_with_structured_content_and_not_only_text(server):
    answer = anyio.run(lambda: server.call_tool("search", {"query": "rescisão antecipada"}))

    assert answer.structured_content is not None
    assert answer.is_error is False


def test_the_tool_surface_names_itself_in_the_search_record(server, schema_dsn):
    import psycopg

    _call(server, "search", query="rescisão antecipada")

    with psycopg.connect(schema_dsn) as check, check.cursor() as cursor:
        cursor.execute("SELECT caller FROM search_events ORDER BY id DESC LIMIT 1")
        (caller,) = cursor.fetchone()
    assert caller == CALLER


def test_asking_from_a_later_position_returns_the_results_after_it(server):
    first = _call(server, "search", query="rescisão antecipada")
    later = _call(server, "search", query="rescisão antecipada", offset=1)

    first_ids = [p["id"] for doc in first["results"] for p in doc["passages"]]
    later_ids = [p["id"] for doc in later["results"] for p in doc["passages"]]
    assert first_ids[1:] == later_ids[: len(first_ids) - 1]


def test_a_fetch_window_continues_where_the_previous_one_stopped(server):
    first = _call(server, "fetch", id="drive-rescisao", offset=0)
    second = _call(server, "fetch", id="drive-rescisao", offset=1)

    assert first["piece_count"] == 2
    assert [piece["ordinal"] for piece in second["pieces"]] == [1]
    assert second["offset"] == 1


RETIRED_ARGUMENTS = ("max_results", "result_count", "max_result_count")


def test_no_description_sends_the_assistant_to_an_argument_that_is_gone(server):
    for name, tool in _tools(server).items():
        for retired in RETIRED_ARGUMENTS:
            assert retired not in tool.description, (
                f"the {name!r} tool's description names {retired!r}, which is not one of "
                f"its arguments ({sorted(tool.input_schema['properties'])})"
            )


def test_every_argument_a_tool_takes_is_named_in_its_own_description(server):
    for name, tool in _tools(server).items():
        for argument in tool.input_schema["properties"]:
            assert f"'{argument}'" in tool.description, (
                f"the {name!r} tool takes {argument!r} and its description never names it"
            )
