from __future__ import annotations

import anyio
import pytest

from src.mcp import build_server
from src.responses import STATUS_FOUND, STATUS_NOTHING_MATCHED
from src.usecases.fetch import Outline as OutlineResult
from src.usecases.fetch import fetch_on, outline_on
from tests.conftest import (
    API_CORPUS,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    FakeQueryEmbedder,
    index_corpus,
)
from src.store_versions import activate_version


PAGED = DocumentSpec(
    "drive-edital-longo",
    "Edital longo",
    tuple(
        ChunkSpec(n, f"Cláusula {n + 1}: da rescisão e do reajuste contratual.", degrees=0, pages=(n + 1, n + 1))
        for n in range(10)
    ),
)


def _passage_ids(payload) -> list[str]:
    return [passage["id"] for document in payload["results"] for passage in document["passages"]]


def test_ac_298(api_world, served_corpus):
    """AC-298: A search answers from the position the caller names @spec:AC-298"""
    client, _service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    first = client.post("/v1/search", json={"question": "rescisão", "limit": 2}).json()
    assert first["status"] == STATUS_FOUND
    assert first["more"]["candidates"] > 0, "the corpus must overflow the window"

    later = client.post("/v1/search", json={"question": "rescisão", "limit": 2, "offset": 1}).json()

    assert _passage_ids(later)[0] == _passage_ids(first)[1]
    assert _passage_ids(first)[0] not in _passage_ids(later)

    whole = client.post("/v1/search", json={"question": "rescisão", "limit": 6}).json()
    assert _passage_ids(later) == _passage_ids(whole)[1:3]


def test_ac_299(api_world, served_corpus):
    """AC-299: A search says where its list ends, instead of deepening itself to hide it @spec:AC-299"""
    client, _service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    counted = client.post("/v1/search", json={"question": "rescisão", "limit": 1}).json()
    past_the_end = counted["more"]["candidates"] + 50

    answer = client.post(
        "/v1/search", json={"question": "rescisão", "limit": 1, "offset": past_the_end}
    )

    payload = answer.json()
    note = payload["more"]["note"].lower()
    assert "candidate_depth" in note
    assert "no results after this position" in note

    assert answer.status_code == 200
    assert payload["status"] == STATUS_NOTHING_MATCHED
    assert payload["results"] == []
    assert "error" not in payload


def test_ac_300(api_world, served_corpus):
    """AC-300: The assistant's page is the deployment's, and its position is its own @spec:AC-300"""
    _client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)
    tools = {tool.name: tool for tool in anyio.run(server.list_tools)}

    answer = anyio.run(lambda: server.call_tool("search", {"query": "rescisão"}))
    payload = answer.structured_content

    returned = sum(len(document["passages"]) for document in payload["results"])
    assert returned <= service.config.mcp.defaults.search.limit
    arguments = set(tools["search"].input_schema["properties"])
    assert not arguments & {"max_results", "limit", "result_count", "count"}

    assert "offset" in arguments
    if "more" in payload:
        assert payload["more"]["next_offset"] == returned


def test_ac_301(conn):
    """AC-301: An outline never spends an entry on what a piece's own position already says @spec:AC-301"""
    index_corpus(conn, (PAGED,))
    activate_version(conn, TEST_STAMP)

    found = outline_on(conn, "drive-edital-longo")

    assert isinstance(found, OutlineResult)
    assert found.basis == "pages"
    assert len(found.entries) == 1
    assert found.entries[0].ordinal == 0
    assert found.entries[0].ordinal_end == 9
    assert found.entries[0].label == "pages 1–10"

    gapped = DocumentSpec(
        "drive-com-branco",
        "Edital com página em branco",
        (
            ChunkSpec(0, "Primeira página.", degrees=0, pages=(1, 1)),
            ChunkSpec(1, "Terceira página — a segunda não tinha texto.", degrees=0, pages=(3, 3)),
        ),
    )
    index_corpus(conn, (gapped,))
    broken = outline_on(conn, "drive-com-branco")
    assert [entry.label for entry in broken.entries] == ["page 1", "page 3"]

    index_corpus(conn, API_CORPUS)
    headed = outline_on(conn, "drive-rescisao")
    assert headed.basis == "headings"
    assert all(entry.ordinal_end is None for entry in headed.entries)


def test_ac_302(conn):
    """AC-302: An outline can be asked for a level, a branch, or the headings matching a clause @spec:AC-302"""
    from src.filters import validate_filter

    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    whole = outline_on(conn, "drive-rescisao")
    assert whole.basis == "headings"
    deepest = max(len(entry.label.split(" › ")) for entry in whole.entries)
    assert deepest > 1, "the fixture must carry a nested heading for this to mean anything"

    shallow = outline_on(conn, "drive-rescisao", depth=1)

    assert all(" › " not in entry.label for entry in shallow.entries)
    assert len(shallow.entries) < len(whole.entries)

    root = whole.entries[0].label.split(" › ")[0]
    branch = outline_on(conn, "drive-rescisao", under=(root,))

    assert branch.entries != ()
    assert all(entry.label.startswith(root) for entry in branch.entries)

    word = whole.entries[-1].label.split(" › ")[-1].split()[0]
    clause = validate_filter(
        {"all": [{"field": "heading_path", "op": "contains", "value": word}]},
        max_clauses=16,
        max_depth=3,
        max_pattern_length=200,
    )
    matching = outline_on(conn, "drive-rescisao", clause=clause)

    assert matching.entries != ()
    assert all(word.lower() in entry.label.lower() for entry in matching.entries)


def test_ac_303(conn):
    """AC-303: A narrowed outline says it was narrowed @spec:AC-303"""
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    whole = outline_on(conn, "drive-rescisao")
    narrowed = outline_on(conn, "drive-rescisao", depth=1)

    assert narrowed.narrowed_by == ("depth=1",)
    assert narrowed.entries_total == len(whole.entries)
    assert "narrowed_by" in narrowed.payload()
    assert narrowed.payload()["entries_total"] == len(whole.entries)

    assert len(narrowed.entries) < narrowed.entries_total

    assert whole.narrowed_by == ()
    assert "narrowed_by" not in whole.payload()


def test_ac_304(api_world, served_corpus):
    """AC-304: The heading vocabulary takes the text operators, and one clause serves both @spec:AC-304"""
    from src.filters import FIELDS, OPERATORS

    operators = OPERATORS[FIELDS["heading_path"].type]

    assert {"contains", "starts_with", "glob", "regex"} <= set(operators)
    assert {"path_starts_with", "has_any"} <= set(operators)

    client, _service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    published = client.get("/v1/filters").json()
    heading = next(field for field in published["fields"] if field["name"] == "heading_path")
    assert {"contains", "path_starts_with"} <= set(heading["operators"])

    written = {"all": [{"field": "heading_path", "op": "contains", "value": "conserva"}]}
    searched = client.post("/v1/search", json={"question": "conservação", "filter": written})
    assert searched.status_code == 200
    listed = client.request(
        "GET", "/v1/documents/drive-rescisao/outline", json={"filter": written}
    )
    assert listed.status_code == 200
    assert listed.json()["status"] == STATUS_FOUND


def test_ac_307(api_world, served_corpus):
    """AC-307: The assistant can ask for a document's map without fetching the document @spec:AC-307"""
    _client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)

    answer = anyio.run(lambda: server.call_tool("outline", {"id": "drive-rescisao"}))
    payload = answer.structured_content

    assert payload["status"] == STATUS_FOUND
    assert payload["entries"] != []
    assert "pieces" not in payload
    assert "text" not in payload
    serialized = str(payload)
    assert "Cláusula" not in serialized or all("text" not in entry for entry in payload["entries"])

    narrowed = anyio.run(
        lambda: server.call_tool(
            "outline",
            {"id": "drive-rescisao", "filter": {"all": [{"field": "heading_path", "op": "contains", "value": "conserva"}]}},
        )
    ).structured_content
    assert narrowed["status"] == STATUS_FOUND
    assert "filter" in narrowed["narrowed_by"]


def test_ac_305(api_world, served_corpus):
    """AC-305: A window above the surface's limit is served short and declared @spec:AC-305"""
    _client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)
    ceiling = service.config.mcp.limits.fetch.limit

    payload = anyio.run(
        lambda: server.call_tool("fetch", {"id": "drive-rescisao", "limit": ceiling + 5})
    ).structured_content

    assert payload["status"] == STATUS_FOUND
    assert len(payload["pieces"]) <= ceiling
    assert payload["reduced"]["limit"] == ceiling
    assert "mcp.limits.fetch.limit" in payload["reduced"]["setting"]

    assert payload["piece_count"] >= len(payload["pieces"])
    assert payload["offset"] == 0


def test_ac_306(api_world, served_corpus):
    """AC-306: An outline the caller did not ask for is not sent @spec:AC-306"""
    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)

    declined = client.get("/v1/documents/drive-rescisao", params={"outline": "false"}).json()
    tool_declined = anyio.run(
        lambda: server.call_tool("fetch", {"id": "drive-rescisao", "outline": False})
    ).structured_content

    assert declined["pieces"] != []
    assert "outline" not in declined
    assert "outline" not in tool_declined

    asked = client.get("/v1/documents/drive-rescisao").json()
    assert asked["outline"] != []
