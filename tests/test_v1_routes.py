
from __future__ import annotations

import pytest
from conftest import FakeQueryEmbedder

from src.responses import STATUS_FAILED, STATUS_FOUND, STATUS_NOTHING_MATCHED


@pytest.fixture
def client(api_world, served_corpus):
    built, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))
    return built


@pytest.fixture
def blind(api_world, served_corpus):
    built, _ = api_world(RAG_DB_URL="postgresql://127.0.0.1:1/nowhere")
    return built


def test_a_search_that_found_something_is_a_200_carrying_the_results(client):
    response = client.post("/v1/search", json={"question": "rescisão antecipada"})

    assert response.status_code == 200
    assert response.json()["status"] == STATUS_FOUND
    assert response.json()["results"]


def test_sync_routes_are_in_openapi_and_the_generated_collection(api_world):
    import json
    from scripts.postman_collection import build

    client, service = api_world()
    schema = client.app.openapi()
    assert "202" in schema["paths"]["/v1/sync"]["post"]["responses"]
    assert "409" in schema["paths"]["/v1/sync"]["post"]["responses"]
    assert "get" in schema["paths"]["/v1/sync/{sync_id}"]
    collection = build(service.config, client.app)
    requests = [item["request"] for folder in collection["item"] for item in folder["item"]]
    start = next(item for item in requests if item["url"]["raw"] == "{{baseUrl}}/v1/sync")
    assert start["method"] == "POST" and json.loads(start["body"]["raw"]) == {}
    assert any(item["url"]["raw"] == "{{baseUrl}}/v1/sync/{{sync_id}}" for item in requests)
    assert any(variable["key"] == "sync_id" for variable in collection["variable"])


def test_a_search_that_matched_nothing_is_also_a_200(api_world, served_corpus):
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=200))

    response = client.post("/v1/search", json={"question": "xilografia bizantina"})

    assert response.status_code == 200
    assert response.json()["status"] == STATUS_NOTHING_MATCHED
    assert response.json()["results"] == []


def test_a_search_that_could_not_run_is_a_502_and_carries_no_results(blind):
    response = blind.post("/v1/search", json={"question": "rescisão"})

    assert response.status_code == 502
    assert response.json()["status"] == STATUS_FAILED
    assert "results" not in response.json()


def test_a_failure_body_is_the_shaped_payload_and_not_a_framework_envelope(blind):
    body = blind.post("/v1/search", json={"question": "rescisão"}).json()

    assert set(body) == {"status", "error", "request_id"}
    assert "detail" not in body


def test_a_misspelled_knob_is_refused_rather_than_ignored(client):
    response = client.post(
        "/v1/search", json={"question": "rescisão", "result_counts": 5}
    )

    assert response.status_code == 422


def test_a_request_with_no_question_never_reaches_the_archive(client):
    assert client.post("/v1/search", json={}).status_code == 422
    assert client.post("/v1/search", json={"question": ""}).status_code == 422


def test_a_document_route_returns_the_pieces_in_order_with_the_whole_count(client):
    response = client.get("/v1/documents/drive-rescisao")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == STATUS_FOUND
    assert body["piece_count"] == 2
    assert [piece["ordinal"] for piece in body["pieces"]] == [0, 1]


def test_a_document_route_windows_on_request(client):
    body = client.get("/v1/documents/drive-rescisao", params={"offset": 1, "limit": 1}).json()

    assert body["offset"] == 1
    assert [piece["ordinal"] for piece in body["pieces"]] == [1]
    assert body["piece_count"] == 2


def test_a_window_larger_than_the_deployment_permits_is_refused_by_name(
    api_world, served_corpus
):
    client, _ = api_world(RAG_API_MAX_DOCUMENT_PIECES="5")

    response = client.get("/v1/documents/drive-rescisao", params={"limit": 6})

    assert response.status_code == 422
    assert "api.limits.fetch.limit" in response.json()["error"]


def test_a_document_the_archive_does_not_hold_is_a_200_with_no_pieces(client):
    response = client.get("/v1/documents/drive-nao-existe")

    assert response.status_code == 200
    assert response.json()["status"] == STATUS_NOTHING_MATCHED
    assert response.json()["pieces"] == []


def test_a_document_route_over_an_unreachable_archive_is_a_502(blind):
    response = blind.get("/v1/documents/drive-rescisao")

    assert response.status_code == 502
    assert response.json()["status"] == STATUS_FAILED


def test_a_result_identifier_works_on_the_document_route_too(client):
    first = client.post("/v1/search", json={"question": "rescisão antecipada"}).json()
    identifier = first["results"][0]["passages"][0]["id"]

    response = client.get(f"/v1/documents/{identifier}")

    assert response.status_code == 200
    assert response.json()["id"] == "drive-rescisao"


def test_a_caller_that_names_itself_is_recorded_under_that_name(client, schema_dsn):
    import psycopg

    client.post("/v1/search", json={"question": "rescisão", "caller": "example-caller"})

    with psycopg.connect(schema_dsn) as check, check.cursor() as cursor:
        cursor.execute("SELECT caller FROM search_events ORDER BY id DESC LIMIT 1")
        (caller,) = cursor.fetchone()
    assert caller == "example-caller"


def test_a_search_asked_briefly_returns_shorter_passages(api_world, served_corpus):
    client, _ = api_world(RAG_SEARCH_BRIEF_LENGTH=55)
    full = client.post("/v1/search", json={"question": "rescisão antecipada"}).json()
    brief = client.post(
        "/v1/search", json={"question": "rescisão antecipada", "brief": True}
    ).json()

    full_text = full["results"][0]["passages"][0]["text"]
    brief_passage = brief["results"][0]["passages"][0]
    assert len(brief_passage["text"]) < len(full_text)
    assert len(brief_passage["text"]) <= 55
    assert brief_passage["truncated"] is True
    assert brief_passage["full_length"] == len(full_text)


def test_a_document_route_reads_by_page_range(client):
    response = client.get(
        "/v1/documents/drive-rescisao", params={"page_start": 4, "page_end": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert [piece["ordinal"] for piece in body["pieces"]] == [0]


def test_a_page_range_on_a_document_with_no_pagination_is_refused_by_reason(
    api_world, schema_dsn
):
    import psycopg
    from conftest import ChunkSpec, DocumentSpec, TEST_STAMP, index_corpus
    from src.store_versions import activate_version

    client, _service = api_world()
    try:
        with psycopg.connect(schema_dsn) as conn:
            index_corpus(
                conn,
                (
                    DocumentSpec(
                        "drive-sem-paginas",
                        "Termo de aditamento — sem paginação",
                        (ChunkSpec(0, "Cláusula única.", degrees=0, headings=("Da ratificação",)),),
                    ),
                ),
            )
            activate_version(conn, TEST_STAMP)
            conn.commit()

        response = client.get(
            "/v1/documents/drive-sem-paginas", params={"page_start": 1, "page_end": 1}
        )

        assert response.status_code == 422
        assert "no pagination" in response.json()["error"].lower()
    finally:
        with psycopg.connect(schema_dsn) as cleanup, cleanup.cursor() as cursor:
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents WHERE external_id = %s", ("drive-sem-paginas",))
            cursor.execute("DELETE FROM active_model_version")
            cleanup.commit()


def test_an_outline_route_states_what_it_was_built_from(client):
    response = client.get("/v1/documents/drive-rescisao/outline")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == STATUS_FOUND
    assert body["basis"] == "headings"
    assert body["entries"]


def test_an_outline_of_a_document_the_archive_does_not_hold_is_nothing_matched(client):
    response = client.get("/v1/documents/drive-nao-existe/outline")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == STATUS_NOTHING_MATCHED
    assert body["entries"] == []


def test_an_outline_route_over_an_unreachable_archive_is_a_502(blind):
    response = blind.get("/v1/documents/drive-rescisao/outline")

    assert response.status_code == 502
    assert response.json()["status"] == STATUS_FAILED
