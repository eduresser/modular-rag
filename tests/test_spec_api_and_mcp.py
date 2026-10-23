from __future__ import annotations

import anyio
import psycopg
from conftest import ALLOWED_HOST, API_TOKEN, FakeQueryEmbedder

from src.embedding.client import EmbeddingError
from src.responses import STATUS_FAILED, STATUS_FOUND, STATUS_NOTHING_MATCHED


def _call(server, name: str, **arguments):
    return anyio.run(lambda: server.call_tool(name, arguments))


def passages_of(payload):
    return [passage for group in payload["results"] for passage in group["passages"]]


def test_ac_130(api_world, served_corpus):
    _, service = api_world()
    from src.mcp import build_server

    server = build_server(service)

    tools = anyio.run(server.list_tools)

    assert sorted(tool.name for tool in tools) == ["fetch", "filters", "outline", "search"]

    for tool in tools:
        assert tool.description and len(tool.description) > 40

    assert anyio.run(server.list_resources) == []
    assert anyio.run(server.list_prompts) == []


def test_ac_131(api_world, served_corpus):
    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    from src.mcp import build_server

    server = build_server(service)

    answer = _call(server, "search", query="rescisão antecipada")
    payload = answer.structured_content

    assert payload["status"] == STATUS_FOUND
    assert payload["results"]
    for group in payload["results"]:
        assert group["id"]
        assert group["title"]
        assert group["passages"]
        for passage in group["passages"]:
            assert passage["id"]
            assert passage["text"]
            assert passage["page_start"] is not None and passage["page_end"] is not None
            assert passage["citation"]
            assert not {"title", "url", "link", "location"} & set(passage)

    first = payload["results"][0]
    assert first["id"] == "drive-rescisao"
    assert first["title"] == "Contrato de locação — Sala 402"

    fetched = _call(server, "fetch", id=first["passages"][0]["id"]).structured_content
    assert fetched["status"] == STATUS_FOUND
    assert fetched["id"] == "drive-rescisao"
    assert fetched["piece_count"] == 2
    assert [piece["ordinal"] for piece in fetched["pieces"]] == [0, 1]
    assert fetched["pieces"][0]["page_start"] == 4
    assert fetched["pieces"][0]["heading_path"] == ["Da rescisão"]


def test_ac_132(api_world, served_corpus, monkeypatch):
    from src.mcp import build_server

    _, unreachable = api_world(RAG_DB_URL="postgresql://127.0.0.1:1/nowhere")
    _, rejected = api_world(
        embedder=FakeQueryEmbedder(
            fails_with=EmbeddingError("the embedding provider refused: 401 invalid api key")
        )
    )
    _, healthy = api_world()

    broke = _call(build_server(unreachable), "search", query="rescisão")
    assert broke.structured_content["status"] == STATUS_FAILED
    assert broke.structured_content["error"]
    assert "results" not in broke.structured_content

    refused = _call(build_server(rejected), "search", query="rescisão")
    assert refused.structured_content["status"] == STATUS_FAILED
    assert "401" in refused.structured_content["error"]

    malformed = _call(build_server(healthy), "search", query="rescisão", offset=-1)
    assert malformed.structured_content["status"] == STATUS_FAILED
    assert "-1" in malformed.structured_content["error"]

    def explode(*args, **kwargs):
        raise RuntimeError("something nobody wrote a branch for")

    monkeypatch.setattr("src.mcp.run_search", explode)
    unexpected = _call(build_server(healthy), "search", query="rescisão")
    assert unexpected.structured_content["status"] == STATUS_FAILED
    assert "RuntimeError" in unexpected.structured_content["error"]

    monkeypatch.undo()
    fetch_broke = _call(build_server(unreachable), "fetch", id="drive-rescisao")
    assert fetch_broke.structured_content["status"] == STATUS_FAILED
    fetch_malformed = _call(build_server(healthy), "fetch", id="drive-rescisao", offset=-1)
    assert fetch_malformed.structured_content["status"] == STATUS_FAILED


def test_ac_133(api_world, served_corpus):
    from src.mcp import build_server

    _, service = api_world(embedder=FakeQueryEmbedder(degrees=200))
    server = build_server(service)

    payload = _call(server, "search", query="xilografia bizantina").structured_content

    assert payload["status"] == STATUS_NOTHING_MATCHED
    assert payload["results"] == []
    assert "error" not in payload

    _, broken = api_world(RAG_DB_URL="postgresql://127.0.0.1:1/nowhere")
    failure = _call(build_server(broken), "search", query="xilografia bizantina")
    assert failure.structured_content["status"] != payload["status"]
    assert "results" in payload and "results" not in failure.structured_content

    absent = _call(server, "fetch", id="drive-que-nao-existe").structured_content
    assert absent["status"] == STATUS_NOTHING_MATCHED
    assert absent["pieces"] == []


def test_ac_134(api_world, served_corpus, schema_dsn):
    from conftest import FakeScorer

    scorer = FakeScorer(
        "cross-encoder/mmarco-mMiniLMv2",
        {
            "Cláusula décima primeira: a multa por rescisão corresponde a três aluguéis.": 9.0,
            "Cláusula décima: a rescisão antecipada exige aviso prévio de trinta dias.": 1.0,
        },
    )
    client, _ = api_world(
        embedder=FakeQueryEmbedder(degrees=0),
        scorer=scorer,
        RAG_RERANK_MODEL="cross-encoder/mmarco-mMiniLMv2",
        RAG_SEARCH_RESULT_COUNT="10",
        RAG_SEARCH_DENSE_WEIGHT="0.5",
        RAG_SEARCH_LEXICAL_WEIGHT="0.5",
    )

    tuned = client.post(
        "/v1/search",
        json={
            "question": "rescisão antecipada",
            "limit": 1,
            "dense_weight": 0.9,
            "lexical_weight": 0.1,
            "rerank": True,
        },
    )

    assert tuned.status_code == 200
    body = tuned.json()
    assert len(passages_of(body)) == 1
    assert scorer.calls, "reranking was asked for and the model was never consulted"
    assert "multa por rescisão" in passages_of(body)[0]["text"]

    with psycopg.connect(schema_dsn) as check, check.cursor() as cursor:
        cursor.execute(
            "SELECT dense_weight, lexical_weight, result_count, rerank_model, caller "
            "FROM search_events ORDER BY id DESC LIMIT 1"
        )
        dense, lexical, count, rerank_model, caller = cursor.fetchone()
    assert (float(dense), float(lexical), count) == (0.9, 0.1, 1)
    assert rerank_model == "cross-encoder/mmarco-mMiniLMv2"
    assert caller == "v1"

    plain = client.post("/v1/search", json={"question": "rescisão antecipada"})
    assert plain.status_code == 200
    assert len(passages_of(plain.json())) == 3
    assert len(plain.json()["results"]) == 2
    with psycopg.connect(schema_dsn) as check, check.cursor() as cursor:
        cursor.execute(
            "SELECT dense_weight, lexical_weight, result_count, rerank_model "
            "FROM search_events ORDER BY id DESC LIMIT 1"
        )
        dense, lexical, count, rerank_model = cursor.fetchone()
    assert (float(dense), float(lexical), count) == (0.5, 0.5, 10)
    assert rerank_model is None


def test_ac_135(api_world, served_corpus):
    client, _ = api_world(
        embedder=FakeQueryEmbedder(degrees=0),
        RAG_SEARCH_MAX_RESULT_COUNT="25",
        RAG_SEARCH_MAX_CANDIDATE_DEPTH="200",
    )

    refused = client.post("/v1/search", json={"question": "rescisão", "limit": 26})

    assert refused.status_code == 422
    assert refused.json()["status"] == STATUS_FAILED
    error = refused.json()["error"]
    assert "26" in error and "25" in error
    assert "api.limits.search.limit" in error

    assert "results" not in refused.json()

    deep = client.post("/v1/search", json={"question": "rescisão", "candidate_depth": 201})
    assert deep.status_code == 422
    assert "search.max_candidate_depth" in deep.json()["error"]

    allowed = client.post("/v1/search", json={"question": "rescisão", "limit": 2})
    assert allowed.status_code == 200
    assert len(passages_of(allowed.json())) == 2


def test_ac_136(api_world, served_corpus):
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    routes = (
        ("POST", "/v1/search", {"json": {"question": "rescisão antecipada"}}),
        ("GET", "/v1/documents/drive-rescisao", {}),
        ("POST", "/mcp", {"json": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}}),
    )
    for method, path, kwargs in routes:
        for headers in ({"Authorization": ""}, {"Authorization": "Bearer wrong-token"}):
            response = client.request(method, path, headers=headers, **kwargs)

            assert response.status_code == 401, f"{method} {path} let a stranger in"
            body = response.text
            assert body.count("drive-") == 0
            assert "rescisão antecipada exige aviso" not in body
            assert "Contrato de locação" not in body

    named = client.post(
        "/v1/search",
        json={"question": "rescisão antecipada"},
        headers={"X-User-Id": "user@example.com"},
    )
    assert named.status_code == 400
    assert "no end-user identity" in named.json()["error"]

    assert client.post("/v1/search", json={"question": "rescisão antecipada"}).status_code == 200


def test_ac_137(api_world, served_corpus, caplog):
    import logging

    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    assert service.config.api.allowed_hosts == [ALLOWED_HOST]

    with caplog.at_level(logging.WARNING, logger="src.auth"):
        response = client.post(
            "/v1/search",
            json={"question": "rescisão antecipada"},
            headers={"Host": "rag.someone-else.example"},
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert "rag.someone-else.example" in error
    assert ALLOWED_HOST in error

    assert any(
        "rag.someone-else.example" in record.getMessage() and ALLOWED_HOST in record.getMessage()
        for record in caplog.records
    )

    with_token = client.post(
        "/v1/search",
        json={"question": "rescisão antecipada"},
        headers={"Host": "rag.someone-else.example", "Authorization": f"Bearer {API_TOKEN}"},
    )
    assert with_token.status_code == 400

    assert client.post("/v1/search", json={"question": "rescisão antecipada"}).status_code == 200


def test_ac_138(api_world, monkeypatch):
    from src.routes.health import StartupChecks

    client, _ = api_world()

    def refuse_every_connection(*args, **kwargs):
        raise AssertionError("the health route opened a connection to the archive")

    monkeypatch.setattr("psycopg.connect", refuse_every_connection)
    healthy = client.get("/healthz")

    assert healthy.status_code == 200
    assert healthy.json() == {"status": "ok", "checks_passed": True, "detail": ""}

    monkeypatch.undo()
    degraded_client, _ = api_world(
        checks=StartupChecks(passed=False, detail="could not reach the archive: connection refused")
    )
    monkeypatch.setattr("psycopg.connect", refuse_every_connection)
    degraded = degraded_client.get("/healthz")

    assert degraded.status_code == 503
    assert degraded.json()["checks_passed"] is False
    assert "could not reach the archive" in degraded.json()["detail"]

    anonymous = degraded_client.get("/healthz", headers={"Authorization": ""})
    assert anonymous.status_code == 503
    assert "drive-" not in anonymous.text
