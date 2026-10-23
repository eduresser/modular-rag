from __future__ import annotations

import dataclasses

import json

import anyio
import psycopg
import pytest

from src.boot import BootCompatibilityError, check_boot_compatibility
from src.mcp import build_server
from src.usecases.fetch import (
    ASSISTANT,
    SERVICE,
    FetchedDocument as FetchResult,
    FetchError,
    ceiling_for,
    fetch_on,
    span_for,
    window,
)
from src.responses import documents_search_payload, grouped_search_payload, result_id
from src.search_result import Found
from src.store_versions import activate_version
from src.usecases.search import SearchQuery, run_search
from tests.conftest import (
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    FakeQueryEmbedder,
    API_CORPUS,
    api_config,
    index_corpus,
)


def _overflowing_answer(monkeypatch, conn):
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")
    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test", result_count=1),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )
    assert isinstance(outcome, Found)
    assert len(outcome.hits) == 1
    return config, outcome

def test_ac_244(api_world, served_corpus, schema_dsn):
    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)

    fetched = anyio.run(lambda: server.call_tool("fetch", {"id": "drive-rescisao"})).structured_content

    with psycopg.connect(schema_dsn) as check, check.cursor() as cursor:
        cursor.execute(
            "SELECT content FROM chunks WHERE document_id = %s ORDER BY ordinal",
            (served_corpus["drive-rescisao"],),
        )
        stored = [row[0] for row in cursor.fetchall()]
    assert stored, "the fixture corpus has to hold this document for the count to mean anything"
    serialized = repr(fetched)
    for content in stored:
        assert content in fetched["text"]
        assert serialized.count(content) == 1, "the window's text is carried once, not twice"

    assert len(fetched["pieces"]) == len(stored)
    for piece in fetched["pieces"]:
        assert "text" not in piece
        assert set(piece) == {"ordinal", "page_start", "page_end", "heading_path"}

def test_ac_245(monkeypatch, schema_dsn):
    config = api_config(monkeypatch, schema_dsn)
    assert config.api.defaults.fetch.limit == 50
    assert config.mcp.defaults.fetch.limit < config.api.defaults.fetch.limit

    assert window(config, offset=0, limit=None, audience=ASSISTANT) == (
        0,
        config.mcp.defaults.fetch.limit,
    )
    assert window(config, offset=0, limit=None, audience=SERVICE) == (
        0,
        config.api.defaults.fetch.limit,
    )
    assert ceiling_for(config, ASSISTANT) < ceiling_for(config, SERVICE)

    over = config.mcp.defaults.fetch.limit + 1
    with pytest.raises(FetchError) as refusal:
        window(config, offset=0, limit=over, audience=ASSISTANT)
    assert str(config.mcp.defaults.fetch.limit) in str(refusal.value)
    assert "mcp.limits.fetch.limit" in str(refusal.value)
    assert window(config, offset=0, limit=over, audience=SERVICE) == (0, over)

    inverted = api_config(
        monkeypatch,
        schema_dsn,
        RAG_API_MAX_DOCUMENT_PIECES="10",
        RAG_API_ASSISTANT_DOCUMENT_PIECES="20",
    )
    with pytest.raises(BootCompatibilityError) as stopped:
        check_boot_compatibility(inverted)
    assert "mcp.defaults.fetch.limit" in str(stopped.value)
    assert "api.limits.fetch.limit" in str(stopped.value)

CONTEXT_BUDGET_CHARS = 60_000


def _model_visible(result) -> int:
    return sum(len(block.text) for block in result.content)


def _wire_size(result) -> int:
    return _model_visible(result) + len(
        json.dumps(result.structured_content, ensure_ascii=False)
    )


def test_ac_264(monkeypatch, schema_dsn, api_world, served_corpus):
    clause = (
        "Cláusula {n}: a parte locatária obriga-se a manter o imóvel em perfeito estado "
        "de conservação durante toda a vigência do presente contrato, respondendo por "
        "qualquer dano causado por uso indevido, que será reparado às suas expensas no "
        "prazo de trinta dias contados da notificação, sob pena de rescisão por justa "
        "causa e de aplicação da multa prevista na cláusula seguinte. "
    ) * 6
    corpus = tuple(
        DocumentSpec(
            f"drive-volume-{d}",
            f"Contrato de locação — Unidade {d} do edifício comercial",
            tuple(
                ChunkSpec(
                    ordinal,
                    clause.format(n=ordinal),
                    degrees=0,
                    pages=(ordinal + 1, ordinal + 2),
                    headings=("Das obrigações da locatária", f"Cláusula {ordinal}"),
                )
                for ordinal in range(12)
            ),
        )
        for d in range(12)
    )
    with psycopg.connect(schema_dsn) as setup:
        index_corpus(setup, corpus)
        setup.commit()

    try:
        _assert_budget(monkeypatch, api_world, schema_dsn)
    finally:
        with psycopg.connect(schema_dsn, autocommit=True) as cleanup, cleanup.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunks WHERE document_id IN "
                "(SELECT id FROM documents WHERE external_id LIKE 'drive-volume-%%')"
            )
            cursor.execute("DELETE FROM documents WHERE external_id LIKE 'drive-volume-%%'")
            cursor.execute("VACUUM (ANALYZE) chunks")


def _assert_budget(monkeypatch, api_world, schema_dsn):
    _, service = api_world(
        embedder=FakeQueryEmbedder(degrees=0),
        RAG_SEARCH_CANDIDATE_DEPTH="500",
    )
    server = build_server(service)
    ceiling = service.config.mcp.defaults.search.limit

    answer = anyio.run(lambda: server.call_tool("search", {"query": "conservação do imóvel"}))
    returned = sum(len(g["passages"]) for g in answer.structured_content["results"])
    assert returned == ceiling, "the corpus must fill the result ceiling"
    left_out = answer.structured_content["more"]["candidates"]
    assert left_out > ceiling, "and must overflow it by more than it returned"

    assert _model_visible(answer) < CONTEXT_BUDGET_CHARS, (
        f"a search at the ceiling puts {_model_visible(answer):,} characters in front of "
        f"the model, over the {CONTEXT_BUDGET_CHARS:,} budgeted "
        f"({_wire_size(answer):,} on the wire)"
    )

    fetched = anyio.run(lambda: server.call_tool("fetch", {"id": "drive-volume-0"}))
    assert (
        len(fetched.structured_content["pieces"])
        == service.config.mcp.defaults.fetch.limit
    ), "the document must be longer than the assistant's window for this to be the worst case"
    assert _model_visible(fetched) < CONTEXT_BUDGET_CHARS, (
        f"a full fetch puts {_model_visible(fetched):,} characters in front of the model, "
        f"over the {CONTEXT_BUDGET_CHARS:,} budgeted ({_wire_size(fetched):,} on the wire)"
    )


_LONG_DOCUMENT = DocumentSpec(
    "drive-longo",
    "Contrato de locação — Edifício inteiro",
    tuple(
        ChunkSpec(
            ordinal,
            f"Cláusula {ordinal}: obrigação de número {ordinal} da parte locatária.",
            degrees=0,
            pages=(ordinal + 1, ordinal + 1),
            headings=(f"Cláusula {ordinal}",),
        )
        for ordinal in range(30)
    ),
)


def test_ac_246(monkeypatch, conn, schema_dsn):
    written = index_corpus(conn, (_LONG_DOCUMENT,))
    activate_version(conn, TEST_STAMP)
    document_id = written["drive-longo"]
    config = api_config(monkeypatch, schema_dsn)

    found = fetch_on(conn, result_id(document_id, 12), config=config)

    assert isinstance(found, FetchResult)
    assert config.api.defaults.fetch.passage_window < config.api.defaults.fetch.limit
    assert [piece.ordinal for piece in found.pieces] == list(
        range(12, 12 + config.api.defaults.fetch.passage_window)
    )

    for audience in (ASSISTANT, SERVICE):
        assert span_for(config, audience=audience, named_piece=12) == config.api.defaults.fetch.passage_window

    wider = fetch_on(conn, result_id(document_id, 12), config=config, limit=8)
    assert isinstance(wider, FetchResult)
    assert [piece.ordinal for piece in wider.pieces] == list(range(12, 20))

def test_ac_247(monkeypatch, conn, schema_dsn):
    index_corpus(conn, (_LONG_DOCUMENT,))
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, schema_dsn)
    whole = len(_LONG_DOCUMENT.chunks)

    found = fetch_on(conn, "drive-longo", config=config)

    assert isinstance(found, FetchResult)
    assert found.offset == 0
    assert found.pieces[0].ordinal == 0
    assert len(found.pieces) == whole > config.api.defaults.fetch.passage_window
    assert span_for(config, audience=SERVICE, named_piece=None) == config.api.defaults.fetch.limit

    assistant = fetch_on(conn, "drive-longo", config=config, audience=ASSISTANT)
    assert isinstance(assistant, FetchResult)
    assert len(assistant.pieces) == config.mcp.defaults.fetch.limit
    assert assistant.piece_count == whole
    assert assistant.offset == 0

    rest = fetch_on(
        conn, "drive-longo", config=config, audience=ASSISTANT, offset=len(assistant.pieces)
    )
    assert isinstance(rest, FetchResult)
    assert rest.pieces[0].ordinal == len(assistant.pieces)

def test_ac_248(monkeypatch, conn):
    _config, outcome = _overflowing_answer(monkeypatch, conn)
    left_out = outcome.omitted
    assert left_out, "the fixture corpus has to overflow for this to mean anything"

    for payload in (grouped_search_payload(outcome), documents_search_payload(outcome)):
        more = payload["more"]
        assert more["candidates"] == len(left_out)
        assert more["documents"] == len({hit.provenance.document_id for hit in left_out})

        assert set(more) == {"candidates", "documents", "next_offset", "note"}
        assert all(isinstance(more[key], int) for key in ("candidates", "documents"))
        for hit in left_out:
            assert hit.provenance.external_id not in repr(more)

def test_ac_249(monkeypatch, conn):
    _config, outcome = _overflowing_answer(monkeypatch, conn)
    query_fields = {field.name for field in dataclasses.fields(SearchQuery)}

    for payload in (grouped_search_payload(outcome), documents_search_payload(outcome)):
        assert not query_fields & {"page", "page_number", "cursor", "next_page"}
        assert not {"page", "cursor", "next_cursor"} & set(payload)

        note = payload["more"]["note"].lower()
        assert "offset" in note
        assert "narrower question" in note
        assert "fetch" not in note, "the note must not point at ids the answer no longer carries"

def test_ac_250(api_world, served_corpus):
    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    answer = anyio.run(
        lambda: build_server(service).call_tool("search", {"query": "rescisão antecipada"})
    ).structured_content
    passages = [p for group in answer["results"] for p in group["passages"]]
    assert passages

    for passage in passages:
        assert "score" not in passage

        assert passage["found_by"]
        assert set(passage["found_by"]) <= {"dense", "lexical"}

        assert "citation" in passage
        assert "page_start" in passage and "heading_path" in passage

def test_ac_251(api_world, served_corpus):
    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))

    conventional = client.post("/v1/search", json={"question": "rescisão antecipada"}).json()
    assistant = anyio.run(
        lambda: build_server(service).call_tool("search", {"query": "rescisão antecipada"})
    ).structured_content

    for group in conventional["results"]:
        for passage in group["passages"]:
            assert isinstance(passage["score"], float)
            assert passage["found_by"]
            assert {"page_start", "page_end", "heading_path", "citation"} <= set(passage)

    conventional_keys = {k for g in conventional["results"] for p in g["passages"] for k in p}
    assistant_keys = {k for g in assistant["results"] for p in g["passages"] for k in p}
    assert conventional_keys - assistant_keys == {"score"}
    assert assistant_keys - conventional_keys == set()

def test_ac_252(monkeypatch, schema_dsn, api_world, served_corpus):
    from src.usecases.fetch import Outline, outline_on

    with psycopg.connect(schema_dsn) as setup:
        index_corpus(setup, (_LONG_DOCUMENT,))
        setup.commit()
    config = api_config(monkeypatch, schema_dsn)
    conn = psycopg.connect(schema_dsn)

    found = fetch_on(conn, result_id(_document_id(conn, "drive-longo"), 12), config=config)
    assert isinstance(found, FetchResult)
    assert len(found.pieces) < found.piece_count

    standalone = outline_on(conn, "drive-longo")
    assert isinstance(standalone, Outline)
    assert found.outline == standalone.entries

    assert [entry.ordinal for entry in found.outline] == list(range(found.piece_count))
    assert max(entry.ordinal for entry in found.outline) > max(
        piece.ordinal for piece in found.pieces
    )

    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    fetched = anyio.run(
        lambda: build_server(service).call_tool("fetch", {"id": "drive-rescisao"})
    ).structured_content
    assert fetched["outline"]
    assert set(fetched["outline"][0]) == {"ordinal", "label"}
    conn.close()


def _document_id(conn, external_id: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", (external_id,))
        return int(cursor.fetchone()[0])

def test_ac_253(api_world, served_corpus, schema_dsn):
    from src.store import upsert_document

    with psycopg.connect(schema_dsn) as setup:
        upsert_document(
            setup,
            external_id="drive-rescisao",
            link="https://drive.google.com/file/d/drive-rescisao/view",
        )
        with setup.cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET link = NULL WHERE external_id <> %s", ("drive-rescisao",)
            )
        setup.commit()

    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    assistant = anyio.run(
        lambda: build_server(service).call_tool("search", {"query": "rescisão antecipada"})
    ).structured_content
    conventional = client.post("/v1/search", json={"question": "rescisão antecipada"}).json()

    for group in assistant["results"]:
        assert {"id", "title", "url"} <= set(group)
        assert group["id"] and group["title"]
        assert "document" not in group and "link" not in group

    linked = next(g for g in assistant["results"] if g["id"] == "drive-rescisao")
    assert linked["url"] == "https://drive.google.com/file/d/drive-rescisao/view"

    unlinked = [g for g in assistant["results"] if g["id"] != "drive-rescisao"]
    assert unlinked, "the corpus needs a document with no link for this half to mean anything"
    for group in unlinked:
        assert group["url"] is None

    for group in conventional["results"]:
        assert {"document", "title", "link"} <= set(group)
        assert "id" not in group and "url" not in group
