from __future__ import annotations

from datetime import UTC, datetime

import anyio
import pytest

from src.ingest import ingest_documents
from src.mcp import build_server
from src.responses import (
    ASSISTANT,
    BRIEF_CUT_MARK,
    STATUS_FOUND,
    documents_search_payload,
    grouped_search_payload,
    result_id,
)
from src.search_result import Found
from src.sources.base import FetchedDocument, SourceDocument
from src.store import ModelStamp, upsert_document
from src.store_versions import activate_version
from src.usecases.fetch import FetchedDocument as FetchResult
from src.usecases.fetch import FetchError, fetch_on
from src.usecases.fetch import Outline as OutlineResult
from src.usecases.fetch import outline_on
from src.usecases.search import SearchQuery, run_search
from tests.conftest import (
    API_CORPUS,
    TEST_DIMENSION,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    FakeQueryEmbedder,
    api_config,
    contract_payload,
    index_corpus,
    ingest_world,
)

import dataclasses

_PREVIOUS_STAMP = ModelStamp(model="text-embedding-2", dimension=TEST_DIMENSION)


class _FixedSource:
    def __init__(self, documents: "list[SourceDocument]") -> None:
        self._documents = list(documents)

    def sweep(self):
        yield from self._documents

    def changes_since(self, position):
        raise NotImplementedError

    def start_position(self):
        raise NotImplementedError

    def fetch(self, document: SourceDocument) -> FetchedDocument:
        return FetchedDocument(
            external_id=document.external_id,
            content=b"%PDF-1.7\n",
            media_type="application/pdf",
            content_digest=f"digest-{document.external_id}",
        )

    def probe(self, external_id):
        raise NotImplementedError


def _document_record(conn, external_id: str):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT title, media_type, link, location, modified_at, size_bytes "
            "FROM documents WHERE external_id = %s",
            (external_id,),
        )
        return cursor.fetchone()


def _ingest_one(monkeypatch, conn, document: SourceDocument):
    world = ingest_world(monkeypatch, outputs={document.external_id: contract_payload()})
    pipeline = dataclasses.replace(world.pipeline, source=_FixedSource([document]))
    ingest_documents(conn, pipeline, [document])


def _grouped_answer(monkeypatch, conn):
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")
    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )
    assert isinstance(outcome, Found)
    return outcome


def test_ac_182(monkeypatch, conn):
    outcome = _grouped_answer(monkeypatch, conn)

    grouped = grouped_search_payload(outcome)["results"]

    documents = [group["document"] for group in grouped]
    assert documents.count("drive-rescisao") == 1
    rescisao = next(group for group in grouped if group["document"] == "drive-rescisao")
    assert rescisao["title"] == "Contrato de locação — Sala 402"
    assert len(rescisao["passages"]) == 2

    for passage in rescisao["passages"]:
        assert "title" not in passage
        assert "link" not in passage
        assert "location" not in passage


def test_ac_182_through_the_surfaces(api_world, served_corpus):
    import anyio

    from src.mcp import build_server

    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))

    for name, key, payload in (
        ("v1", "document", client.post("/v1/search", json={"question": "rescisão antecipada"}).json()),
        (
            "mcp",
            "id",
            anyio.run(
                lambda: build_server(service).call_tool("search", {"query": "rescisão antecipada"})
            ).structured_content,
        ),
    ):
        assert payload["status"] == STATUS_FOUND, name
        for group in payload["results"]:
            assert group[key] and group["title"], name
            assert group["passages"], name
            for passage in group["passages"]:
                assert not {"title", "link", "url", "location"} & set(passage), name
        names = [group[key] for group in payload["results"]]
        assert len(names) == len(set(names)), f"{name}: a document was named more than once"


def test_ac_183(monkeypatch, conn):
    outcome = _grouped_answer(monkeypatch, conn)
    flat_order = [hit.provenance.external_id for hit in outcome.hits]
    assert flat_order == ["drive-rescisao", "drive-rescisao", "drive-reajuste"]

    grouped = grouped_search_payload(outcome)["results"]

    ranks = [passage["rank"] for group in grouped for passage in group["passages"]]
    assert ranks == [1, 2, 3]

    assert [group["document"] for group in grouped] == ["drive-rescisao", "drive-reajuste"]

def test_ac_184(monkeypatch, conn):
    modified = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    document = SourceDocument(
        external_id="doc-184", name="Contrato de locação", kind="application/pdf",
        revision_marker="v1", modified_at=modified, size_bytes=4096,
        parents=("folder-a", "folder-b"),
        link="https://drive.google.com/file/d/doc-184/view",
    )

    _ingest_one(monkeypatch, conn, document)

    title, media_type, link, location, modified_at, size_bytes = _document_record(conn, "doc-184")
    assert title == "Contrato de locação"
    assert media_type == "application/pdf"
    assert link == "https://drive.google.com/file/d/doc-184/view"
    assert location == ["folder-a", "folder-b"]
    assert modified_at == modified
    assert size_bytes == 4096


def test_ac_185(monkeypatch, conn):
    document = SourceDocument(
        external_id="doc-185", name="Nota interna",
        kind="application/vnd.google-apps.document", revision_marker="7",
        modified_at=None, size_bytes=None, parents=(), link=None,
    )

    _ingest_one(monkeypatch, conn, document)

    _, media_type, link, location, modified_at, size_bytes = _document_record(conn, "doc-185")
    assert media_type == "application/vnd.google-apps.document"
    assert link is None
    assert location == []
    assert modified_at is None
    assert size_bytes is None


def test_a_touch_still_records_the_documents_own_facts(monkeypatch, conn):
    document = SourceDocument(
        external_id="doc-106", name="Contrato antigo", kind="application/pdf",
        revision_marker="v1", modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        size_bytes=10, parents=("folder-a",), link="https://drive.google.com/file/d/doc-106/view",
    )
    world = ingest_world(monkeypatch, outputs={"doc-106": contract_payload()})
    pipeline = dataclasses.replace(world.pipeline, source=_FixedSource([document]))
    ingest_documents(conn, pipeline, [document])

    from src.reprocess import ACTION_TOUCH

    run = ingest_documents(conn, pipeline, [document])
    assert [result.action for result in run.results] == [ACTION_TOUCH]

    _, media_type, link, location, modified_at, size_bytes = _document_record(conn, "doc-106")
    assert media_type == "application/pdf"
    assert link == "https://drive.google.com/file/d/doc-106/view"
    assert location == ["folder-a"]
    assert modified_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert size_bytes == 10

def test_ac_186(monkeypatch, conn):
    written = index_corpus(conn, API_CORPUS, stamp=_PREVIOUS_STAMP)
    activate_version(conn, _PREVIOUS_STAMP)
    document_id = written["drive-rescisao"]
    identifier = result_id(document_id, 1)
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    before = fetch_on(conn, identifier, config=config)
    assert isinstance(before, FetchResult)
    assert before.pieces[0].heading_path == ("Da rescisão", "Da multa")

    index_corpus(conn, API_CORPUS, stamp=TEST_STAMP)
    activate_version(conn, TEST_STAMP)

    after = fetch_on(conn, identifier, config=config)
    assert isinstance(after, FetchResult)
    assert after.external_id == "drive-rescisao"
    assert after.pieces[0].ordinal == 1
    assert after.pieces[0].heading_path == ("Da rescisão", "Da multa")

def test_ac_187(monkeypatch, conn):
    written = index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    document_id = written["drive-rescisao"]
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    found = fetch_on(conn, result_id(document_id, 1), config=config)

    assert isinstance(found, FetchResult)
    assert [piece.ordinal for piece in found.pieces] == [1]
    assert found.pieces[0].heading_path == ("Da rescisão", "Da multa")

    earlier = fetch_on(conn, result_id(document_id, 0), config=config)
    assert isinstance(earlier, FetchResult)
    assert [piece.ordinal for piece in earlier.pieces] == [0, 1]

def test_ac_188(monkeypatch, conn):
    written = index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id FROM chunks WHERE document_id = %s ORDER BY ordinal LIMIT 1",
            (written["drive-rescisao"],),
        )
        (chunk_id,) = cursor.fetchone()
    identifier_issued_earlier = f"drive-rescisao#{chunk_id}"

    from_earlier_identifier = fetch_on(conn, identifier_issued_earlier, config=config)
    from_the_sources_own = fetch_on(conn, "drive-rescisao", config=config)

    assert isinstance(from_earlier_identifier, FetchResult)
    assert from_earlier_identifier.external_id == "drive-rescisao"
    assert isinstance(from_the_sources_own, FetchResult)
    assert from_the_sources_own.external_id == "drive-rescisao"

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


def test_ac_190(monkeypatch, conn):
    _config, outcome = _overflowing_answer(monkeypatch, conn)
    query_fields = {field.name for field in dataclasses.fields(SearchQuery)}

    for payload in (grouped_search_payload(outcome), documents_search_payload(outcome)):
        assert not query_fields & {"page", "page_number", "cursor", "next_page"}
        assert "offset" in query_fields
        assert "page" not in payload
        assert "cursor" not in payload

        note = payload["more"]["note"].lower()
        assert "offset" in note
        assert "narrower question" in note
        assert payload["more"]["next_offset"] > 0

def test_ac_192(monkeypatch, conn):
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    found = fetch_on(conn, "drive-rescisao", config=config, page_start=4, page_end=5)

    assert isinstance(found, FetchResult)
    assert [piece.ordinal for piece in found.pieces] == [0, 1]

    narrower = fetch_on(conn, "drive-rescisao", config=config, page_start=4, page_end=4)
    assert isinstance(narrower, FetchResult)
    assert [piece.ordinal for piece in narrower.pieces] == [0]

def test_ac_193(monkeypatch, conn):
    unpaginated = DocumentSpec(
        "drive-sem-paginas",
        "Termo de aditamento — sem paginação",
        (
            ChunkSpec(
                0,
                "Cláusula única: as partes ratificam os demais termos.",
                degrees=0,
                headings=("Da ratificação",),
            ),
        ),
    )
    index_corpus(conn, (unpaginated,))
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    with pytest.raises(FetchError) as excinfo:
        fetch_on(conn, "drive-sem-paginas", config=config, page_start=1, page_end=1)

    reason = str(excinfo.value).lower()
    assert "no pagination" in reason
    assert "piece offset" in reason

    still_readable = fetch_on(conn, "drive-sem-paginas", config=config)
    assert isinstance(still_readable, FetchResult)
    assert [piece.ordinal for piece in still_readable.pieces] == [0]

def test_ac_194(conn):
    index_corpus(conn, API_CORPUS)
    activate_version(conn, TEST_STAMP)

    found = outline_on(conn, "drive-rescisao")

    assert isinstance(found, OutlineResult)
    assert found.basis == "headings"
    assert found.entries != ()

def test_ac_195(conn):
    unheaded = DocumentSpec(
        "drive-sem-titulos",
        "Aditivo sem títulos",
        (
            ChunkSpec(0, "Primeira cláusula do aditivo.", degrees=0, pages=(1, 1)),
            ChunkSpec(1, "Segunda cláusula do aditivo.", degrees=0, pages=(2, 2)),
        ),
    )
    index_corpus(conn, (unheaded,))
    activate_version(conn, TEST_STAMP)

    found = outline_on(conn, "drive-sem-titulos")

    assert isinstance(found, OutlineResult)
    assert found.basis == "pages"
    assert found.entries != ()

def test_ac_196(api_world, served_corpus, schema_dsn):
    import psycopg

    clause = (
        "Cláusula vigésima: a parte locatária obriga-se a manter o imóvel em "
        "perfeito estado de conservação durante toda a vigência do contrato. "
        "Qualquer dano causado por uso indevido será reparado às expensas da "
        "parte locatária, no prazo de trinta dias contados da notificação. "
        "A ausência de reparo no prazo estipulado autoriza a parte locadora a "
        "promover a rescisão do contrato por justa causa."
    )
    with psycopg.connect(schema_dsn) as setup:
        index_corpus(
            setup,
            (
                DocumentSpec(
                    "drive-conservacao",
                    "Contrato de locação — Sala 900",
                    (
                        ChunkSpec(
                            0,
                            clause,
                            degrees=0,
                            pages=(9, 9),
                            headings=("Da conservação",),
                        ),
                    ),
                ),
            ),
        )
        setup.commit()
    client, service = api_world(
        embedder=FakeQueryEmbedder(degrees=0), RAG_SEARCH_BRIEF_LENGTH=160
    )
    server = build_server(service)

    answer = anyio.run(
        lambda: server.call_tool("search", {"query": "conservação do imóvel"})
    ).structured_content
    group = next(item for item in answer["results"] if item["id"] == "drive-conservacao")
    preview = group["passages"][0]

    assert len(preview["text"]) <= 160
    assert len(preview["text"]) < len(clause) * 0.6
    assert preview["text"].endswith(BRIEF_CUT_MARK)
    assert preview["truncated"] is True
    assert preview["full_length"] == len(clause)

    assert group["id"] == "drive-conservacao"
    assert preview["id"]
    assert preview["page_start"] == 9
    assert preview["heading_path"] == ["Da conservação"]
    assert preview["citation"]

    fetched = anyio.run(
        lambda: server.call_tool("fetch", {"id": preview["id"]})
    ).structured_content
    assert fetched["text"] == clause
    assert "text" not in fetched["pieces"][0]
    assert fetched["pieces"][0]["page_start"] == 9

    full = client.post("/v1/search", json={"question": "conservação do imóvel"}).json()
    full_group = next(item for item in full["results"] if item["document"] == "drive-conservacao")
    full_passage = full_group["passages"][0]
    assert full_passage["text"] == clause
    assert full_passage["truncated"] is False
    assert full_passage["full_length"] == len(clause)

def test_ac_197(monkeypatch, conn):
    index_corpus(conn, API_CORPUS)
    upsert_document(
        conn, external_id="drive-rescisao", link="https://drive.google.com/file/d/drive-rescisao/view"
    )
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")
    outcome = run_search(
        SearchQuery(question="rescisão antecipada", caller="test"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )
    assert isinstance(outcome, Found)

    grouped = grouped_search_payload(outcome)["results"]

    rescisao = next(group for group in grouped if group["document"] == "drive-rescisao")
    assert rescisao["link"] == "https://drive.google.com/file/d/drive-rescisao/view"

def test_ac_198(api_world, served_corpus):
    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)

    tools = {tool.name: tool for tool in anyio.run(server.list_tools)}
    assert set(tools) == {"search", "fetch", "filters", "outline"}

    for tool in tools.values():
        assert tool.output_schema is not None
        assert tool.output_schema.get("type") == "object"
        assert tool.output_schema.get("properties")
        assert "status" in tool.output_schema["properties"]

        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True


def test_ac_199(api_world, served_corpus, schema_dsn):
    import psycopg

    with psycopg.connect(schema_dsn) as setup:
        upsert_document(
            setup, external_id="drive-rescisao", link="https://drive.google.com/file/d/drive-rescisao/view"
        )
        setup.commit()

    _, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)
    found = anyio.run(lambda: server.call_tool("search", {"query": "rescisão antecipada"})).structured_content
    result = next(hit for hit in found["results"] if hit["id"] == "drive-rescisao")

    with psycopg.connect(schema_dsn) as check:
        outcome = run_search(
            SearchQuery(question="rescisão antecipada", caller="test"),
            config=service.config,
            embedder=FakeQueryEmbedder(degrees=0),
            connection=check,
        )
        assert isinstance(outcome, Found)
        document_record = next(
            group
            for group in grouped_search_payload(outcome, ASSISTANT)["results"]
            if group["id"] == "drive-rescisao"
        )

    fetched = anyio.run(
        lambda: server.call_tool("fetch", {"id": result["passages"][0]["id"]})
    ).structured_content

    assert fetched["id"] == "drive-rescisao"
    assert fetched["title"] == document_record["title"]
    assert isinstance(fetched["text"], str) and fetched["text"]
    assert fetched["url"] == "https://drive.google.com/file/d/drive-rescisao/view"

    assert fetched["pieces"]
    assert {"page_start", "page_end", "heading_path"} <= fetched["pieces"][0].keys()

    assert fetched["id"] == document_record["id"]
    assert fetched["url"] == document_record["url"]
    assert fetched["location"] == document_record["location"]

def test_ac_200(monkeypatch, conn):
    outcome = _grouped_answer(monkeypatch, conn)

    documents = documents_search_payload(outcome)["results"]


    assert [doc["document"] for doc in documents] == ["drive-rescisao", "drive-reajuste"]
    rescisao = documents[0]
    assert rescisao["match_count"] == 2
    assert isinstance(rescisao["evidence"], str) and rescisao["evidence"]
    assert rescisao["score"] >= rescisao["average_score"]

    for doc in documents:
        assert "passages" not in doc
        assert "text" not in doc


def test_ac_200_through_the_surface(api_world, served_corpus):
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    body = client.post(
        "/v1/search", json={"question": "rescisão antecipada", "view": "documents"}
    ).json()

    assert body["status"] == STATUS_FOUND
    assert body["results"]
    for document in body["results"]:
        assert document["document"] and document["title"]
        assert document["match_count"] >= 1
        assert "passages" not in document

    refused = client.post("/v1/search", json={"question": "rescisão", "view": "everything"})
    assert refused.status_code == 422
