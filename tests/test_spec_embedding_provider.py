
from decimal import Decimal

import pytest

from src.boot import BootCompatibilityError, check_boot_compatibility
from src.config import load_config
from src.embedding.batch import Group, embed_batches, plan_batches
from src.embedding.client import EmbeddingError
from src.embedding.cost import record_embedding, record_run
from src.embedding.pinning import (
    FIELD_ALLOW_FALLBACKS,
    FIELD_ORDER,
    FIELD_ROUTING,
    ProviderSubstituted,
    pinned_client,
)
from src.embedding.reembed import cutover, reembed
from src.embedding.stamp import attach, stamp_from_config, stamp_of
from src.models import Chunk, VersionStamps
from src.prices import PriceBook
from src.search_dense import dense_candidates
from src.store import ModelStamp, upsert_document, upsert_pieces
from src.store_versions import active_version, activate_version
from src.usage import (
    OUTCOME_FAILED,
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    read_usage,
)
from tests.conftest import (
    EMBED_MODEL,
    configured_amount,
    TEST_CLIENT,
    TEST_DIMENSION,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    FakeEmbeddings,
    direction,
    index_corpus,
    token_count,
)
from tests.conftest import FULL_DEPLOYMENT, deployment_config


def _deployment(monkeypatch, **overrides):
    return deployment_config(
        FULL_DEPLOYMENT, **{"RAG_EMBED_DIMENSION": TEST_DIMENSION, **overrides}
    )


def _chunk(ordinal: int, content: str) -> Chunk:
    return Chunk(
        document_external_id="doc-1",
        ordinal=ordinal,
        content=content,
        stamps=VersionStamps(reader_version="v1", fingerprint="fp1"),
        page_start=1,
        page_end=1,
    )


def test_ac_050(monkeypatch, conn):
    """AC-050: The provider is selected entirely by configuration @spec:AC-050"""
    first = _deployment(
        monkeypatch, RAG_EMBED_MODEL="qwen/qwen3-embedding-8b", RAG_EMBED_PROVIDER="DeepInfra"
    )
    second = _deployment(
        monkeypatch, RAG_EMBED_MODEL="openai/text-embedding-3-large", RAG_EMBED_PROVIDER="Novita"
    )
    providers = {
        first.embed.model: FakeEmbeddings(model=first.embed.model),
        second.embed.model: FakeEmbeddings(model=second.embed.model),
    }

    vectors = {}
    for config in (first, second):
        provider = providers[config.embed.model]
        client = pinned_client(config, http=provider.client(), token=lambda: "fake-key")
        vectors[config.embed.model] = client.embed_documents(["cláusula de rescisão"]).vectors[0]

    assert set(vectors) == {"qwen/qwen3-embedding-8b", "openai/text-embedding-3-large"}
    assert providers["qwen/qwen3-embedding-8b"].bodies[0]["model"] == "qwen/qwen3-embedding-8b"
    assert providers["qwen/qwen3-embedding-8b"].bodies[0][FIELD_ROUTING][FIELD_ORDER] == ["DeepInfra"]
    assert providers["openai/text-embedding-3-large"].bodies[0][FIELD_ROUTING][FIELD_ORDER] == ["Novita"]

    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")
    for config in (first, second):
        stamp = stamp_from_config(config)
        upsert_pieces(
            conn,
            document_id=document_id,
            stamp=stamp,
            pieces=attach([_chunk(0, "cláusula de rescisão")], [vectors[config.embed.model]], stamp),
        )
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM chunks WHERE document_id = %s", (document_id,))
        stamps = {stamp_of(conn, row[0]) for row in cursor.fetchall()}
    assert stamps == {stamp_from_config(first), stamp_from_config(second)}


def test_ac_051(monkeypatch):
    """AC-051: The configured request reaches the provider, except where it would overwrite a guarantee @spec:AC-051"""
    config = _deployment(
        monkeypatch,
        RAG_EMBED_REQUEST={
            "headers": {"x-api-key": "the-key"},
            "params": {"api-version": "2023-05-15"},
            "body": {"encoding_format": "float", "user": "example-user"},
        },
    )
    provider = FakeEmbeddings()

    check_boot_compatibility(config)
    pinned_client(config, http=provider.client(), token=lambda: "fake-key").embed_documents(["texto"])

    assert provider.bodies[0]["encoding_format"] == "float"
    assert provider.bodies[0]["user"] == "example-user"
    sent = provider.requests[0]
    assert sent.headers["x-api-key"] == "the-key"
    assert sent.headers["Authorization"].startswith("Bearer ")
    assert "api-version=2023-05-15" in str(sent.url)

    smuggled = _deployment(monkeypatch, RAG_EMBED_REQUEST={"body": {"model": "something-else"}})
    with pytest.raises(BootCompatibilityError):
        check_boot_compatibility(smuggled)

    refused = FakeEmbeddings()
    with pytest.raises(EmbeddingError) as excinfo:
        pinned_client(smuggled, http=refused.client(), token=lambda: "fake-key")
    assert "model" in str(excinfo.value)
    assert refused.requests == []


def test_ac_052(monkeypatch):
    """AC-052: A routed provider is pinned, and a substitution is an error @spec:AC-052"""
    config = _deployment(monkeypatch, RAG_EMBED_PROVIDER="DeepInfra")
    provider = FakeEmbeddings(served_provider="DeepInfra")

    pinned_client(config, http=provider.client(), token=lambda: "fake-key").embed_documents(["texto"])

    assert provider.bodies[0][FIELD_ROUTING] == {
        FIELD_ALLOW_FALLBACKS: False,
        FIELD_ORDER: ["DeepInfra"],
    }

    unavailable = FakeEmbeddings(fail_batches=(0,), status=503)
    with pytest.raises(EmbeddingError):
        pinned_client(config, http=unavailable.client(), token=lambda: "fake-key").embed_documents(["texto"])

    substituted = FakeEmbeddings(served_provider="Novita")
    with pytest.raises(ProviderSubstituted):
        pinned_client(config, http=substituted.client(), token=lambda: "fake-key").embed_documents(["texto"])

    other_model = FakeEmbeddings(served_model="google/gemini-embedding-001")
    with pytest.raises(EmbeddingError):
        pinned_client(config, http=other_model.client(), token=lambda: "fake-key").embed_documents(["texto"])


def test_ac_053(conn):
    """AC-053: The model that produced a vector is stored with it @spec:AC-053"""
    other = ModelStamp(model="qwen/qwen3-embedding-8b", dimension=TEST_DIMENSION)
    corpus = [
        DocumentSpec(
            external_id="doc-1",
            title="Contrato",
            chunks=(ChunkSpec(0, "cláusula de rescisão", degrees=0.0, pages=(1, 1)),),
        )
    ]
    index_corpus(conn, corpus)
    index_corpus(conn, corpus, stamp=other)

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, embed_model, embed_dimension FROM chunks ORDER BY embed_model"
        )
        rows = cursor.fetchall()

    assert {(row[1], int(row[2])) for row in rows} == {
        (TEST_STAMP.model, TEST_STAMP.dimension),
        (other.model, other.dimension),
    }
    assert stamp_of(conn, rows[0][0]) == ModelStamp(model=rows[0][1], dimension=int(rows[0][2]))

    activate_version(conn, TEST_STAMP)
    candidates = dense_candidates(
        conn,
        query_vector=direction(0.0),
        stamp=active_version(conn),
        depth=10,
        index_search_effort=40,
    )
    active_ids = {row[0] for row in rows if row[1] == TEST_STAMP.model}
    assert {candidate.chunk_id for candidate in candidates} == active_ids


def test_ac_054(conn):
    """AC-054: Re-embedding writes new rows rather than updating @spec:AC-054"""
    index_corpus(
        conn,
        [
            DocumentSpec(
                external_id="doc-1",
                title="Contrato",
                chunks=(
                    ChunkSpec(0, "cláusula primeira", pages=(1, 1)),
                    ChunkSpec(1, "cláusula segunda", pages=(2, 2)),
                ),
            )
        ],
    )
    activate_version(conn, TEST_STAMP)
    target = ModelStamp(model="qwen/qwen3-embedding-4b", dimension=TEST_DIMENSION)

    def tuples(stamp: ModelStamp) -> set[tuple]:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, ctid::text, embedding::text FROM chunks
                 WHERE embed_model = %s AND embed_dimension = %s
                """,
                (stamp.model, stamp.dimension),
            )
            return set(cursor.fetchall())

    before = tuples(TEST_STAMP)

    outcome = reembed(
        conn, source=TEST_STAMP, target=target, embedder=FakeEmbeddings(model=target.model).embedder(model=target.model), batch_size=8, max_inputs=2048
    )
    cutover(conn, target)

    assert outcome.pieces_written == 2
    assert len(tuples(target)) == 2
    assert {row[0] for row in tuples(target)}.isdisjoint({row[0] for row in before})
    assert active_version(conn) == target

    assert tuples(TEST_STAMP) == before


def test_ac_055():
    """AC-055: Pieces are embedded in batches, grouped by document @spec:AC-055"""
    documents = {
        "doc-1": tuple(f"doc-1 peça {position}" for position in range(5)),
        "doc-2": tuple(f"doc-2 peça {position}" for position in range(4)),
    }
    groups = [Group(key=key, texts=texts) for key, texts in documents.items()]
    provider = FakeEmbeddings()

    run = embed_batches(plan_batches(groups, batch_size=6, max_inputs=2048), provider.embedder())

    assert sum(len(texts) for texts in documents.values()) == 9
    assert len(provider.bodies) == 2

    for body in provider.bodies:
        sent = body["input"]
        for key, texts in documents.items():
            present = [text for text in sent if text in texts]
            if not present:
                continue
            start = sent.index(present[0])
            assert sent[start : start + len(texts)] == list(texts), key
    assert run.embedded == 9


def test_ac_056(conn, price_book):
    """AC-056: Every embedding call records its cost and its reason @spec:AC-056"""
    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")
    provider = FakeEmbeddings()
    texts = ("cláusula de rescisão antecipada", "prazo de vigência")
    run = embed_batches(
        plan_batches([Group(key="doc-1", texts=texts)], batch_size=8, max_inputs=2048), provider.embedder()
    )
    (usage_id,) = record_run(
        conn,
        price_book,
        client=TEST_CLIENT,
        run=run,
        model="qwen/qwen3-embedding-8b",
        provider="openrouter",
        reason=REASON_CONTENT_CHANGED,
        document_ids={"doc-1": document_id},
    )

    row = read_usage(conn, usage_id)

    assert row.unit == "token"
    assert row.units == sum(token_count(text) for text in texts)
    assert row.unit_price == configured_amount(operation="embed", provider="openrouter", model=EMBED_MODEL, unit="token")
    assert row.model == "qwen/qwen3-embedding-8b"
    assert row.reason == REASON_CONTENT_CHANGED
    assert row.cost == row.units * row.unit_price

    for reason in (REASON_NEW_DOCUMENT, REASON_CONFIGURATION_CHANGED):
        other = read_usage(
            conn,
            record_embedding(
                conn,
                price_book,
                client=TEST_CLIENT,
                document_id=document_id,
                model="qwen/qwen3-embedding-8b",
                provider="openrouter",
                tokens=100,
                reason=reason,
                duration_ms=1,
            ),
        )
        assert other.reason == reason

    dearer = PriceBook([{"operation": "embed", "provider": "openrouter",
                         "model": "qwen/qwen3-embedding-8b", "unit": "token",
                         "amount": "0.5"}])
    assert read_usage(conn, usage_id).unit_price == configured_amount(operation="embed", provider="openrouter", model=EMBED_MODEL, unit="token")
    assert dearer.price_for(
        operation="embed", provider="openrouter", model="qwen/qwen3-embedding-8b", unit="token"
    ).amount == Decimal("0.5")


def test_ac_057(conn, price_book):
    """AC-057: A provider failure never becomes a silent gap @spec:AC-057"""
    written = index_corpus(
        conn,
        [
            DocumentSpec(
                external_id="doc-1",
                title="Contrato",
                chunks=(ChunkSpec(0, "cláusula primeira", pages=(1, 1)),),
            ),
            DocumentSpec(
                external_id="doc-2",
                title="Aditivo",
                chunks=(ChunkSpec(0, "prazo de vigência", pages=(1, 1)),),
            ),
        ],
    )
    target = ModelStamp(model="qwen/qwen3-embedding-4b", dimension=TEST_DIMENSION)
    provider = FakeEmbeddings(fail_batches=(0,))

    outcome = reembed(
        conn, source=TEST_STAMP, target=target, embedder=provider.embedder(model=target.model), batch_size=1, max_inputs=2048
    )

    assert len(provider.bodies) == 2
    assert outcome.failed_documents == (written["doc-1"],)
    assert outcome.documents_written == (written["doc-2"],)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT document_id FROM chunks WHERE embed_model = %s AND embed_dimension = %s",
            (target.model, target.dimension),
        )
        assert [row[0] for row in cursor.fetchall()] == [written["doc-2"]]

    assert outcome.run.unembedded == 1
    assert outcome.run.embedded == 1
    assert outcome.complete is False

    ids = record_run(
        conn,
        price_book,
        client=TEST_CLIENT,
        run=outcome.run,
        model=target.model,
        provider="openrouter",
        reason=REASON_CONFIGURATION_CHANGED,
        document_ids=written,
    )
    assert [read_usage(conn, usage_id).outcome for usage_id in ids].count(OUTCOME_FAILED) == 1
