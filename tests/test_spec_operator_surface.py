
from __future__ import annotations

from decimal import Decimal

import psycopg
import pytest

from src.usecases.inventory import (
    DEFAULT_PAGE,
    MAX_PAGE,
    InventoryError,
    page_size,
)
from tests.conftest import FakeQueryEmbedder

def test_ac_254(api_world, served_corpus):
    """AC-254: The archive lists the documents it holds @spec:AC-254"""
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    listing = client.get("/v1/documents").json()

    assert listing["documents"]
    for document in listing["documents"]:
        assert document["id"] and document["document"] and document["title"]
        assert document["state"] in {"indexed", "gone"}
        assert {"link", "location", "piece_count", "indexed_at", "last_seen_at"} <= set(document)

    assert listing["total_count"] >= len(listing["documents"])
    assert listing["next_cursor"] is None, "this corpus fits in one page"

    one = client.get("/v1/documents", params={"limit": 1}).json()
    assert len(one["documents"]) == 1
    assert one["total_count"] == listing["total_count"]
    assert one["next_cursor"], "a listing that left documents out has to say how to continue"

    seen = [d["document"] for d in one["documents"]]
    cursor = one["next_cursor"]
    while cursor:
        page = client.get("/v1/documents", params={"limit": 1, "cursor": cursor}).json()
        seen += [d["document"] for d in page["documents"]]
        cursor = page["next_cursor"]
    assert seen == [d["document"] for d in listing["documents"]]
    assert len(seen) == len(set(seen))

    refused = client.get("/v1/documents", params={"limit": MAX_PAGE + 1})
    assert refused.status_code == 422
    assert str(MAX_PAGE) in refused.json()["error"]
    assert page_size(None) == DEFAULT_PAGE
    with pytest.raises(InventoryError):
        page_size(0)

def test_ac_255(api_world, served_corpus, schema_dsn):
    """AC-255: A document says what state it is in, and why @spec:AC-255"""
    with psycopg.connect(schema_dsn) as setup, setup.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", ("drive-rescisao",))
        (read_id,) = cursor.fetchone()
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", ("drive-reajuste",))
        (failed_id,) = cursor.fetchone()
        cursor.execute(
            "INSERT INTO document_reads (document_id, state, reader, reader_version, page_count) "
            "VALUES (%s, 'read', 'docai-layout-parser', 'pretrained-layout-parser-v1.0-2024-06-03', 12)",
            (read_id,),
        )
        cursor.execute(
            "INSERT INTO document_reads (document_id, state, reader, reader_version, failure_reason) "
            "VALUES (%s, 'failed', 'docai-layout-parser', "
            "'pretrained-layout-parser-v1.0-2024-06-03', %s)",
            (failed_id, "the layout parser does not accept text/plain"),
        )
        setup.commit()

    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    read = client.get("/v1/documents/drive-rescisao/status").json()
    assert read["document"] == "drive-rescisao"
    assert read["read"]["state"] == "read"
    assert read["read"]["reader_version"] == "pretrained-layout-parser-v1.0-2024-06-03"
    assert read["read"]["page_count"] == 12
    assert read["read"]["failure_reason"] is None

    refused = client.get("/v1/documents/drive-reajuste/status").json()
    assert refused["read"]["state"] == "failed"
    assert "text/plain" in refused["read"]["failure_reason"]

    absent = client.get("/v1/documents/drive-nao-existe/status").json()
    assert absent["status"] == "nothing_matched"
    assert absent["id"] == "drive-nao-existe"

def _record_run(dsn, *, kind, status, counters=None):
    from src.sync_ledger import RunCounters, finish_run, start_run

    with psycopg.connect(dsn) as conn:
        run_id = start_run(conn, client="example", kind=kind)
        finish_run(
            conn,
            run_id,
            status=status,
            counters=counters or RunCounters(documents_seen=3, documents_added=1),
            sweep_completed=status == "completed",
        )
        conn.commit()
    return run_id


def test_ac_256(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-256: The synchronisations are listed with what each one did @spec:AC-256"""
    from src.sync_ledger import RunCounters

    _record_run(schema_dsn, kind="sweep", status="completed")
    _record_run(
        schema_dsn,
        kind="incremental",
        status="completed",
        counters=RunCounters(
            documents_seen=9,
            documents_removed=1,
            access_lost=2,
            documents_unproven=1,
        ),
    )
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    history = client.get("/v1/sync").json()

    assert len(history["runs"]) == 2
    for run in history["runs"]:
        assert run["run_id"] and run["client"] == "example"
        assert run["kind"] in {"sweep", "incremental"}
        assert run["status"] == "completed"
        assert run["started_at"] and run["finished_at"]
        assert run["counters"]["documents_seen"]

    newest = history["runs"][0]
    assert newest["counters"]["documents_unproven"] == 1
    assert newest["counters"]["access_lost"] == 2

    assert "detail" not in newest

def test_ac_257(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-257: A run in flight is distinguishable from the last one that finished @spec:AC-257"""
    from src.sync_history import start_exclusive_run

    finished = _record_run(schema_dsn, kind="sweep", status="completed")
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    quiet = client.get("/v1/sync").json()
    assert quiet["current"] is None
    assert quiet["last"]["run_id"] == finished

    with psycopg.connect(schema_dsn) as holder:
        running = start_exclusive_run(holder, client="example", kind="incremental")
        holder.commit()

        state = client.get("/v1/sync").json()

        assert state["current"]["run_id"] == running
        assert state["current"]["finished_at"] is None
        assert state["last"]["run_id"] == finished
        assert state["last"]["run_id"] != state["current"]["run_id"]

        assert state["runs"][0]["run_id"] == running

@pytest.fixture
def clean_ledgers(schema_dsn):

    def wipe():
        with psycopg.connect(schema_dsn) as conn, conn.cursor() as cursor:
            cursor.execute("DELETE FROM sync_run_lock")
            cursor.execute("DELETE FROM sync_requests")
            cursor.execute("DELETE FROM sync_runs")
            cursor.execute("DELETE FROM usage_events")
            conn.commit()

    wipe()
    yield
    wipe()


def _record_usage(dsn, *, operation, provider, model, unit, units, price, reason="new_document"):
    from decimal import Decimal

    from src.usage import UsageEvent, record_usage

    with psycopg.connect(dsn) as conn:
        record_usage(
            conn,
            UsageEvent(
                operation=operation,
                reason=reason,
                provider=provider,
                model=model,
                unit=unit,
                units=Decimal(units),
                duration_ms=Decimal("120"),
                unit_price=None if price is None else Decimal(price),
            ),
            client="example",
        )
        conn.commit()


def test_ac_258(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-258: The cost is readable by operation, by provider and by reason @spec:AC-258"""
    _record_usage(
        schema_dsn, operation="read", provider="google", model="layout-parser",
        unit="page", units="543", price="0.01",
    )
    _record_usage(
        schema_dsn, operation="embed", provider="openrouter", model="qwen/qwen3-embedding-8b",
        unit="token", units="1000000", price="0.00000001",
    )
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    for grouping, expected in (
        ("operation", {"read", "embed"}),
        ("provider", {"google", "openrouter"}),
        ("reason", {"new_document"}),
    ):
        bill = client.get("/v1/usage", params={"by": grouping}).json()
        assert bill["by"] == grouping
        keys = {line["key"] for line in bill["slices"]}
        assert expected <= keys, grouping

        for line in bill["slices"]:
            assert Decimal(line["units"]) > 0
            assert Decimal(line["cost"]) >= 0
            assert line["currency"]

    by_operation = client.get("/v1/usage", params={"by": "operation"}).json()
    read = next(line for line in by_operation["slices"] if line["key"] == "read")
    assert Decimal(read["cost"]) == Decimal("5.43")
    assert Decimal(by_operation["total"]["cost"]) == Decimal("5.44")

    refused = client.get("/v1/usage", params={"by": "vendor"})
    assert refused.status_code == 422
    assert "operation" in refused.json()["error"]

def test_ac_259(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-259: An unpriced unit is reported, never counted as free @spec:AC-259"""
    _record_usage(
        schema_dsn, operation="read", provider="google", model="layout-parser",
        unit="page", units="100", price="0.01",
    )
    _record_usage(
        schema_dsn, operation="rerank", provider="cohere", model="rerank-v3",
        unit="request", units="40", price=None,
    )
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    bill = client.get("/v1/usage", params={"by": "operation"}).json()
    rerank = next(line for line in bill["slices"] if line["key"] == "rerank")

    assert Decimal(rerank["units"]) == Decimal("40")
    assert rerank["complete"] is False
    assert bill["total"]["complete"] is False

    assert Decimal(rerank["unpriced_units"]) == Decimal("40")
    assert rerank["unpriced_events"] == 1
    assert Decimal(bill["total"]["unpriced_units"]) == Decimal("40")

    read = next(line for line in bill["slices"] if line["key"] == "read")
    assert read["complete"] is True
    assert Decimal(read["cost"]) == Decimal("1.00")

def _record_search(dsn, *, document_ids, effort=40, duration_ms="120", outcome="found"):
    from decimal import Decimal as _D

    from src.search_events import SearchEvent, record_search

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM chunks WHERE document_id = ANY(%s) ORDER BY id",
                (list(document_ids),),
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]
        record_search(
            conn,
            SearchEvent(
                caller="v1",
                question="rescisão antecipada",
                dense_weight=0.5,
                lexical_weight=0.5,
                candidate_depth=50,
                result_count=10,
                index_search_effort=effort,
                duration_ms=_D(duration_ms),
                outcome=outcome,
                document_ids=document_ids,
                chunk_ids=chunk_ids,
            ),
            client="example",
        )
        conn.commit()


def test_ac_260(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-260: The documents nobody has ever been shown are listable @spec:AC-260"""
    from src.processing.to_chunks import NO_LOCATOR_REASON


    shown = served_corpus["drive-rescisao"]
    never = served_corpus["drive-reajuste"]
    _record_usage(
        schema_dsn, operation="read", provider="google", model="layout-parser",
        unit="page", units="12", price="0.01",
    )
    with psycopg.connect(schema_dsn) as setup, setup.cursor() as cursor:
        cursor.execute(
            "INSERT INTO documents (external_id, title) VALUES (%s, %s) RETURNING id",
            ("drive-sem-locador", "Anexo digitalizado sem cabeçalho"),
        )
        (uncitable_id,) = cursor.fetchone()
        cursor.execute(
            "INSERT INTO document_reads (document_id, state, reader, reader_version, failure_reason) "
            "VALUES (%s, 'failed', 'docai-layout-parser', 'v1', %s)",
            (uncitable_id, f"{NO_LOCATOR_REASON}: none of its 4 pieces carry a page or a heading"),
        )
        setup.commit()
    _record_search(schema_dsn, document_ids=[shown])

    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    reach = client.get("/v1/corpus/reach").json()

    never_returned = {d["document"] for d in reach["never_returned"]["documents"]}
    assert "drive-reajuste" in never_returned
    assert "drive-rescisao" not in never_returned, "a document that was returned is not unused"
    assert reach["never_returned"]["count"] == len(reach["never_returned"]["documents"])

    uncitable = {d["document"] for d in reach["uncitable"]["documents"]}
    assert uncitable == {"drive-sem-locador"}
    assert NO_LOCATOR_REASON in reach["uncitable"]["documents"][0]["failure_reason"]

    assert "drive-sem-locador" in never_returned
    assert never_returned - uncitable, "not every unused document is a broken one"
    assert "drive-reajuste" not in uncitable

    reajuste = next(
        d for d in reach["never_returned"]["documents"] if d["document"] == "drive-reajuste"
    )
    assert "cost" in reajuste and "currency" in reajuste and "complete" in reajuste

def test_ac_261(api_world, served_corpus, schema_dsn, clean_ledgers):
    """AC-261: What a search cost in time is readable @spec:AC-261"""
    shown = served_corpus["drive-rescisao"]

    for duration in ("400", "420", "380"):
        _record_search(schema_dsn, document_ids=[shown], effort=10, duration_ms=duration)
    for duration in ("40", "45", "38"):
        _record_search(schema_dsn, document_ids=[shown], effort=100, duration_ms=duration)

    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    latency = client.get("/v1/searches/latency").json()

    groups = latency["by_index_search_effort"]
    efforts = [group["index_search_effort"] for group in groups]
    assert 10 in efforts and 100 in efforts
    assert efforts == sorted(efforts), "ascending by effort: the step is only legible in the order the knob turns"
    for group in groups:
        assert group["searches"] >= 1
        assert Decimal(group["median_ms"]) > 0
        assert Decimal(group["slowest_ms"]) >= Decimal(group["median_ms"])

    assert latency["largest_step"]["below_effort"] == 10
    assert latency["largest_step"]["above_effort"] == 100
    assert Decimal(latency["largest_step"]["factor"]) > 1

    assert "question" not in repr(latency)

def test_ac_262(api_world, served_corpus, schema_dsn):
    """AC-262: The service reports what is serving @spec:AC-262"""
    from src.ingest_idempotency import APP_VERSION
    from tests.conftest import TEST_STAMP

    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))

    answer = client.get("/v1/serving").json()

    assert answer["active"]["embed_model"] == TEST_STAMP.model
    assert answer["active"]["embed_dimension"] == TEST_STAMP.dimension
    assert answer["configured"]["embed_model"] == service.config.embed.model
    assert answer["app_version"] == APP_VERSION

    assert answer["agrees"] is True

    crossed, _ = api_world(
        embedder=FakeQueryEmbedder(degrees=0), RAG_EMBED_MODEL="openai/text-embedding-3-large"
    )
    disagreeing = crossed.get("/v1/serving").json()
    assert disagreeing["agrees"] is False
    assert disagreeing["configured"]["embed_model"] == "openai/text-embedding-3-large"
    assert disagreeing["active"]["embed_model"] == TEST_STAMP.model

    with psycopg.connect(schema_dsn) as reset, reset.cursor() as cursor:
        cursor.execute("DELETE FROM active_model_version")
        reset.commit()
    fresh = client.get("/v1/serving").json()
    assert fresh["active"] is None
    assert fresh["agrees"] is True

def test_ac_263(api_world, served_corpus, caplog):
    """AC-263: A failure carries an identifier that is also in the logs @spec:AC-263"""
    import logging

    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    with caplog.at_level(logging.WARNING, logger="src.main"):
        refused_knob = client.post(
            "/v1/search", json={"question": "rescisão", "limit": 10_000}
        )
        refused_page = client.get("/v1/documents", params={"limit": MAX_PAGE + 1})
        unauthenticated = client.post(
            "/v1/search", json={"question": "rescisão"}, headers={"Authorization": "Bearer wrong"}
        )

    for response, status in (
        (refused_knob, 422),
        (refused_page, 422),
        (unauthenticated, 401),
    ):
        assert response.status_code == status
        identifier = response.json()["request_id"]
        assert identifier and len(identifier) == 32

        assert response.headers["X-Request-Id"] == identifier

        assert any(identifier in record.getMessage() for record in caplog.records), (
            f"the {status} response carries {identifier}, which appears in no log line — "
            "an identifier that names nothing anybody can find is worse than none"
        )

    identifiers = {
        refused_knob.json()["request_id"],
        refused_page.json()["request_id"],
        unauthenticated.json()["request_id"],
    }
    assert len(identifiers) == 3

    found = client.post("/v1/search", json={"question": "rescisão antecipada"})
    assert found.status_code == 200
    assert "request_id" not in found.json()
    assert found.headers["X-Request-Id"]
