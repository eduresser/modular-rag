from decimal import Decimal

import pytest

from src.fingerprint import stage_fingerprints
from src.prices import PriceBook
from src.reindex import (
    CHARACTERS_PER_TOKEN,
    ReindexError,
    ReindexOutcome,
    ReindexPlan,
    estimate_reindex,
    plan_reindex,
    reindex,
)
from src.reprocess import ACTION_READ, ACTION_RECUT, ACTION_REEMBED, stale_states, stored_state
from tests.conftest import (
    TEST_DIMENSION,
    FakeEmbeddings,
    configured_prices,
    contract_payload,
    ingest_config,
    ingest_world,
    pdf_file,
)

CLAUSES = ("Cláusula primeira: do objeto da locação.", "Cláusula segunda: do prazo.")


def _indexed(monkeypatch, conn, names=("doc-1",)):
    from src.ingest import ingest_documents

    world = ingest_world(
        monkeypatch,
        files=[pdf_file(name, content=f"%PDF {name}".encode()) for name in names],
        outputs={name: contract_payload(clauses=CLAUSES) for name in names},
    )
    ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))
    return world


def _usage(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, operation, units FROM usage_events ORDER BY id")
        return cursor.fetchall()


def _document_id(conn, external_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", (external_id,))
        return int(cursor.fetchone()[0])


def _contents(conn, document_id):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT content FROM chunks WHERE document_id = %s ORDER BY ordinal", (document_id,)
        )
        return [row[0] for row in cursor.fetchall()]


def _stale(monkeypatch, world, **env):
    return ingest_world(
        monkeypatch,
        files=list(world.drive.files.values()),
        outputs=world.docai.outputs,
        storage=world.storage,
        **env,
    )


def test_a_current_archive_has_nothing_to_reprocess(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn)

    plan = plan_reindex(conn, world.config, price_book)

    assert plan.documents == ()
    assert plan.estimate.documents == 0
    assert plan.estimate.cost == 0
    assert plan.started is False
    assert "nothing to reprocess" in plan.estimate.summary()


def test_the_estimate_splits_the_corpus_where_the_bill_will(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn, names=("doc-1", "doc-2"))
    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="40")

    estimate = estimate_reindex(
        conn, moved, price_book, states=stale_states(conn, stage_fingerprints(moved).whole)
    )

    assert estimate.documents == 2
    assert estimate.for_action(ACTION_RECUT).documents == 2
    assert estimate.for_action(ACTION_READ) is None
    assert estimate.for_action(ACTION_REEMBED) is None
    assert estimate.pages > 0
    assert estimate.tokens > 0


def test_a_changed_reader_is_estimated_as_a_read_and_a_changed_model_as_an_embed(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    reader_moved = ingest_config(
        monkeypatch, RAG_DOCAI_PROCESSOR_VERSION="pretrained-layout-parser-v1.5"
    )
    model_moved = ingest_config(monkeypatch, RAG_EMBED_MODEL="google/gemini-embedding-001")

    reading = estimate_reindex(
        conn, reader_moved, price_book,
        states=stale_states(conn, stage_fingerprints(reader_moved).whole),
    )
    embedding = estimate_reindex(
        conn, model_moved, price_book,
        states=stale_states(conn, stage_fingerprints(model_moved).whole),
    )

    assert reading.for_action(ACTION_READ).documents == 1
    assert embedding.for_action(ACTION_REEMBED).documents == 1
    assert embedding.for_action(ACTION_REEMBED).pages == 0
    assert reading.cost > embedding.cost


def test_the_estimate_uses_the_prices_the_usage_rows_will_record(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn)
    moved = ingest_config(monkeypatch, RAG_EMBED_EXTRA_PARAMS='{"user": "example-user"}')
    states = stale_states(conn, stage_fingerprints(moved).whole)

    estimate = estimate_reindex(conn, moved, price_book, states=states)

    price = price_book.price_for(
        operation="embed", provider="openrouter", model="qwen/qwen3-embedding-8b", unit="token"
    )
    assert estimate.for_action(ACTION_REEMBED).documents == 1
    assert estimate.cost == price.cost_of(estimate.tokens)


def test_an_operation_nobody_priced_costs_nothing_and_the_forecast_says_so_plainly(
    monkeypatch, conn, empty_price_book
):
    world = _indexed(monkeypatch, conn)
    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="40")

    estimate = estimate_reindex(
        conn, moved, empty_price_book, states=stale_states(conn, stage_fingerprints(moved).whole)
    )

    assert estimate.cost == 0
    assert estimate.documents == 1
    assert "floor" not in estimate.summary()
    assert "no configured price" not in estimate.summary()


def test_what_the_archive_keeps_is_a_null_so_free_and_unpriced_stay_different(
    monkeypatch, conn, empty_price_book
):
    from src.usage import REASON_CONFIGURATION_CHANGED, priced

    unpriced = priced(
        empty_price_book, operation="embed", reason=REASON_CONFIGURATION_CHANGED,
        provider="openrouter", model="modelo/sem-preco", unit="token", units=1_000,
        duration_ms=12,
    )

    assert unpriced.unit_price is None
    assert unpriced.units == 1_000


def test_a_cut_that_runs_on_our_own_cpu_is_priced_at_zero_rather_than_left_out(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="40")

    estimate = estimate_reindex(
        conn, moved, price_book, states=stale_states(conn, stage_fingerprints(moved).whole)
    )
    cut = estimate.for_action(ACTION_RECUT)

    assert cut is not None and cut.documents == 1
    assert price_book.price_for(
        operation="chunk", provider="internal", model="local", unit="page"
    ).amount == 0


def test_the_token_figure_is_an_estimate_over_the_text_already_stored(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn)
    moved = ingest_config(monkeypatch, RAG_EMBED_EXTRA_PARAMS='{"user": "example-user"}')

    estimate = estimate_reindex(
        conn, moved, price_book, states=stale_states(conn, stage_fingerprints(moved).whole)
    )

    characters = sum(len(clause) for clause in CLAUSES)
    assert estimate.tokens == -(-characters // CHARACTERS_PER_TOKEN)


def test_a_plan_can_be_narrowed_to_the_documents_somebody_complained_about(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn, names=("doc-1", "doc-2", "doc-3"))
    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="40")
    one = stored_state(conn, stale_states(conn, stage_fingerprints(moved).whole)[0].document_id)

    plan = plan_reindex(conn, moved, price_book, document_ids=[one.document_id])

    assert plan.documents == (one.document_id,)
    assert plan.estimate.documents == 1
    assert plan.staleness.affected == 3


def test_an_unconfirmed_request_returns_a_plan_and_touches_nothing(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn)
    stale = _stale(monkeypatch, world, RAG_CHUNK_SIZE="40")
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        (before,) = cursor.fetchone()

    plan = reindex(conn, stale.pipeline, price_book)

    assert isinstance(plan, ReindexPlan) and plan.started is False
    assert stale.reads == 0 and stale.embedded == 0
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        assert cursor.fetchone()[0] == before


def test_a_confirmed_request_runs_the_work_under_the_operator_s_own_reason(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    stale = _stale(monkeypatch, world, RAG_EMBED_MODEL="openai/text-embedding-3-large")

    outcome = reindex(conn, stale.pipeline, price_book, confirm=True)

    assert isinstance(outcome, ReindexOutcome)
    assert outcome.plan.started is True
    assert len(outcome.run.indexed) == 1
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT DISTINCT reason FROM usage_events WHERE operation IN ('chunk', 'embed') "
            "AND id > (SELECT min(id) FROM usage_events)"
        )
        reasons = {row[0] for row in cursor.fetchall()}
    assert "operator_requested" in reasons


def test_confirming_a_reprocessing_of_nothing_is_refused(monkeypatch, conn, price_book):
    world = _indexed(monkeypatch, conn)

    with pytest.raises(ReindexError) as excinfo:
        reindex(conn, world.pipeline, price_book, confirm=True)

    assert "nothing to reprocess" in str(excinfo.value)


def test_a_confirmed_reprocessing_reads_from_the_source_when_the_reader_moved(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    stale = _stale(
        monkeypatch, world, RAG_DOCAI_PROCESSOR_VERSION="pretrained-layout-parser-v1.5"
    )

    plan = reindex(conn, stale.pipeline, price_book)
    assert plan.estimate.for_action(ACTION_READ).documents == 1

    outcome = reindex(conn, stale.pipeline, price_book, confirm=True)

    assert stale.reads == 1
    assert len(outcome.run.indexed) == 1
    assert stored_state(conn, outcome.run.results[0].document_id).reader_version == (
        "pretrained-layout-parser-v1.5"
    )


def _sweep(conn, world):
    from src.ingest import ingest_documents

    return ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))


def test_a_setting_that_changes_no_piece_is_cut_off_before_the_embedding(monkeypatch, conn):
    world = _indexed(monkeypatch, conn)
    moved = _stale(monkeypatch, world, RAG_CHUNK_SIZE="40")

    run = _sweep(conn, moved)

    assert len(run.cut_off) == 1
    assert run.cut_off[0].action == ACTION_RECUT
    assert moved.embedded == 0
    assert moved.reads == 0
    assert _contents(conn, _document_id(conn, "doc-1")) == list(CLAUSES)


def test_a_cut_off_document_records_the_configuration_it_was_cut_off_under(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    moved = _stale(monkeypatch, world, RAG_CHUNK_SIZE="40")

    _sweep(conn, moved)

    assert stored_state(conn, _document_id(conn, "doc-1")).fingerprint == (
        moved.pipeline.fingerprints.whole
    )
    assert plan_reindex(conn, moved.config, price_book).documents == ()


def test_a_cut_that_really_changes_the_pieces_is_not_cut_off(monkeypatch, conn):
    world = _indexed(monkeypatch, conn)
    moved = _stale(monkeypatch, world)

    import src.processing.to_chunks as to_chunks

    original = to_chunks._content_of
    monkeypatch.setattr(
        to_chunks,
        "_content_of",
        lambda reader_chunk, sources: f"{original(reader_chunk, sources)}\n\n[remapped]",
    )
    monkeypatch.setattr("src.fingerprint.cut_version", lambda _module: "a-changed-mapping")

    run = _sweep(conn, moved)

    assert run.cut_off == ()
    assert moved.embedded > 0


def test_a_new_embedding_stamp_is_never_cut_off_however_identical_the_text(
    monkeypatch, conn
):
    world = _indexed(monkeypatch, conn)
    moved = _stale(monkeypatch, world, RAG_EMBED_MODEL="openai/text-embedding-3-large")

    run = _sweep(conn, moved)

    assert run.cut_off == ()
    assert moved.embedded > 0


def test_an_operator_who_forced_a_read_is_still_spared_the_embedding(
    monkeypatch, conn, price_book
):
    world = _indexed(monkeypatch, conn)
    moved = _stale(monkeypatch, world, RAG_CHUNK_SIZE="40")

    outcome = reindex(conn, moved.pipeline, price_book, confirm=True)

    assert moved.reads == 1
    assert len(outcome.run.cut_off) == 1
    assert moved.embedded == 0
