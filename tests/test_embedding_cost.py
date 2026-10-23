"""Embedding cost and failure accounting (T-055).

AC-056 and AC-057 are proven in tests/test_spec_embedding_provider.py — see
CLAUDE.md for why the proof lives there. This file covers what the criteria do
not say: who the provider is when the model is reached through a gateway, what a
batch carrying several documents is attributed to, and what happens when nobody
has priced the model yet.
"""

from decimal import Decimal

import pytest

from src.embedding.batch import Group, embed_batches, plan_batches
from src.embedding.cost import (
    PROVIDER_UNKNOWN,
    provider_of,
    record_batch,
    record_embedding,
    record_query_embedding,
    record_run,
)
from src.prices import OPERATION_EMBED, UNIT_TOKEN
from src.store import upsert_document
from src.usage import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    REASON_CONFIGURATION_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_QUESTION_ASKED,
    UsageError,
    read_usage,
)
from tests.conftest import configured_amount, TEST_CLIENT, FakeEmbeddings

MODEL = "qwen/qwen3-embedding-8b"
PROVIDER = "openrouter"


def test_the_provider_is_who_the_invoice_comes_from():
    # Through a gateway that is the gateway — which is how the price key is
    # written, with the vendor's name inside the model's own slug.
    assert provider_of("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_of("https://api.openai.com/v1") == "openai"
    assert provider_of("") == PROVIDER_UNKNOWN


def test_a_call_records_its_tokens_its_price_its_model_and_its_reason(conn, price_book):
    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")

    usage_id = record_embedding(
        conn,
        price_book,
        client=TEST_CLIENT,
        document_id=document_id,
        model=MODEL,
        provider=PROVIDER,
        tokens=1_500,
        reason=REASON_NEW_DOCUMENT,
        duration_ms=Decimal("120"),
    )

    row = read_usage(conn, usage_id)
    assert row.operation == OPERATION_EMBED
    assert row.unit == UNIT_TOKEN
    assert row.units == 1_500
    assert row.model == MODEL
    assert row.provider == PROVIDER
    assert row.reason == REASON_NEW_DOCUMENT
    assert row.document_id == document_id
    assert row.unit_price == configured_amount(operation="embed", provider="openrouter", model=MODEL, unit="token")
    assert row.cost == configured_amount(operation="embed", provider="openrouter", model=MODEL, unit="token") * 1_500


def test_a_model_nobody_has_priced_is_recorded_consumed_and_unpriced(conn, empty_price_book):
    # ASM-075: the units are a fact and the price is an annotation. Losing the
    # price loses a field; refusing the row loses the history.
    usage_id = record_embedding(
        conn,
        empty_price_book,
        client=TEST_CLIENT,
        document_id=None,
        model="a/model-nobody-priced",
        provider=PROVIDER,
        tokens=42,
        reason=REASON_NEW_DOCUMENT,
        duration_ms=0,
    )

    row = read_usage(conn, usage_id)
    assert row.units == 42
    assert row.unit_price is None
    assert row.cost is None


def test_a_document_is_never_embedded_because_a_question_was_asked(conn, price_book):
    # It would put indexing cost in the bucket that answers "what do queries
    # cost us", which is the one comparison the reason column exists to make.
    with pytest.raises(UsageError) as excinfo:
        record_embedding(
            conn,
            price_book,
            client=TEST_CLIENT,
            document_id=None,
            model=MODEL,
            provider=PROVIDER,
            tokens=10,
            reason=REASON_QUESTION_ASKED,
            duration_ms=0,
        )

    assert "question was asked" in str(excinfo.value)


def test_embedding_a_question_carries_the_reason_a_caller_cannot_choose(conn, price_book):
    usage_id = record_query_embedding(
        conn,
        price_book,
        client=TEST_CLIENT,
        model=MODEL,
        provider=PROVIDER,
        tokens=12,
        duration_ms=Decimal("8"),
    )

    row = read_usage(conn, usage_id)
    assert row.reason == REASON_QUESTION_ASKED
    assert row.document_id is None


def test_a_batch_carrying_one_document_is_attributed_to_it(conn, price_book):
    document_id = upsert_document(conn, external_id="doc-1", title="Contrato")
    provider = FakeEmbeddings()
    run = embed_batches(
        plan_batches([Group(key="doc-1", texts=("um", "dois"))], batch_size=4, max_inputs=2048), provider.embedder()
    )

    usage_id = record_batch(
        conn,
        price_book,
        client=TEST_CLIENT,
        outcome=run.outcomes[0],
        model=MODEL,
        provider=PROVIDER,
        reason=REASON_CONFIGURATION_CHANGED,
        document_ids={"doc-1": document_id},
    )

    assert read_usage(conn, usage_id).document_id == document_id


def test_a_batch_carrying_several_documents_is_attributed_to_none_of_them(conn, price_book):
    # A cost attributed to the wrong document makes "what did this document cost
    # us" quietly wrong instead of visibly incomplete.
    ids = {
        "doc-1": upsert_document(conn, external_id="doc-1"),
        "doc-2": upsert_document(conn, external_id="doc-2"),
    }
    provider = FakeEmbeddings()
    run = embed_batches(
        plan_batches([Group(key="doc-1", texts=("um",)), Group(key="doc-2", texts=("dois",))], batch_size=4, max_inputs=2048),
        provider.embedder(),
    )

    usage_id = record_batch(
        conn,
        price_book,
        client=TEST_CLIENT,
        outcome=run.outcomes[0],
        model=MODEL,
        provider=PROVIDER,
        reason=REASON_NEW_DOCUMENT,
        document_ids=ids,
    )

    assert read_usage(conn, usage_id).document_id is None


def test_a_failed_batch_writes_a_row_too(conn, price_book):
    # AC-160. A provider that accepted a batch, counted the tokens and then
    # failed charged for them — and this is the only record, once the run's own
    # report is gone, that a run left pieces unembedded.
    provider = FakeEmbeddings(fail_batches=(0,))
    run = embed_batches(
        plan_batches([Group(key="doc-1", texts=("um",))], batch_size=4, max_inputs=2048), provider.embedder()
    )

    usage_id = record_batch(
        conn,
        price_book,
        client=TEST_CLIENT,
        outcome=run.outcomes[0],
        model=MODEL,
        provider=PROVIDER,
        reason=REASON_NEW_DOCUMENT,
    )

    assert read_usage(conn, usage_id).outcome == OUTCOME_FAILED


def test_a_run_writes_one_row_per_request_including_the_ones_that_failed(conn, price_book):
    provider = FakeEmbeddings(fail_batches=(1,))
    run = embed_batches(
        plan_batches(
            [Group(key="doc-1", texts=("um",)), Group(key="doc-2", texts=("dois",))], batch_size=1, max_inputs=2048
        ),
        provider.embedder(),
    )

    ids = record_run(
        conn,
        price_book,
        client=TEST_CLIENT,
        run=run,
        model=MODEL,
        provider=PROVIDER,
        reason=REASON_NEW_DOCUMENT,
    )

    outcomes = [read_usage(conn, usage_id).outcome for usage_id in ids]
    assert outcomes == [OUTCOME_SUCCEEDED, OUTCOME_FAILED]
