"""Batching (T-052): how much travels together, and what a failure costs.

AC-055 and AC-057 are proven in tests/test_spec_embedding_provider.py — see
CLAUDE.md for why the proof lives there. This file covers the planner's edges:
a document bigger than a request, a batch that fails in the middle of a run, and
a document split across two requests of which only one came back.
"""

import pytest

from src.embedding.batch import (
    BatchError,
    Group,
    embed_batches,
    plan_batches,
    split_oversized,
)
from tests.conftest import FakeEmbeddings, vector_for


def _group(key: str, count: int) -> Group:
    return Group(key=key, texts=tuple(f"{key}-{position}" for position in range(count)))


def test_whole_documents_share_a_request_up_to_the_size():
    batches = plan_batches([_group("a", 3), _group("b", 3), _group("c", 3)], batch_size=6, max_inputs=2048)

    assert [batch.keys for batch in batches] == [("a", "b"), ("c",)]


def test_a_document_is_never_cut_to_make_a_batch_come_out_even():
    # A document that FITS the size is never cut merely to fill a request to
    # the brim: two of four go in two requests of four, not one of six and one
    # of two. Only a document larger than the size is split, and then into
    # parts of that size — see the test below.
    batches = plan_batches([_group("a", 4), _group("b", 4)], batch_size=6, max_inputs=2048)

    assert [len(batch) for batch in batches] == [4, 4]
    assert all(len(batch.groups) == 1 for batch in batches)


def test_a_document_larger_than_the_batch_size_is_split_into_requests_of_that_size():
    # This used to assert the opposite — one request of 20 — on the grounds
    # that cutting a document across requests would trade the "whole or not at
    # all" guarantee for a preference. It does not: that guarantee is
    # `EmbeddingRun.failed_keys`, which drops every part of a document if any
    # part failed, and it holds however many requests the document took.
    #
    # What the old rule left unbounded was the request itself. Against this
    # deployment's own provider, 200 inputs answered in 23.6s and 410 came back
    # `HTTP 504: Worker upstream fetch timed out after 60000ms` — so the
    # binding limit is the gateway's minute, not the 2,048-input ceiling, and
    # the only knob an operator has for it is this one.
    batches = plan_batches([_group("big", 20)], batch_size=6, max_inputs=2048)

    assert [len(batch) for batch in batches] == [6, 6, 6, 2]
    assert all(group.partial for batch in batches for group in batch.groups)
    assert {group.key for batch in batches for group in batch.groups} == {"big"}


def test_a_document_larger_than_the_request_ceiling_is_split_and_says_so():
    # The one thing that overrides the rule, because the alternative is refusing
    # to index the document at all.
    parts = split_oversized(_group("huge", 2048 + 5), limit=2048)

    assert [len(part) for part in parts] == [2048, 5]
    assert all(part.partial for part in parts)
    assert {part.key for part in parts} == {"huge"}


def test_a_document_that_fits_is_not_marked_partial():
    assert split_oversized(_group("a", 3), limit=10) == (_group("a", 3),)


def test_a_batch_size_below_one_is_refused():
    with pytest.raises(BatchError):
        plan_batches([_group("a", 1)], batch_size=0, max_inputs=2048)


def test_a_group_with_no_pieces_or_no_document_is_refused():
    with pytest.raises(BatchError):
        Group(key="a", texts=())
    with pytest.raises(BatchError):
        Group(key="  ", texts=("texto",))


def test_the_vectors_come_back_cut_apart_by_document():
    provider = FakeEmbeddings()
    batches = plan_batches([_group("a", 2), _group("b", 2)], batch_size=4, max_inputs=2048)

    run = embed_batches(batches, provider.embedder())

    assert run.vectors()["a"] == (vector_for("a-0"), vector_for("a-1"))
    assert run.vectors()["b"] == (vector_for("b-0"), vector_for("b-1"))


def test_one_failed_request_does_not_stop_the_others():
    provider = FakeEmbeddings(fail_batches=(0,))
    batches = plan_batches([_group("a", 2), _group("b", 2)], batch_size=2, max_inputs=2048)

    run = embed_batches(batches, provider.embedder())

    assert run.failed_keys == ("a",)
    assert set(run.vectors()) == {"b"}
    assert run.succeeded is False


def test_a_failed_request_produces_no_vectors_at_all():
    # Never a placeholder: a stand-in vector is indistinguishable from a real
    # one once it is in the index, and it answers questions.
    provider = FakeEmbeddings(fail_batches=(0,))

    run = embed_batches(plan_batches([_group("a", 3)], batch_size=3, max_inputs=2048), provider.embedder())

    assert run.outcomes[0].vectors == ()
    assert run.vectors() == {}
    assert run.unembedded == 3
    assert run.embedded == 0


def test_the_run_counts_what_it_could_not_embed():
    provider = FakeEmbeddings(fail_batches=(1,))
    batches = plan_batches([_group("a", 2), _group("b", 5)], batch_size=2, max_inputs=2048)

    run = embed_batches(batches, provider.embedder())

    assert run.total == 7
    assert run.embedded == 2
    assert run.unembedded == 5


def test_a_split_document_that_partly_failed_is_dropped_whole():
    # Half an indexed contract answers a question as confidently as a whole one.
    provider = FakeEmbeddings(fail_batches=(1,))
    parts = split_oversized(_group("huge", 6), limit=3)
    batches = plan_batches(parts, batch_size=3, max_inputs=2048)

    run = embed_batches(batches, provider.embedder())

    assert run.failed_keys == ("huge",)
    assert run.vectors() == {}
    assert run.unembedded == 6


def test_a_failed_request_still_reports_how_long_it_took():
    # It lands on a usage row: a provider that accepted a batch, counted the
    # tokens and then failed still charged for the time.
    ticks = iter([0.0, 1.5])
    provider = FakeEmbeddings(fail_batches=(0,))

    run = embed_batches(
        plan_batches([_group("a", 1)], batch_size=1, max_inputs=2048), provider.embedder(), clock=lambda: next(ticks)
    )

    assert run.outcomes[0].duration_ms == 1500
    assert "unavailable" in run.outcomes[0].detail


def test_the_run_adds_up_the_tokens_of_the_requests_that_succeeded():
    provider = FakeEmbeddings()

    run = embed_batches(
        plan_batches([_group("a", 1), _group("b", 1)], batch_size=1, max_inputs=2048), provider.embedder()
    )

    assert len(provider.bodies) == 2
    assert run.tokens == 2
