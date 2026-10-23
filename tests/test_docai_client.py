"""Unit-level coverage for src/processing/docai.py that is not itself a criterion.

AC-036 (the version is pinned) and AC-037 (the concurrency ceiling) are proven
in tests/test_spec_processing_docai.py — see CLAUDE.md. This file covers the
request body, how batches are formed, and the failures the client has to report
rather than absorb.
"""

import json

import pytest

from src.processing.docai import (
    MAX_CONCURRENT_BATCHES,
    MAX_DOCUMENTS_PER_BATCH,
    Batch,
    ChunkingOptions,
    InputDocument,
    ReaderError,
    plan_batches,
)
from tests.conftest import (
    ARTIFACT_BUCKET,
    DOCAI_PROCESSOR,
    DOCAI_VERSION,
    FakeDocumentAI,
    reader_payload,
)

OUTPUT = f"gs://{ARTIFACT_BUCKET}/runs/1"


def _documents(count: int) -> list[InputDocument]:
    return [
        InputDocument(f"doc-{n}", f"gs://{ARTIFACT_BUCKET}/doc-{n}.pdf", "application/pdf")
        for n in range(count)
    ]


def test_the_request_switches_table_and_image_description_on():
    # ASM-024. Off, annotations.description comes back empty and a table
    # reaches the index as a grid of numbers with nothing naming them — the
    # failure already confirmed in the house's earlier pipeline.
    docai = FakeDocumentAI(rounds_until_done=0)
    docai.reader().run(plan_batches(_documents(1), output_prefix=OUTPUT))

    layout = docai.submissions[0]["processOptions"]["layoutConfig"]
    assert layout["enableTableAnnotation"] is True
    assert layout["enableImageAnnotation"] is True


def test_the_request_asks_for_the_geometry_of_every_block():
    # AC-209: the module's own note that this reader carries no geometry was a
    # statement about our request, not about the reader — so every batch read
    # asks for it, whatever the pinned version does with the question.
    docai = FakeDocumentAI(rounds_until_done=0)
    docai.reader().run(plan_batches(_documents(1), output_prefix=OUTPUT))

    layout = docai.submissions[0]["processOptions"]["layoutConfig"]
    assert layout["returnBoundingBoxes"] is True


def test_the_request_carries_the_readers_two_cutting_controls():
    # ASM-021: the reader's own chunking is used, and its two controls are the
    # target size and whether ancestor headings are included. Both are sent.
    docai = FakeDocumentAI(rounds_until_done=0)
    reader = docai.reader(chunking=ChunkingOptions(size=512, cutter="local", include_ancestor_headings=False))
    reader.run(plan_batches(_documents(1), output_prefix=OUTPUT))

    chunking = docai.submissions[0]["processOptions"]["layoutConfig"]["chunkingConfig"]
    assert chunking == {"chunkSize": 512, "includeAncestorHeadings": False}


def test_the_request_asks_for_structure_and_never_for_review():
    docai = FakeDocumentAI(rounds_until_done=0)
    docai.reader().run(plan_batches(_documents(2), output_prefix=OUTPUT))

    body = docai.submissions[0]
    assert body["skipHumanReview"] is True
    assert body["documentOutputConfig"]["gcsOutputConfig"]["gcsUri"].startswith(OUTPUT)
    assert [entry["gcsUri"] for entry in body["inputDocuments"]["gcsDocuments"]["documents"]] == [
        f"gs://{ARTIFACT_BUCKET}/doc-0.pdf",
        f"gs://{ARTIFACT_BUCKET}/doc-1.pdf",
    ]


def test_every_request_carries_a_freshly_asked_for_token():
    # A first full read of this corpus is long enough for a token to expire,
    # and only the supplier can refresh one.
    docai = FakeDocumentAI(rounds_until_done=0)
    tokens = iter(["first", "second", "third", "fourth"])
    docai.reader(token=lambda: next(tokens)).run(plan_batches(_documents(1), output_prefix=OUTPUT))

    sent = [request.headers["authorization"] for request in docai.requests]
    assert sent == ["Bearer first", "Bearer second"]


def test_batches_are_never_larger_than_the_provider_accepts():
    batches = plan_batches(_documents(7), output_prefix=OUTPUT, max_documents_per_batch=3)

    assert [len(batch.documents) for batch in batches] == [3, 3, 1]
    # Each gets its own output location: two batches sharing one would make an
    # artifact's owner ambiguous.
    assert len({batch.output_prefix for batch in batches}) == 3


def test_a_deployment_cannot_ask_for_a_bigger_batch_than_the_provider_allows():
    batches = plan_batches(
        _documents(3), output_prefix=OUTPUT, max_documents_per_batch=MAX_DOCUMENTS_PER_BATCH + 500
    )

    assert len(batches) == 1
    with pytest.raises(ReaderError, match=str(MAX_DOCUMENTS_PER_BATCH)):
        Batch(documents=tuple(_documents(MAX_DOCUMENTS_PER_BATCH + 1)), output_prefix=OUTPUT)


def test_a_document_outside_object_storage_is_refused():
    # The batch path reads from object storage only (ASM-020).
    with pytest.raises(ReaderError, match="object storage"):
        InputDocument("doc", "/tmp/contrato.pdf", "application/pdf")
    with pytest.raises(ReaderError, match="object-storage"):
        Batch(documents=tuple(_documents(1)), output_prefix="/tmp/out")


def test_an_empty_batch_is_refused():
    with pytest.raises(ReaderError):
        Batch(documents=(), output_prefix=OUTPUT)


def test_a_failed_operation_is_reported_and_does_not_raise():
    # The run has to be able to continue and report; a raised exception here
    # would take the other batches with it.
    docai = FakeDocumentAI(rounds_until_done=0, operation_error="quota exhausted")
    run = docai.reader().run(plan_batches(_documents(2), output_prefix=OUTPUT))

    assert run.succeeded is False
    assert run.batches[0].detail == "quota exhausted"


def test_one_unreadable_document_does_not_condemn_the_others():
    unreadable = f"gs://{ARTIFACT_BUCKET}/doc-1.pdf"
    docai = FakeDocumentAI(rounds_until_done=0, failing_documents=[unreadable])
    run = docai.reader().run(plan_batches(_documents(3), output_prefix=OUTPUT))

    outcomes = {outcome.external_id: outcome for outcome in run.documents()}
    assert outcomes["doc-1"].succeeded is False
    assert outcomes["doc-0"].succeeded is True
    assert outcomes["doc-0"].output_uri.startswith(OUTPUT)
    assert outcomes["doc-1"].output_uri is None


def test_a_document_the_provider_said_nothing_about_reads_as_failed():
    # A document that is missing from the status list and reads as done is
    # worse than one that failed: nothing would ever notice it.
    docai = FakeDocumentAI(rounds_until_done=0)
    batch = Batch(documents=tuple(_documents(1)), output_prefix=OUTPUT)
    operation = docai.reader().submit(batch)
    ghost = Batch(
        documents=(InputDocument("ghost", f"gs://{ARTIFACT_BUCKET}/ghost.pdf", "application/pdf"),),
        output_prefix=OUTPUT,
    )
    outcome = docai.reader().poll(operation, ghost)

    assert outcome.documents[0].succeeded is False
    assert "no status" in outcome.documents[0].detail


def test_an_operation_that_never_finishes_is_abandoned_rather_than_polled_forever():
    docai = FakeDocumentAI(rounds_until_done=10**6)
    reader = docai.reader(poll_round_limit=4)

    with pytest.raises(ReaderError, match="never reported done"):
        reader.run(plan_batches(_documents(1), output_prefix=OUTPUT))
    assert len(docai.waits) == 4


def test_reading_one_document_inline_is_not_something_this_reader_offers():
    # The provider's synchronous endpoint keeps nothing, so a document read
    # through it could never have an artifact to re-cut (P-009). Batch is the
    # only route, and there is no switch that opens a second one.
    reader = FakeDocumentAI(response=reader_payload([], [])).reader()

    assert not hasattr(reader, "process")
    assert not any(field.startswith("allow_") for field in reader.__dataclass_fields__)


def test_a_recut_sends_the_document_back_inline_and_reads_no_page_again():
    # The one call that reaches `:process`, and it is not a second route: the
    # document travels back whole, so nothing is fetched and nothing is read.
    docai = FakeDocumentAI(response=reader_payload([], []))
    reader = docai.reader()
    document = {"documentLayout": {"blocks": []}}

    reader.recut(document, ChunkingOptions(size=400, cutter="docai"))

    body = json.loads(docai.requests[0].content)
    assert str(docai.requests[0].url).endswith(f"{DOCAI_VERSION}:process")
    assert body["inlineDocument"] == document
    assert "rawDocument" not in body
    assert body["processOptions"]["layoutConfig"]["chunkingConfig"]["chunkSize"] == 400
    assert body["processOptions"]["layoutConfig"]["enableTableAnnotation"] is True


def test_a_reader_without_a_processor_or_with_too_much_concurrency_is_refused():
    docai = FakeDocumentAI()

    with pytest.raises(ReaderError, match="processor"):
        docai.reader(processor_id="")
    with pytest.raises(ReaderError, match=str(MAX_CONCURRENT_BATCHES)):
        docai.reader(max_concurrent_batches=0)
    assert docai.reader(processor_id=DOCAI_PROCESSOR).version_name.endswith(DOCAI_VERSION)


def test_a_chunk_size_of_zero_is_refused():
    with pytest.raises(ReaderError, match="positive"):
        ChunkingOptions(size=0, cutter="local")
