import httpx
import pytest

from src.db import DatabaseUnavailable
from src.embedding.batch import BatchError
from src.embedding.client import EmbeddingError
from src.embedding.pinning import ProviderSubstituted
from src.ingest import _not_embedded
from src.ingest_errors import (
    FAILURE_PERMANENT,
    FAILURE_TRANSIENT,
    STAGE_CUT,
    STAGE_EMBED,
    STAGE_READ,
    FailureLog,
    NothingToIndex,
    ReadRefused,
    classify,
    describe,
    failure_of,
    isolate,
    mark_failed,
)
from src.models import ChunkError
from src.processing.artifact import ArtifactError, ChunksBeforeArtifact
from src.store import StoreError, upsert_document
from src.versions import UnknownStageVersion


@pytest.mark.parametrize(
    "error",
    [
        ReadRefused("is text/plain, and the reader does not accept text files"),
        NothingToIndex("doc-1 produced no piece to index"),
        ChunksBeforeArtifact("no kept reader output"),
        ChunkError("a chunk has to name the document it came from"),
        StoreError("piece 0 carries a 4-dimension vector"),
        BatchError("a batch has to hold at least one piece"),
        UnknownStageVersion("this build no longer declares that version"),
        ProviderSubstituted("the router served another model"),
    ],
)
def test_a_failure_about_the_document_is_permanent(error):
    assert classify(error) == FAILURE_PERMANENT


@pytest.mark.parametrize(
    "error",
    [
        DatabaseUnavailable("could not reach the archive"),
        httpx.ConnectTimeout("timed out"),
        httpx.ConnectError("no route to host"),
        ArtifactError("could not write gs://bucket/parsed/doc-1.json"),
        EmbeddingError("the provider is unavailable"),
    ],
)
def test_a_failure_about_the_moment_is_transient(error):
    assert classify(error) == FAILURE_TRANSIENT


def test_a_permanent_subclass_of_a_transient_type_is_still_permanent():
    assert issubclass(ChunksBeforeArtifact, ArtifactError)
    assert classify(ChunksBeforeArtifact("no kept output")) == FAILURE_PERMANENT
    assert classify(ArtifactError("the bucket was unreachable")) == FAILURE_TRANSIENT


def test_a_provider_substitution_is_permanent_even_though_it_is_an_embedding_error():
    assert issubclass(ProviderSubstituted, EmbeddingError)
    assert classify(EmbeddingError("the provider is unavailable")) == FAILURE_TRANSIENT
    assert classify(ProviderSubstituted("served another model")) == FAILURE_PERMANENT


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (403, FAILURE_PERMANENT),
        (404, FAILURE_PERMANENT),
        (400, FAILURE_PERMANENT),
        (408, FAILURE_TRANSIENT),
        (429, FAILURE_TRANSIENT),
        (500, FAILURE_TRANSIENT),
        (503, FAILURE_TRANSIENT),
    ],
)
def test_a_status_code_decides_when_the_exception_type_does_not(status, kind):
    request = httpx.Request("GET", "https://example.invalid/x")
    error = httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )

    assert classify(error) == kind
    assert classify(RuntimeError(f"the reader refused a batch: {status} something")) == kind


def test_an_unrecognised_failure_is_transient_rather_than_abandoned():
    assert classify(ValueError("something nobody has classified")) == FAILURE_TRANSIENT


def test_the_reason_names_the_step_and_the_type_not_only_the_message():
    reason = describe(ReadRefused("is 900 pages, above the 500-page limit"), stage=STAGE_READ)

    assert reason.startswith("read: ")
    assert "900 pages" in reason
    assert "ReadRefused" in reason


def test_an_exception_with_no_message_still_produces_a_reason():
    assert "ReadRefused" in describe(ReadRefused(), stage=STAGE_READ)


def test_a_failure_record_carries_what_a_retry_loop_needs():
    failure = failure_of(
        external_id="doc-1", stage=STAGE_EMBED, error=httpx.ConnectTimeout("t"), document_id=7
    )

    assert (failure.external_id, failure.stage, failure.document_id) == ("doc-1", STAGE_EMBED, 7)
    assert failure.retryable is True


def test_a_stage_nobody_recognises_is_refused():
    with pytest.raises(ValueError) as excinfo:
        failure_of(external_id="doc-1", stage="rerank", error=ValueError("x"))

    assert "rerank" in str(excinfo.value)


def test_the_log_counts_and_separates_what_can_be_tried_again():
    log = FailureLog()

    log.add(failure_of(external_id="doc-1", stage=STAGE_READ, error=ReadRefused("text/plain")))
    log.add(failure_of(external_id="doc-2", stage=STAGE_EMBED, error=httpx.ConnectTimeout("t")))

    assert log.count == 2
    assert log.documents == ("doc-1", "doc-2")
    assert [f.external_id for f in log.permanent] == ["doc-1"]
    assert [f.external_id for f in log.transient] == ["doc-2"]


def test_isolate_turns_one_document_s_failure_into_a_record_and_carries_on():
    log = FailureLog()
    settled = []

    for external_id in ("doc-1", "doc-2", "doc-3"):
        with isolate(log, external_id=external_id, stage=STAGE_CUT, on_failure=settled.append):
            if external_id == "doc-2":
                raise ChunkError("chunk 0 carries no text")

    assert log.documents == ("doc-2",)
    assert [f.external_id for f in settled] == ["doc-2"]
    assert settled[0].kind == FAILURE_PERMANENT


def test_isolate_does_not_swallow_an_operator_stopping_the_run():
    log = FailureLog()

    with pytest.raises(KeyboardInterrupt):
        with isolate(log, external_id="doc-1", stage=STAGE_CUT):
            raise KeyboardInterrupt

    assert log.count == 0


def test_a_failed_read_writes_the_reader_s_own_record_too(conn):
    document_id = upsert_document(conn, external_id="doc-1")

    failure = mark_failed(
        conn,
        external_id="doc-1",
        stage=STAGE_READ,
        error=ReadRefused("is text/plain, and the reader does not accept text files"),
        document_id=document_id,
        reader="docai-layout-parser",
        reader_version="pretrained-layout-parser-v1.0-2024-06-03",
    )

    assert failure.kind == FAILURE_PERMANENT
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT state, failure_reason, reader_version FROM document_reads WHERE document_id = %s",
            (document_id,),
        )
        state, reason, version = cursor.fetchone()
    assert state == "failed"
    assert "text/plain" in reason
    assert version == "pretrained-layout-parser-v1.0-2024-06-03"


def test_a_failure_at_another_step_leaves_the_reader_s_record_alone(conn):
    document_id = upsert_document(conn, external_id="doc-1")

    mark_failed(
        conn, external_id="doc-1", stage=STAGE_EMBED, error=EmbeddingError("unavailable"),
        document_id=document_id, reader="docai-layout-parser", reader_version="v1",
    )

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM document_reads WHERE document_id = %s", (document_id,))
        assert cursor.fetchone()[0] == 0


def test_a_provider_timeout_does_not_settle_a_document_for_ever():
    failure = _not_embedded(
        "doc-1", sent=410, back=0,
        refusal="the embedding provider refused a batch of 410: HTTP 504: "
                "Worker upstream fetch timed out after 60000ms",
        status=504,
    )

    assert classify(failure) == FAILURE_TRANSIENT
    assert "504" in str(failure)
    assert "410 piece(s)" in str(failure)


def test_a_provider_that_answered_badly_is_still_permanent():
    failure = _not_embedded("doc-1", sent=410, back=409, refusal="")

    assert isinstance(failure, NothingToIndex)
    assert classify(failure) == FAILURE_PERMANENT


def test_a_refusal_the_provider_calls_final_stays_final():
    failure = _not_embedded(
        "doc-1", sent=3, back=0,
        refusal="the embedding provider refused a batch of 3: HTTP 400: input too long",
        status=400,
    )

    assert classify(failure) == FAILURE_PERMANENT


def test_an_unreachable_provider_is_not_read_as_a_status_in_our_own_sentence():
    failure = _not_embedded(
        "doc-1", sent=410, back=0,
        refusal="the embedding provider could not be reached: ConnectTimeout",
        status=None,
    )

    assert classify(failure) == FAILURE_TRANSIENT
