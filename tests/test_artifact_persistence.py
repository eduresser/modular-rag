"""Unit-level coverage for src/processing/artifact.py that is not itself a criterion.

AC-033 and principle P-009 are proven in tests/test_spec_processing_docai.py —
see CLAUDE.md. AC-207 is proven in tests/test_spec_read_fidelity.py. This file
covers the object store's own behaviour, the shape of the artifact's name, and
the rest of the split-response refusal's edges.
"""

import base64
import json

import psycopg
import pytest

from src.processing.artifact import (
    READ_STATE_FAILED,
    READ_STATE_READ,
    ArtifactError,
    artifact_name,
    load_parsed_artifact,
    parsed_artifact,
    read_batch_output,
    read_record,
    record_read_failure,
    store_parsed_artifact,
    strip_image_binaries,
)
from src.store import upsert_document
from tests.conftest import (
    ARTIFACT_BUCKET,
    DOCAI_PROCESSOR,
    DOCAI_VERSION,
    FakeObjectStorage,
    reader_chunk,
    reader_payload,
    text_block,
)

EXTERNAL_ID = "1AbCdEf"


def _payload(marker: str = "um") -> dict:
    return reader_payload(
        [text_block("b1", f"Cláusula {marker}", pages=(1, 1))],
        [reader_chunk("c1", f"Cláusula {marker}", ["b1"])],
    )


def _document(conn: psycopg.Connection, external_id: str = EXTERNAL_ID) -> int:
    return upsert_document(conn, external_id=external_id, title="Contrato")


def _image_block(
    block_id: str,
    *,
    description: str = "",
    image_text: str = "",
    pages: tuple[int, int] = (1, 1),
    binary: bytes = b"\x89PNG\r\n\x1a\nnot a real image",
) -> dict:
    """An ``imageBlock`` carrying its binary inline, the way the reader returns one."""
    inner: dict = {
        "image": {"content": base64.b64encode(binary).decode(), "mimeType": "image/png"},
    }
    if description:
        inner["annotations"] = {"description": description}
    if image_text:
        inner["imageText"] = image_text
    return {
        "blockId": str(block_id),
        "pageSpan": {"pageStart": pages[0], "pageEnd": pages[1]},
        "imageBlock": inner,
    }


def _payload_with_image(*, binary: bytes = b"\x89PNG\r\n\x1a\nnot a real image") -> dict:
    return reader_payload(
        [_image_block(
            "b1",
            description="Planta baixa do imóvel locado.",
            image_text="ÁREA TOTAL: 78,4 m²",
            binary=binary,
        )],
        [reader_chunk("c1", "", ["b1"])],
    )


def test_the_artifact_name_is_derived_from_the_document_and_the_version():
    # Derived, never invented: a run that died after the upload has to find the
    # same object instead of paying to write a second copy.
    first = artifact_name(EXTERNAL_ID, DOCAI_VERSION)

    assert first == artifact_name(EXTERNAL_ID, DOCAI_VERSION)
    assert DOCAI_VERSION in first
    assert EXTERNAL_ID in first
    # Two reader versions of one document do not overwrite each other, which is
    # what makes comparing them possible at all.
    assert artifact_name(EXTERNAL_ID, "pretrained-layout-parser-v1.5") != first


def test_an_identifier_with_a_slash_in_it_stays_one_object():
    name = artifact_name("pasta/documento", DOCAI_VERSION)

    assert name.count("/") == 2
    assert "pasta%2Fdocumento" in name


def test_an_artifact_with_no_document_behind_it_is_refused():
    with pytest.raises(ArtifactError):
        artifact_name("  ", DOCAI_VERSION)


def test_a_bucket_reads_only_its_own_artifacts():
    # One deployment, one client (principle P-015): a uri from somewhere else
    # is a mistake worth naming rather than a request worth making.
    store = FakeObjectStorage().store()

    with pytest.raises(ArtifactError, match="not in bucket"):
        store.name_of("gs://outro-cliente/parsed/v1/doc.json")


def test_a_store_without_a_bucket_cannot_be_built():
    # ASM-020: storage is mandatory, and a deployment that omits it cannot
    # serve production.
    from src.processing.artifact import ObjectStore

    with pytest.raises(ArtifactError, match="ASM-020"):
        ObjectStore(bucket="", http=FakeObjectStorage().client(), token=lambda: "t")


def test_a_refused_upload_leaves_nothing_recorded(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    storage.refuse_upload_with = 503
    document_id = _document(conn)

    with pytest.raises(ArtifactError, match="could not write"):
        store_parsed_artifact(
            conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
            payload=_payload(), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
        )
    # No pointer into nothing: the row is only written once the object is there.
    assert parsed_artifact(conn, document_id) is None
    assert read_record(conn, document_id) is None


def test_reading_the_artifact_back_returns_the_readers_own_output(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    artifact = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=_payload("primeira"), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
        page_count=2,
    )

    kept = load_parsed_artifact(conn, storage.store(), artifact)

    assert kept["document"]["chunkedDocument"]["chunks"][0]["content"] == "Cláusula primeira"
    assert artifact.page_count == 2
    assert artifact.byte_size > 0


def test_an_artifact_that_does_not_hold_the_readers_output_is_refused(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    artifact = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=_payload(), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )
    storage.objects[storage.store().name_of(artifact.uri)] = b"nao e json"

    with pytest.raises(ArtifactError, match="does not hold"):
        load_parsed_artifact(conn, storage.store(), artifact)


def test_reading_a_document_again_replaces_its_record_rather_than_adding_one(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    for marker in ("um", "dois"):
        store_parsed_artifact(
            conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
            payload=_payload(marker), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
            page_count=3,
        )

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM document_reads WHERE document_id = %s", (document_id,))
        assert cursor.fetchone()[0] == 1
    assert read_record(conn, document_id).state == READ_STATE_READ


def test_a_failure_replaces_a_previous_success_and_carries_its_reason(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=_payload(), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )

    record_read_failure(
        conn, document_id=document_id, reason="has 640 pages, above the 500-page limit",
        reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION, page_count=640,
    )

    record = read_record(conn, document_id)
    assert record.state == READ_STATE_FAILED
    assert "500-page" in record.failure_reason
    # The artifact from the earlier successful read is no longer offered: the
    # document's current verdict is that it could not be read.
    assert parsed_artifact(conn, document_id) is None


def test_a_failure_without_a_reason_is_refused(conn: psycopg.Connection):
    with pytest.raises(ArtifactError, match="AC-035"):
        record_read_failure(
            conn, document_id=_document(conn), reason="   ",
            reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
        )


def test_the_database_refuses_a_failure_with_no_reason_that_arrives_another_way(conn: psycopg.Connection):
    document_id = _document(conn)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO document_reads (document_id, state, reader, reader_version)
                VALUES (%s, %s, %s, %s)
                """,
                (document_id, READ_STATE_FAILED, DOCAI_PROCESSOR, DOCAI_VERSION),
            )


def test_the_database_refuses_a_read_that_names_no_version(conn: psycopg.Connection):
    # AC-036 is not only a property of the request: a row that cannot say which
    # version read the document makes the whole corpus unattributable.
    document_id = _document(conn)

    with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO document_reads (document_id, state, reader, reader_version)
                VALUES (%s, %s, %s, %s)
                """,
                (document_id, READ_STATE_READ, DOCAI_PROCESSOR, "  "),
            )


def test_a_long_documents_split_response_is_refused_not_merged(conn):
    # A tenth part would sort between the first and the second by filename
    # alone, and every field outside the two lists would be taken from
    # whichever part happened to come first — both defects AC-207 makes
    # unreachable by refusing the response outright instead of repairing them.
    storage = FakeObjectStorage()
    destination = f"gs://{ARTIFACT_BUCKET}/runs/1/batch-0000/0/"
    storage.put(
        "runs/1/batch-0000/0/output-1.json",
        reader_payload(
            [text_block("b1", "Cláusula primeira", pages=(1, 1))],
            [reader_chunk("c1", "Cláusula primeira", ["b1"])],
        ),
    )
    storage.put(
        "runs/1/batch-0000/0/output-2.json",
        reader_payload(
            [text_block("b2", "Cláusula segunda", pages=(2, 2))],
            [reader_chunk("c2", "Cláusula segunda", ["b2"])],
        ),
    )

    with pytest.raises(ArtifactError, match="2 parts"):
        read_batch_output(conn, storage.store(), destination)


def test_a_destination_with_nothing_in_it_is_an_error_not_an_empty_document(conn):
    # A read the provider reported as done and wrote nothing for cannot be
    # recorded as done: the document would read as parsed and hold no text.
    storage = FakeObjectStorage()

    with pytest.raises(ArtifactError, match="nothing there"):
        read_batch_output(conn, storage.store(), f"gs://{ARTIFACT_BUCKET}/runs/1/batch-0000/0/")


def test_listing_ignores_objects_outside_the_destination(conn):
    storage = FakeObjectStorage()
    storage.put("runs/1/batch-0000/0/output-1.json", _payload())
    storage.put("runs/1/batch-0001/0/output-1.json", _payload("outra"))

    merged = read_batch_output(conn, storage.store(), f"gs://{ARTIFACT_BUCKET}/runs/1/batch-0000/0")

    assert len(merged["chunkedDocument"]["chunks"]) == 1


def test_what_is_written_is_exactly_what_is_read_back(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    payload = _payload("terceira")

    artifact = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=payload, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )

    assert json.loads(storage.objects[storage.store().name_of(artifact.uri)]) == payload


def test_strip_image_binaries_drops_only_the_binary_field():
    stripped = strip_image_binaries(_payload_with_image())

    image_block = stripped["document"]["documentLayout"]["blocks"][0]["imageBlock"]
    assert "image" not in image_block
    # The two fields an image reaches the index through are untouched.
    assert image_block["annotations"]["description"] == "Planta baixa do imóvel locado."
    assert image_block["imageText"] == "ÁREA TOTAL: 78,4 m²"


def test_strip_image_binaries_reaches_a_block_nested_under_a_list_or_table():
    nested = reader_payload(
        [
            {
                "blockId": "list-1",
                "pageSpan": {"pageStart": 1, "pageEnd": 1},
                "listBlock": {
                    "type": "unordered",
                    "listEntries": [{"blocks": [_image_block("list-1-img")]}],
                },
            }
        ],
        [reader_chunk("c1", "", ["list-1-img"])],
    )

    stripped = strip_image_binaries(nested)

    entry_block = (
        stripped["document"]["documentLayout"]["blocks"][0]["listBlock"]["listEntries"][0]
        ["blocks"][0]
    )
    assert "image" not in entry_block["imageBlock"]


def test_a_document_with_no_image_block_is_left_untouched():
    payload = _payload("quarta")

    assert strip_image_binaries(payload) == payload


def test_storing_an_artifact_drops_the_image_binary_before_it_is_written(conn: psycopg.Connection):
    storage = FakeObjectStorage()
    document_id = _document(conn)
    binary = b"\x89PNG\r\n\x1a\nnot a real image, repeated to be unmistakable"
    encoded = base64.b64encode(binary).decode()
    payload = _payload_with_image(binary=binary)

    artifact = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id=EXTERNAL_ID,
        payload=payload, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )

    stored_bytes = storage.objects[storage.store().name_of(artifact.uri)]
    stored = json.loads(stored_bytes)
    image_block = stored["document"]["documentLayout"]["blocks"][0]["imageBlock"]
    assert "image" not in image_block
    assert encoded not in stored_bytes.decode()

    kept = load_parsed_artifact(conn, storage.store(), artifact)
    assert "image" not in kept["document"]["documentLayout"]["blocks"][0]["imageBlock"]
    assert kept["document"]["documentLayout"]["blocks"][0]["imageBlock"]["annotations"][
        "description"
    ] == "Planta baixa do imóvel locado."
