from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from urllib.parse import quote

import httpx
import psycopg

from src.processing.docai import document_of
from migrations import load as load_migration

_migration = load_migration("005_processing")

READ_STATE_READ: str = _migration.READ_STATE_READ
READ_STATE_FAILED: str = _migration.READ_STATE_FAILED
READ_STATES: tuple[str, ...] = _migration.READ_STATES

STORAGE_ROOT = "https://storage.googleapis.com"
SCOPE_STORAGE_READ_WRITE = "https://www.googleapis.com/auth/devstorage.read_write"

ARTIFACT_MEDIA_TYPE = "application/json"


class ArtifactError(Exception):
    ...


class ChunksBeforeArtifact(ArtifactError):
    ...


@dataclass(frozen=True, slots=True)
class ParsedArtifact:

    document_id: int
    external_id: str
    uri: str
    reader: str
    reader_version: str
    page_count: int | None = None
    byte_size: int | None = None


@dataclass(frozen=True, slots=True)
class ReadRecord:

    document_id: int
    state: str
    reader: str
    reader_version: str
    page_count: int | None = None
    byte_size: int | None = None
    failure_reason: str | None = None

    @property
    def was_read(self) -> bool:
        return self.state == READ_STATE_READ


class ObjectStore:

    def __init__(self, *, bucket: str, http: httpx.Client, token: Callable[[], str]) -> None:
        if not bucket.strip():
            raise ArtifactError(
                "object storage is not optional (ASM-020): the batch path writes the "
                "reader's output there, and the kept output is what makes every later "
                "choice reversible (principle P-009)"
            )
        self._bucket = bucket
        self._http = http
        self._token = token

    @property
    def bucket(self) -> str:
        return self._bucket

    def uri_for(self, name: str) -> str:
        return f"gs://{self._bucket}/{name.lstrip('/')}"

    def put(
        self,
        conn: psycopg.Connection,
        name: str,
        payload: bytes,
        *,
        media_type: str = ARTIFACT_MEDIA_TYPE,
    ) -> str:
        del conn
        response = self._http.post(
            f"{STORAGE_ROOT}/upload/storage/v1/b/{quote(self._bucket, safe='')}/o",
            params={"uploadType": "media", "name": name},
            content=payload,
            headers={"Authorization": f"Bearer {self._token()}", "Content-Type": media_type},
        )
        if response.status_code >= 400:
            raise ArtifactError(f"could not write {self.uri_for(name)}: {_message(response)}")
        return self.uri_for(name)

    def get(self, conn: psycopg.Connection, uri: str) -> bytes:
        del conn
        name = self.name_of(uri)
        response = self._http.get(
            f"{STORAGE_ROOT}/storage/v1/b/{quote(self._bucket, safe='')}/o/{quote(name, safe='')}",
            params={"alt": "media"},
            headers={"Authorization": f"Bearer {self._token()}"},
        )
        if response.status_code >= 400:
            raise ArtifactError(f"could not read {uri}: {_message(response)}")
        return response.content

    def list_names(self, prefix: str) -> tuple[str, ...]:
        names: list[str] = []
        page_token: str | None = None
        while True:
            params = {"prefix": prefix, "maxResults": "1000"}
            if page_token:
                params["pageToken"] = page_token
            response = self._http.get(
                f"{STORAGE_ROOT}/storage/v1/b/{quote(self._bucket, safe='')}/o",
                params=params,
                headers={"Authorization": f"Bearer {self._token()}"},
            )
            if response.status_code >= 400:
                raise ArtifactError(f"could not list {self.uri_for(prefix)}: {_message(response)}")
            body = response.json()
            names.extend(str(item.get("name", "")) for item in body.get("items", []))
            page_token = body.get("nextPageToken")
            if not page_token:
                break
        return tuple(sorted(name for name in names if name))

    def name_of(self, uri: str) -> str:
        prefix = f"gs://{self._bucket}/"
        if not uri.startswith(prefix):
            raise ArtifactError(
                f"{uri} is not in bucket {self._bucket!r}; a deployment reads only its own "
                "artifacts (principle P-015)"
            )
        return uri[len(prefix) :]


@runtime_checkable
class ArtifactStore(Protocol):

    def uri_for(self, name: str) -> str: ...

    def name_of(self, uri: str) -> str: ...

    def put(
        self,
        conn: psycopg.Connection,
        name: str,
        payload: bytes,
        *,
        media_type: str = ARTIFACT_MEDIA_TYPE,
    ) -> str: ...

    def get(self, conn: psycopg.Connection, uri: str) -> bytes: ...


ARCHIVE_SCHEME = "pg://"


class ArchiveStore:

    def uri_for(self, name: str) -> str:
        return f"{ARCHIVE_SCHEME}{name.lstrip('/')}"

    def name_of(self, uri: str) -> str:
        if not uri.startswith(ARCHIVE_SCHEME):
            raise ArtifactError(
                f"{uri} was not kept in this archive; a deployment that changed process "
                f"module has artifacts under the old module's store, and reading one here "
                "is not possible — the next run reads the document again, which is what "
                "changing the module costs"
            )
        return uri[len(ARCHIVE_SCHEME) :]

    def put(
        self,
        conn: psycopg.Connection,
        name: str,
        payload: bytes,
        *,
        media_type: str = ARTIFACT_MEDIA_TYPE,
    ) -> str:
        uri = self.uri_for(name)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO parsed_artifacts (uri, payload, media_type, byte_size)
                     VALUES (%s, %s, %s, %s)
                ON CONFLICT (uri) DO UPDATE
                        SET payload    = EXCLUDED.payload,
                            media_type = EXCLUDED.media_type,
                            byte_size  = EXCLUDED.byte_size,
                            written_at = now()
                """,
                (uri, payload, media_type, len(payload)),
            )
        return uri

    def get(self, conn: psycopg.Connection, uri: str) -> bytes:
        self.name_of(uri)
        with conn.cursor() as cursor:
            cursor.execute("SELECT payload FROM parsed_artifacts WHERE uri = %s", (uri,))
            row = cursor.fetchone()
        if row is None:
            raise ArtifactError(
                f"{uri} is recorded against a document and is not in this archive; the "
                "read was paid for and what it produced is gone"
            )
        return bytes(row[0])


def artifact_name(external_id: str, reader_version: str) -> str:
    if not external_id.strip():
        raise ArtifactError("an artifact has to be named after the document it came from")
    return f"parsed/{quote(reader_version, safe='')}/{quote(external_id, safe='')}.json"


def strip_image_binaries(payload: dict) -> dict:
    return _without_image_binaries(payload)


def _without_image_binaries(node: object) -> object:
    if isinstance(node, dict):
        cleaned = {key: _without_image_binaries(value) for key, value in node.items()}
        image_block = cleaned.get("imageBlock")
        if isinstance(image_block, dict) and "image" in image_block:
            image_block = dict(image_block)
            image_block.pop("image", None)
            cleaned["imageBlock"] = image_block
        return cleaned
    if isinstance(node, list):
        return [_without_image_binaries(item) for item in node]
    return node


def store_parsed_artifact(
    conn: psycopg.Connection,
    store: ObjectStore,
    *,
    document_id: int,
    external_id: str,
    payload: dict,
    reader: str,
    reader_version: str,
    page_count: int | None = None,
) -> ParsedArtifact:
    body = json.dumps(
        strip_image_binaries(payload), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    name = artifact_name(external_id, reader_version)
    uri = store.put(conn, name, body)
    with conn.transaction():
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET parsed_uri = %s, updated_at = now() WHERE id = %s",
                (uri, document_id),
            )
            if cursor.rowcount != 1:
                raise ArtifactError(
                    f"no document {document_id} to record artifact {uri} against; the read "
                    "was paid for and its location has nowhere to live"
                )
            _write_read_record(
                cursor,
                document_id=document_id,
                state=READ_STATE_READ,
                reader=reader,
                reader_version=reader_version,
                page_count=page_count,
                byte_size=len(body),
                failure_reason=None,
            )
    return ParsedArtifact(
        document_id=document_id,
        external_id=external_id,
        uri=uri,
        reader=reader,
        reader_version=reader_version,
        page_count=page_count,
        byte_size=len(body),
    )


def record_read_failure(
    conn: psycopg.Connection,
    *,
    document_id: int,
    reason: str,
    reader: str,
    reader_version: str,
    page_count: int | None = None,
    byte_size: int | None = None,
) -> ReadRecord:
    if not reason.strip():
        raise ArtifactError(
            f"document {document_id} was marked failed with no reason; a failure nobody "
            "can act on is the failure AC-035 exists to prevent"
        )
    with conn.transaction():
        with conn.cursor() as cursor:
            _write_read_record(
                cursor,
                document_id=document_id,
                state=READ_STATE_FAILED,
                reader=reader,
                reader_version=reader_version,
                page_count=page_count,
                byte_size=byte_size,
                failure_reason=reason,
            )
    return ReadRecord(
        document_id=document_id,
        state=READ_STATE_FAILED,
        reader=reader,
        reader_version=reader_version,
        page_count=page_count,
        byte_size=byte_size,
        failure_reason=reason,
    )


def read_record(conn: psycopg.Connection, document_id: int) -> ReadRecord | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, state, reader, reader_version, page_count, byte_size,
                   failure_reason
              FROM document_reads
             WHERE document_id = %s
            """,
            (document_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return ReadRecord(
        document_id=int(row[0]),
        state=row[1],
        reader=row[2],
        reader_version=row[3],
        page_count=int(row[4]) if row[4] is not None else None,
        byte_size=int(row[5]) if row[5] is not None else None,
        failure_reason=row[6],
    )


def parsed_artifact(conn: psycopg.Connection, document_id: int) -> ParsedArtifact | None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.external_id, d.parsed_uri, r.reader, r.reader_version,
                   r.page_count, r.byte_size
              FROM documents d
              JOIN document_reads r ON r.document_id = d.id
             WHERE d.id = %s AND r.state = %s AND d.parsed_uri IS NOT NULL
            """,
            (document_id, READ_STATE_READ),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return ParsedArtifact(
        document_id=document_id,
        external_id=row[0],
        uri=row[1],
        reader=row[2],
        reader_version=row[3],
        page_count=int(row[4]) if row[4] is not None else None,
        byte_size=int(row[5]) if row[5] is not None else None,
    )


def require_parsed_artifact(conn: psycopg.Connection, document_id: int) -> ParsedArtifact:
    artifact = parsed_artifact(conn, document_id)
    if artifact is None:
        raise ChunksBeforeArtifact(
            f"document {document_id} has no kept reader output, so no piece may be derived "
            "from it (principle P-009). Read it first: the artifact is what makes a later "
            "change of mind about cutting or embedding cost a rounding error instead of the "
            "original invoice"
        )
    return artifact


def load_parsed_artifact(
    conn: psycopg.Connection, store: "ArtifactStore", artifact: ParsedArtifact
) -> dict:
    raw = store.get(conn, artifact.uri)
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ArtifactError(f"{artifact.uri} does not hold the reader's output: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{artifact.uri} holds {type(payload).__name__}, not the reader's output")
    return payload


def read_batch_output(
    conn: psycopg.Connection, store: ObjectStore, destination: str
) -> dict:
    prefix = store.name_of(destination if destination.endswith("/") else f"{destination}/")
    names = [name for name in store.list_names(prefix) if name.endswith(".json")]
    if not names:
        raise ArtifactError(
            f"the reader reported output at {destination} and there is nothing there; "
            "the read cannot be recorded as done"
        )
    if len(names) > 1:
        raise ArtifactError(
            f"the reader split its response for {destination} into {len(names)} parts; "
            "a split response is refused rather than stitched (AC-207), because the part "
            "order and the page numbering cannot be recovered from what the reader returns"
        )
    document = document_of(json.loads(store.get(conn, store.uri_for(names[0]))))
    document["documentLayout"] = {"blocks": document.get("documentLayout", {}).get("blocks", [])}
    document["chunkedDocument"] = {"chunks": document.get("chunkedDocument", {}).get("chunks", [])}
    return document


def _write_read_record(
    cursor: psycopg.Cursor,
    *,
    document_id: int,
    state: str,
    reader: str,
    reader_version: str,
    page_count: int | None,
    byte_size: int | None,
    failure_reason: str | None,
) -> None:
    cursor.execute(
        """
        INSERT INTO document_reads (
            document_id, state, reader, reader_version, page_count, byte_size, failure_reason
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE
                SET state          = EXCLUDED.state,
                    reader         = EXCLUDED.reader,
                    reader_version = EXCLUDED.reader_version,
                    page_count     = EXCLUDED.page_count,
                    byte_size      = EXCLUDED.byte_size,
                    failure_reason = EXCLUDED.failure_reason,
                    read_at        = now()
        """,
        (document_id, state, reader, reader_version, page_count, byte_size, failure_reason),
    )


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", response.text))
    except ValueError:
        return response.text
