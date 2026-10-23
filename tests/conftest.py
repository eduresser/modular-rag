"""A real PostgreSQL for every test that touches the archive.

The store was chosen for one property — the indivisible delete in AC-073 — and
a substitute that only pretends to have transactions cannot prove it. Nor can a
substitute provide ``halfvec``, an HNSW index or a Portuguese text-search
configuration, which the schema is built on. So these tests run against the
same image the deployment runs: ``pgvector/pgvector:pg16``.

Docker missing is a hard failure here, never a skip. The engine refuses a
skipped test as proof of an acceptance criterion, so a silent skip would report
a green suite that has verified nothing at all — the one outcome worse than a
red one.

Isolation is a transaction per test, rolled back at the end. The store's own
functions open their transaction with ``conn.transaction()``, which nests as a
savepoint inside this one, so the atomicity under test is the real thing and
still leaves no trace between tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from decimal import Decimal
import socket
import subprocess
import tempfile
import time
import urllib.parse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import httpx
import yaml

from src.config import load_config
import psycopg
import pytest

from src.db import create_schema
from src.prices import PriceBook
from src.store import ModelStamp, Piece, mark_document_gone, upsert_document, upsert_pieces

IMAGE = "pgvector/pgvector:pg16"
CONTAINER_NAME = "modular-rag-tests-pg"
STARTUP_TIMEOUT_SECONDS = 30.0

# Small on purpose: the schema is parameterised by dimension (ASM-041) and
# nothing under test depends on the width, so eight keeps fixtures readable.
TEST_DIMENSION = 8

# The stamp every fixture writes under, and the one a search is scoped to.
#
# **It names the model `ingest_config` names**, and that is not cosmetic: a
# search embeds the question with the configured model and compares it against
# the generation the archive is serving, so a fixture stamped with some other
# name was a deployment answering questions with two models crossed. Nothing
# noticed, because until `_generation_mismatch` existed nothing compared them —
# which is the same reason the real corpus degraded in silence.
#
# `EMBED_MODEL` is defined further down, beside the rest of the configuration
# it belongs to; the literal is repeated here rather than reordering the file,
# and `test_the_fixture_stamp_is_the_configured_model` in tests/test_schema.py
# fails if the two ever drift.
TEST_STAMP = ModelStamp(model="qwen/qwen3-embedding-8b", dimension=TEST_DIMENSION)

# The client every usage and search row is attributed to (AC-168). One value,
# because one database serves one client (principle P-015) — a test that needs
# a second client is testing the attribution itself and says so.
TEST_CLIENT = "example"

#: The reference the deployment's configuration carries, and the value the
#: fake secret manager resolves it to. Two different strings on purpose: the
#: committed setting is a reference and never a token (principle P-012), and a
#: test that used one string for both could not tell them apart.
API_CREDENTIAL_REFERENCE = "env:API_TOKEN"
API_TOKEN = "an-api-token-nobody-guesses"
EMBED_TOKEN = "an-embedding-key"

#: The one name this deployment answers to in tests.
ALLOWED_HOST = "rag.example.internal"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _require_docker() -> None:
    probe = _docker("ps", "--quiet")
    if probe.returncode != 0:
        raise pytest.UsageError(
            "these tests need a running Docker daemon: the archive's guarantees "
            f"(atomic delete, halfvec, HNSW, Portuguese text search) are proven against {IMAGE}, "
            "and a stand-in cannot prove them. Docker said: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )


def _wait_until_accepting(dsn: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=2):
                return
        except psycopg.OperationalError as exc:  # not up yet
            last_error = exc
            time.sleep(0.25)
    raise pytest.UsageError(f"{IMAGE} did not accept connections within {STARTUP_TIMEOUT_SECONDS}s: {last_error}")


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """Start one throwaway PostgreSQL for the whole session and yield its DSN."""
    _require_docker()
    _docker("rm", "--force", CONTAINER_NAME)
    port = _free_port()
    started = _docker(
        "run",
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--env",
        "POSTGRES_HOST_AUTH_METHOD=trust",
        "--publish",
        f"127.0.0.1:{port}:5432",
        # The data never outlives the session, so durability is pure cost here.
        "--tmpfs",
        "/var/lib/postgresql/data:rw",
        IMAGE,
        "-c",
        "fsync=off",
        "-c",
        "synchronous_commit=off",
        "-c",
        "full_page_writes=off",
    )
    if started.returncode != 0:
        raise pytest.UsageError(f"could not start {IMAGE}: {started.stderr.strip()}")
    dsn = f"postgresql://postgres@127.0.0.1:{port}/postgres"
    try:
        _wait_until_accepting(dsn)
        yield dsn
    finally:
        _docker("rm", "--force", CONTAINER_NAME)


@pytest.fixture(scope="session")
def schema_dsn(postgres_dsn: str) -> str:
    """The same DSN, with the whole schema applied once."""
    with psycopg.connect(postgres_dsn) as conn:
        create_schema(conn, dimension=TEST_DIMENSION)
        conn.commit()
    return postgres_dsn


# Nothing is priced in code any more (ASM-075), so a test that needs prices
# needs a deployment's configuration. Reading the first client's own committed
# file rather than restating figures here means the prices under test are the
# prices that ship, and a typo in that file fails these tests instead of
# surfacing as an empty cost report months later.
BASE_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "apps" / "base" / "config.yaml"


def configured_prices(config_file: pathlib.Path | None = None) -> list[dict]:
    """The ``usage.prices`` list out of a committed deployment.

    A list of objects now, where it was a mapping of slashed keys and, before
    that, a four-hundred-character JSON string that `source` mangled unless it
    was single-quoted — which is how the first deployment ran with no price at
    all. Each shape fixed the defect the one before it had: the mapping removed
    the escaping, and the list closed the sets, so a misspelled operation stops
    the boot instead of being a price that matches nothing.
    """
    location = config_file or BASE_CONFIG
    document = yaml.safe_load(location.read_text()) or {}
    prices = (document.get("usage") or {}).get("prices")
    if not prices:
        raise pytest.UsageError(f"{location} declares no usage.prices")
    return [dict(entry) for entry in prices]


@pytest.fixture
def price_book() -> PriceBook:
    """The prices the first client's deployment is configured with."""
    return PriceBook(configured_prices())


def configured_amount(*, operation: str, provider: str, model: str, unit: str) -> Decimal:
    """What the committed file says this costs, read from the file.

    A test that asserts a rate as a literal has copied the deployment's price
    book into Python, and the day the rate is corrected in the YAML the test
    fails for a reason that has nothing to do with what it was written to
    check. There is one place a price lives, and a test about what a row
    RECORDS reads it from there — the only literal figures left are in
    tests/test_prices.py, whose whole job is to guard the file itself.
    """
    return PriceBook(configured_prices()).price_for(
        operation=operation, provider=provider, model=model, unit=unit
    ).amount


@pytest.fixture
def wider_price_book() -> PriceBook:
    """A deployment that bills more vendors than the committed one does.

    The base prices what a deployment on this configuration can actually bill
    and nothing else — it stopped being a catalogue, because a rate for a vendor
    nobody runs goes stale where nobody is looking. A test about cost
    ARITHMETIC across providers still needs more than one provider, so it
    configures them here rather than asking the committed file to carry rows for
    its benefit.
    """
    return PriceBook([
        *configured_prices(),
        # Round and obviously invented, so nobody mistakes them for researched
        # rates: no real price lives outside the deployment's own file.
        {"operation": "embed", "provider": "openai", "model": "text-embedding-3-large",
         "unit": "token", "amount": "0.0000002"},
        {"operation": "embed", "provider": "google", "model": "gemini-embedding-001",
         "unit": "token", "amount": "0.0000003"},
        {"operation": "rerank", "provider": "self-hosted", "unit": "request", "amount": "0"},
        # A second read provider, so a test about the invoice view has two
        # invoices to tell apart. Measured local throughput, never invoiced.
        {"operation": "read", "provider": "docling", "unit": "page", "amount": "0.0001"},
        # Paying the reader to cut a document it has already read. A path that
        # exists (`recut_document` takes a reader) and that no caller reaches,
        # so no deployment prices it — but a test about cost arithmetic across
        # operations still needs it to cost something.
        {"operation": "chunk", "provider": "google", "unit": "page", "amount": "0.00002"},
    ])


@pytest.fixture
def empty_price_book() -> PriceBook:
    """A deployment that has not been told any of its rates yet.

    A valid state, not a broken one: everything it records is unpriced, and
    every cost figure reports the gap instead of totalling as though the work
    had been free.
    """
    return PriceBook()


@pytest.fixture(scope="module", autouse=True)
def bounded_index_turnover(schema_dsn: str) -> Iterator[None]:
    """Give the vector index back between test modules.

    A rolled-back INSERT still leaves a dead heap tuple, and pgvector's HNSW
    keeps a graph entry per dead tuple: the rows go, the graph does not. Over
    the hundreds of corpora this suite builds and rolls back on one shared
    schema, a later dense search spends its whole ``ef_search`` budget walking
    entries that point at nothing. It comes back with too few candidates or
    none, ``MAX_MATCH_DISTANCE`` drops what little it found, and a test that
    passes on its own fails inside the suite — a property of the run order and
    not of the code it is meant to be checking.

    Per module rather than per test: a rebuild costs milliseconds at this size
    and one file's worth of turnover is measurably harmless, where a whole
    suite's is not.
    """
    yield
    with psycopg.connect(schema_dsn) as rebuild:
        rebuild.autocommit = True
        with rebuild.cursor() as cursor:
            cursor.execute("REINDEX INDEX chunks_embedding_idx")


@pytest.fixture
def conn(schema_dsn: str) -> Iterator[psycopg.Connection]:
    """A connection inside a transaction that is rolled back when the test ends."""
    with psycopg.connect(schema_dsn) as connection:
        try:
            yield connection
        finally:
            connection.rollback()


@pytest.fixture
def sync_database(schema_dsn: str) -> Iterator[str]:
    """An isolated schema for API/worker tests that must commit across connections."""
    from uuid import uuid4
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = "sync_test_" + uuid4().hex
    with psycopg.connect(schema_dsn) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    dsn = make_conninfo(schema_dsn, options=f"-csearch_path={name},public")
    try:
        with psycopg.connect(dsn) as setup:
            create_schema(setup, dimension=TEST_DIMENSION)
        yield dsn
    finally:
        with psycopg.connect(schema_dsn) as cleanup:
            cleanup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))


# ── Building a corpus to search ───────────────────────────────────────────
#
# Directions in the first two dimensions, so cosine distance is something a
# test can reason about: two pieces 10 degrees apart rank in a known order,
# and nothing depends on a real embedding model being reachable.


def direction(degrees: float) -> list[float]:
    """A unit vector pointing ``degrees`` around the first two dimensions."""
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)] + [0.0] * (TEST_DIMENSION - 2)


@dataclass(frozen=True)
class ChunkSpec:
    """One piece to index: its text, where it sits, and which way it points."""

    ordinal: int
    content: str
    degrees: float = 0.0
    pages: tuple[int, int] | None = None
    headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSpec:
    external_id: str
    title: str
    chunks: tuple[ChunkSpec, ...]
    gone: bool = False


def index_corpus(
    conn: "psycopg.Connection",
    documents: Sequence[DocumentSpec],
    *,
    stamp: "ModelStamp | None" = None,
) -> dict[str, int]:
    """Write ``documents`` into the archive, returning external id → row id."""
    written: dict[str, int] = {}
    for document in documents:
        document_id = upsert_document(conn, external_id=document.external_id, title=document.title)
        upsert_pieces(
            conn,
            document_id=document_id,
            stamp=stamp or TEST_STAMP,
            pieces=[
                Piece(
                    ordinal=chunk.ordinal,
                    content=chunk.content,
                    embedding=direction(chunk.degrees),
                    page_start=chunk.pages[0] if chunk.pages else None,
                    page_end=chunk.pages[1] if chunk.pages else None,
                    heading_path=chunk.headings,
                )
                for chunk in document.chunks
            ],
        )
        written[document.external_id] = document_id
        if document.gone:
            mark_document_gone(conn, external_id=document.external_id)
    return written


# ── A fake Google Drive, at the HTTP layer ────────────────────────────────
#
# Faked at the transport rather than by stubbing the source class, so the code
# under test is the real request construction: the field projections ASM-014
# insists on, the shared-drive parameters, the status codes principle P-005
# turns on. Every request is recorded, which is what lets a test assert that no
# call went out without asking for its fields — the failure ASM-014 describes
# is silence, so it can only be caught by looking at what was sent.


@dataclass
class DriveFile:
    """One entry in the fake corpus, in the source's own field names."""

    id: str
    name: str
    mimeType: str
    version: str | None = None
    md5Checksum: str | None = None
    size: int | None = None
    parents: tuple[str, ...] = ()
    trashed: bool = False
    targetId: str | None = None
    content: bytes = b""
    #: 403 on any direct lookup — a permission change, not a deletion.
    denied: bool = False
    #: 404 on any direct lookup — actually gone.
    missing: bool = False

    def projection(self) -> dict:
        entry: dict = {"id": self.id, "name": self.name, "mimeType": self.mimeType}
        if self.version is not None:
            entry["version"] = self.version
        if self.md5Checksum is not None:
            entry["md5Checksum"] = self.md5Checksum
        if self.size is not None:
            entry["size"] = str(self.size)
        if self.parents:
            entry["parents"] = list(self.parents)
        if self.trashed:
            entry["trashed"] = True
        if self.targetId is not None:
            entry["shortcutDetails"] = {"targetId": self.targetId, "targetMimeType": "text/plain"}
        return entry


#: The source's own type string for a folder, so the fake answers with exactly
#: what the real API does rather than with something that only looks like it.
DRIVE_FOLDER_KIND = "application/vnd.google-apps.folder"


@dataclass
class DriveChange:
    """One entry in the fake change feed."""

    fileId: str
    removed: bool = False


class FakeDrive:
    """A Drive that answers over HTTP without a network.

    ``requests`` holds every request made, in order. ``refusals`` maps a file
    id to a status code so a test can make one lookup fail without touching the
    rest of the corpus.
    """

    def __init__(
        self,
        files: "Sequence[DriveFile]" = (),
        *,
        changes: "Sequence[DriveChange]" = (),
        start_token: str = "t0",
        page_size: int = 1000,
    ) -> None:
        self.files: dict[str, DriveFile] = {entry.id: entry for entry in files}
        self.changes: list[DriveChange] = list(changes)
        self.start_token = start_token
        self.page_size = page_size
        self.requests: list[httpx.Request] = []
        #: Status codes to return instead of a body, keyed by path fragment.
        self.rate_limit_first: int = 0

    # -- building a corpus ------------------------------------------------

    def add(self, entry: DriveFile) -> DriveFile:
        self.files[entry.id] = entry
        return entry

    def ensure_folder(self, folder_id: str) -> DriveFile:
        """Answer for this folder as an ordinary readable folder, if nothing else does.

        The source resolves every watched folder against the source before it
        enumerates anything, so a fake drive that holds documents but never the
        folder they hang in would refuse the whole run. A test that cares about
        the folder — because it is missing, denied, trashed or not a folder at
        all — adds its own entry first, and this leaves it alone.
        """
        existing = self.files.get(folder_id)
        if existing is not None:
            return existing
        return self.add(DriveFile(id=folder_id, name=folder_id, mimeType=DRIVE_FOLDER_KIND))

    def push_change(self, file_id: str, *, removed: bool = False) -> None:
        self.changes.append(DriveChange(fileId=file_id, removed=removed))

    # -- the transport ----------------------------------------------------

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.rate_limit_first > 0:
            self.rate_limit_first -= 1
            return httpx.Response(
                429,
                json={"error": {"errors": [{"reason": "userRateLimitExceeded"}]}},
                request=request,
            )
        path = request.url.path.removeprefix("/drive/v3")
        params = dict(request.url.params)
        if path == "/changes/startPageToken":
            return httpx.Response(200, json={"startPageToken": self.start_token}, request=request)
        if path == "/changes":
            return self._changes(request, params)
        if path == "/files":
            return self._list(request, params)
        if path.endswith("/export"):
            return self._export(request, path.removeprefix("/files/").removesuffix("/export"), params)
        if path.startswith("/files/"):
            return self._file(request, path.removeprefix("/files/"), params)
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}}, request=request)

    def _refusal(self, request: httpx.Request, entry: DriveFile | None, file_id: str) -> httpx.Response | None:
        if entry is None or entry.missing:
            return httpx.Response(
                404,
                json={"error": {"errors": [{"reason": "notFound"}], "message": f"File not found: {file_id}"}},
                request=request,
            )
        if entry.denied:
            return httpx.Response(
                403,
                json={"error": {"errors": [{"reason": "forbidden"}], "message": "insufficient permissions"}},
                request=request,
            )
        return None

    def _file(self, request: httpx.Request, file_id: str, params: dict) -> httpx.Response:
        entry = self.files.get(file_id)
        refusal = self._refusal(request, entry, file_id)
        if refusal is not None:
            return refusal
        assert entry is not None
        if params.get("alt") == "media":
            return httpx.Response(200, content=entry.content, request=request)
        return httpx.Response(200, json=entry.projection(), request=request)

    def _export(self, request: httpx.Request, file_id: str, params: dict) -> httpx.Response:
        entry = self.files.get(file_id)
        refusal = self._refusal(request, entry, file_id)
        if refusal is not None:
            return refusal
        assert entry is not None
        return httpx.Response(200, content=entry.content, request=request)

    def _list(self, request: httpx.Request, params: dict) -> httpx.Response:
        query = params.get("q", "")
        parent = query.split("'")[1] if "'" in query else ""
        matching = [
            entry
            for entry in self.files.values()
            if parent in entry.parents and not entry.trashed and not entry.missing
        ]
        start = int(params.get("pageToken", "0"))
        page = matching[start : start + self.page_size]
        body: dict = {"files": [entry.projection() for entry in page]}
        if start + self.page_size < len(matching):
            body["nextPageToken"] = str(start + self.page_size)
        return httpx.Response(200, json=body, request=request)

    def _changes(self, request: httpx.Request, params: dict) -> httpx.Response:
        token = params.get("pageToken", self.start_token)
        start = int(token.removeprefix("t")) if token.startswith("t") else 0
        page = self.changes[start : start + self.page_size]
        entries = []
        for change in page:
            item: dict = {"fileId": change.fileId, "removed": change.removed}
            if not change.removed:
                entry = self.files.get(change.fileId)
                if entry is not None:
                    item["file"] = entry.projection()
            entries.append(item)
        body: dict = {"changes": entries}
        position = start + len(page)
        if position < len(self.changes):
            body["nextPageToken"] = f"t{position}"
        else:
            body["newStartPageToken"] = f"t{position}"
        return httpx.Response(200, json=body, request=request)


def drive_source(drive: FakeDrive, *, folder_ids: Sequence[str] = ("root-folder",), **kwargs):
    """A real GoogleDriveSource wired to ``drive``, with a constant token."""
    from src.sources.gdrive import GoogleDriveSource

    for folder in folder_ids:
        drive.ensure_folder(folder)
    return GoogleDriveSource(
        folder_ids=list(folder_ids), http=drive.client(), token=lambda: "fake-token", **kwargs
    )


# ── A fake Document AI, and a fake bucket, both at the HTTP layer ─────────
#
# Same decision as the Drive source, for the same reason: what has to be proven
# is the request that goes out. The pinned processor version lives in the URL
# (AC-036), the description switches and the cutting controls live in the body
# (ASM-021, ASM-024), and the concurrency ceiling (AC-037) is a property of how
# many operations the provider ever has open at once — which only the far end
# can observe. A stubbed client class would assert none of those.

DOCAI_PROCESSOR = "projects/modular-rag/locations/us/processors/abc123"
DOCAI_VERSION = "pretrained-layout-parser-v1.0-2024-06-03"
ARTIFACT_BUCKET = "modular-rag-artifacts"


class FakeObjectStorage:
    """One bucket in memory, answering the object-storage JSON API."""

    def __init__(self, bucket: str = ARTIFACT_BUCKET) -> None:
        self.bucket = bucket
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        #: Set to a status code to make the next upload fail.
        self.refuse_upload_with: int | None = None

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def store(self):
        """A real ObjectStore wired to this bucket, with a constant token."""
        from src.processing.artifact import ObjectStore

        return ObjectStore(bucket=self.bucket, http=self.client(), token=lambda: "fake-token")

    def uri(self, name: str) -> str:
        return f"gs://{self.bucket}/{name}"

    def name_of(self, uri: str) -> str:
        """The object name inside this bucket, for a gs:// location."""
        return uri[len(f"gs://{self.bucket}/") :]

    def put(self, name: str, payload: bytes | dict) -> str:
        """Seed an object without going through the API."""
        self.objects[name] = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        )
        return self.uri(name)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # raw_path, because an object name contains slashes and the decoded
        # path would make "a/b.json" indistinguishable from a directory.
        path = urllib.parse.unquote(request.url.raw_path.decode("ascii").split("?")[0])
        params = dict(request.url.params)
        upload_prefix = f"/upload/storage/v1/b/{self.bucket}/o"
        object_prefix = f"/storage/v1/b/{self.bucket}/o"
        if path == upload_prefix and request.method == "POST":
            if self.refuse_upload_with is not None:
                status, self.refuse_upload_with = self.refuse_upload_with, None
                return httpx.Response(status, json={"error": {"message": "refused"}}, request=request)
            self.objects[params["name"]] = request.content
            return httpx.Response(200, json={"name": params["name"]}, request=request)
        if path == object_prefix and request.method == "GET":
            prefix = params.get("prefix", "")
            items = [{"name": name} for name in sorted(self.objects) if name.startswith(prefix)]
            return httpx.Response(200, json={"items": items}, request=request)
        if path.startswith(f"{object_prefix}/") and request.method == "GET":
            name = path[len(object_prefix) + 1 :]
            if name not in self.objects:
                return httpx.Response(
                    404, json={"error": {"message": f"no such object {name}"}}, request=request
                )
            return httpx.Response(200, content=self.objects[name], request=request)
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}}, request=request)


class FakeDocumentAI:
    """A reader that answers over HTTP, and counts what it has open at once.

    ``peak_active`` is the number AC-037 turns on, and it is observed here
    rather than reported by the client: a run that respected the ceiling and a
    run that blew through it and recovered look identical from the client's
    own results.
    """

    def __init__(
        self,
        *,
        rounds_until_done: int = 1,
        response: dict | None = None,
        failing_documents: "Sequence[str]" = (),
        operation_error: str = "",
        storage: "FakeObjectStorage | None" = None,
        outputs: "dict[str, dict] | None" = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.submissions: list[dict] = []
        self.response = response or {}
        #: Where the batch path's output is written, when a test wants the
        #: whole flow rather than only the request. The real provider writes
        #: shards into the caller's bucket and reports the location; nothing
        #: downstream can be exercised without that half actually happening.
        self.storage = storage
        #: The reader's output per document, keyed by the document's external
        #: identifier — which is the last segment of the input location.
        self.outputs = dict(outputs or {})
        self.rounds_until_done = rounds_until_done
        self.failing_documents = set(failing_documents)
        self.operation_error = operation_error
        self.active = 0
        self.peak_active = 0
        #: Every interval the client slept for, so a test can assert that the
        #: batches over the ceiling waited rather than failing.
        self.waits: list[float] = []
        self._operations: dict[str, dict] = {}

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def reader(self, **kwargs):
        """A real LayoutParser wired to this reader, pinned and with a fake token."""
        from src.processing.docai import ChunkingOptions, LayoutParser

        settings = {
            "processor_id": DOCAI_PROCESSOR,
            "processor_version": DOCAI_VERSION,
            "http": self.client(),
            "token": lambda: "fake-token",
            "chunking": ChunkingOptions(size=800, cutter="local"),
            "sleeper": self.waits.append,
        }
        settings.update(kwargs)
        return LayoutParser(**settings)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith(":batchProcess"):
            return self._submit(request)
        if path.endswith(":process"):
            return httpx.Response(200, json=self.response, request=request)
        if "/operations/" in path:
            return self._poll(request, path.split("/v1/", 1)[1])
        return httpx.Response(404, json={"error": {"message": f"no such path {path}"}}, request=request)

    def _submit(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.submissions.append(body)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        name = f"projects/modular-rag/locations/us/operations/{len(self._operations) + 1}"
        self._operations[name] = {"polls": 0, "body": body}
        return httpx.Response(200, json={"name": name}, request=request)

    def _poll(self, request: httpx.Request, name: str) -> httpx.Response:
        operation = self._operations.get(name)
        if operation is None:
            return httpx.Response(404, json={"error": {"message": f"no operation {name}"}}, request=request)
        operation["polls"] += 1
        if operation["polls"] <= self.rounds_until_done:
            return httpx.Response(200, json={"name": name, "done": False}, request=request)
        if not operation.get("closed"):
            operation["closed"] = True
            self.active -= 1
        body: dict = {"name": name, "done": True, "metadata": self._statuses(operation["body"])}
        if self.operation_error:
            body["error"] = {"code": 3, "message": self.operation_error}
        else:
            body["response"] = {}
        return httpx.Response(200, json=body, request=request)

    def _statuses(self, body: dict) -> dict:
        prefix = body["documentOutputConfig"]["gcsOutputConfig"]["gcsUri"].rstrip("/")
        documents = body["inputDocuments"]["gcsDocuments"]["documents"]
        statuses = []
        for position, document in enumerate(documents):
            uri = document["gcsUri"]
            if uri in self.failing_documents:
                statuses.append(
                    {"inputGcsSource": uri, "status": {"code": 3, "message": "unreadable"}}
                )
                continue
            destination = f"{prefix}/{position}/"
            if self.storage is not None:
                self.storage.put(
                    f"{self.storage.name_of(destination)}0.json",
                    self._output_for(uri),
                )
            statuses.append(
                {
                    "inputGcsSource": uri,
                    "status": {"code": 0},
                    "outputGcsDestination": destination,
                }
            )
        return {"individualProcessStatuses": statuses}

    def _output_for(self, uri: str) -> dict:
        """What this document's shard holds, by the identifier in its location."""
        external_id = uri.rstrip("/").rsplit("/", 1)[-1]
        return self.outputs.get(external_id) or self.response or reader_payload([], [])


# ── Building a reader payload to map ──────────────────────────────────────
#
# The block tree and the chunks the reader cut from it, in the provider's own
# field names. Written by hand rather than recorded from a real call because
# the interesting cases — a list, a table with its caption, a chunk with no
# heading above it — are exactly the ones a convenient sample would not have.


def text_block(block_id, text, *, pages=(1, 1), kind="paragraph", blocks=(), annotation=""):
    body: dict = {"text": text, "type": kind}
    if blocks:
        body["blocks"] = list(blocks)
    if annotation:
        body["annotations"] = {"description": annotation}
    return {
        "blockId": str(block_id),
        "pageSpan": {"pageStart": pages[0], "pageEnd": pages[1]},
        "textBlock": body,
    }


def list_block(block_id, entries, *, pages=(1, 1), ordered=True):
    return {
        "blockId": str(block_id),
        "pageSpan": {"pageStart": pages[0], "pageEnd": pages[1]},
        "listBlock": {
            "type": "ordered" if ordered else "unordered",
            "listEntries": [
                {"blocks": [text_block(f"{block_id}-{position}", entry, pages=pages)]}
                for position, entry in enumerate(entries, start=1)
            ],
        },
    }


def table_block(block_id, header, rows, *, pages=(1, 1), annotation=""):
    counter = iter(range(1_000))

    def row(cells):
        return {
            "cells": [
                {"blocks": [text_block(f"{block_id}-c{next(counter)}", cell, pages=pages)]}
                for cell in cells
            ]
        }

    body: dict = {}
    if header:
        body["headerRows"] = [row(header)]
    body["bodyRows"] = [row(cells) for cells in rows]
    if annotation:
        body["annotations"] = {"description": annotation}
    return {
        "blockId": str(block_id),
        "pageSpan": {"pageStart": pages[0], "pageEnd": pages[1]},
        "tableBlock": body,
    }


def reader_chunk(chunk_id, content, source_block_ids, *, pages=None):
    chunk: dict = {
        "chunkId": str(chunk_id),
        "content": content,
        "sourceBlockIds": [str(block_id) for block_id in source_block_ids],
    }
    if pages is not None:
        chunk["pageSpan"] = {"pageStart": pages[0], "pageEnd": pages[1]}
    return chunk


def reader_payload(blocks, chunks, *, wrapped=True):
    """The reader's output, bare as the batch path writes it or in an envelope.

    The re-cut call returns the envelope; the batch path writes the document on
    its own. Both shapes reach ``to_chunks.document_of``, which is why a test
    can produce either.
    """
    document = {
        "documentLayout": {"blocks": list(blocks)},
        "chunkedDocument": {"chunks": list(chunks)},
    }
    return {"document": document} if wrapped else document


# ── A real PDF, small enough to read ──────────────────────────────────────
#
# Hand-written rather than produced by a library, for the reason the word
# document below is: the free module is about the text a PDF already carries,
# and the only honest way to assert that is to hand the real extractor a real
# file. Every byte here is the format's own — a catalogue, a page tree, one
# content stream per page — and the pages come out of `pypdf` as the strings
# put in. A page given an empty string carries no text, which is what a blank
# page in the middle of a scan-and-print job actually looks like.


def pdf_bytes(pages: "Sequence[str]") -> bytes:
    """A valid PDF whose pages carry exactly the text they were given."""
    out = bytearray(b"%PDF-1.4\n")
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(pages)))
    body: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()),
        # WinAnsiEncoding, or the accents this corpus is made of come back as
        # the font's built-in encoding: `Cláusula` extracted as `Clˆ¡usula`.
        # A real producer declares one, so a helper that did not would be a
        # test tripping over its own fixture rather than over the extractor.
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"),
    ]
    for index, text in enumerate(pages):
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("cp1252") if text else b""
        )
        body.append((
            4 + 2 * index,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources "
            f"<< /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>".encode(),
        ))
        body.append((
            5 + 2 * index,
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        ))
    offsets: dict[int, int] = {}
    for number, payload in body:
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"
    start = len(out)
    size = max(offsets) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        out += (
            f"{offsets[number]:010d} 00000 n \n".encode()
            if number in offsets
            else b"0000000000 65535 f \n"
        )
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    return bytes(out)


# ── A real word-processor file, small enough to read ──────────────────────
#
# AC-016 is about a format keeping its heading levels, and the only honest way
# to assert that is against a file in the format. A .docx is a zip holding
# word/document.xml, and a heading is a paragraph carrying a Heading<N> style —
# so twenty lines of zipfile build one, and the assertion is about the bytes
# that actually came back rather than about a string somebody hand-wrote.

#: The three levels that reconstruct "Contrato › Cláusula Quinta › Rescisão
#: antecipada", which is the citation AC-016 exists to make possible.
HEADING_LEVELS: tuple[tuple[int, str], ...] = (
    (1, "Contrato"),
    (2, "Cláusula Quinta"),
    (3, "Rescisão antecipada"),
)

_DOCX_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def word_document(
    headings: "Sequence[tuple[int, str]]" = HEADING_LEVELS,
    *,
    body: str = "texto da cláusula",
) -> bytes:
    """A minimal .docx carrying ``headings`` as real heading-styled paragraphs."""
    import io
    import zipfile

    paragraphs = "".join(
        f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>'
        f"<w:r><w:t>{text}</w:t></w:r></w:p>"
        for level, text in headings
    )
    paragraphs += f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD_NAMESPACE}"><w:body>{paragraphs}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def document_xml(content: bytes) -> str:
    """The main part of a .docx, as text."""
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return archive.read("word/document.xml").decode("utf-8")


def heading_styles(content: bytes) -> list[str]:
    """The heading styles a .docx carries, in document order."""
    import re

    return re.findall(r'w:pStyle w:val="(Heading\d)"', document_xml(content))


# ── A fake embedding provider, at the HTTP layer ──────────────────────────
#
# The same decision as the Drive source and the reader, for the same reason:
# what has to be proven is the request that goes out. Provider pinning is a
# field in the body (AC-052), the width is a field in the body (ASM-031), the
# permitted extra parameters are fields in the body and the forbidden ones are
# their absence (AC-051) — a stubbed client class would assert none of it.
#
# The vectors are a deterministic function of the text, so a test can say which
# vector belongs to which piece and catch a batch that came back mis-paired.

EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_BASE_URL = "https://openrouter.ai/api/v1"


def vector_for(text: str, dimension: int = TEST_DIMENSION) -> tuple[float, ...]:
    """A unit vector this text and no other maps to, in the first two dimensions."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    degrees = (int.from_bytes(digest[:4], "big") % 3600) / 10.0
    return tuple(direction(degrees))


def token_count(text: str) -> int:
    """What the fake provider bills for one text. Words, so a test can add up."""
    return len(text.split())


class FakeEmbeddings:
    """An OpenAI-shaped embeddings endpoint that answers without a network.

    ``bodies`` holds every request body in order, which is where the assertions
    about pinning, width and extra parameters are made.
    """

    def __init__(
        self,
        *,
        model: str = EMBED_MODEL,
        dimension: int = TEST_DIMENSION,
        served_model: str | None = None,
        served_provider: str = "",
        fail_batches: "Sequence[int]" = (),
        status: int = 500,
        honour_dimensions: bool = True,
        out_of_order: bool = False,
        scale: float = 1.0,
    ) -> None:
        self.model = model
        self.dimension = dimension
        #: What the answer claims produced it; None means "whatever was asked".
        self.served_model = served_model
        #: What the answer claims served it; "" means the field is absent.
        self.served_provider = served_provider
        #: Which requests (0-based, in arrival order) fail instead of answering.
        self.fail_batches = set(fail_batches)
        self.status = status
        self.honour_dimensions = honour_dimensions
        self.out_of_order = out_of_order
        #: Multiplies every returned vector, so a test can hand back an
        #: unnormalised answer the way one real provider does below its native
        #: width (Q-031).
        self.scale = scale
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def embedder(self, **kwargs):
        """A real EmbeddingClient wired to this provider, with a constant key."""
        from src.embedding.client import EmbeddingClient

        settings = {
            "model": self.model,
            "dimension": self.dimension,
            "base_url": EMBED_BASE_URL,
            "http": self.client(),
            "token": lambda: "fake-key",
        }
        settings.update(kwargs)
        return EmbeddingClient(**settings)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not request.url.path.endswith("/embeddings"):
            return httpx.Response(
                404, json={"error": {"message": f"no such path {request.url.path}"}}, request=request
            )
        body = json.loads(request.content)
        self.bodies.append(body)
        if len(self.bodies) - 1 in self.fail_batches:
            return httpx.Response(
                self.status,
                json={"error": {"message": "the provider is unavailable"}},
                request=request,
            )
        texts = body["input"]
        width = int(body.get("dimensions") or self.dimension) if self.honour_dimensions else self.dimension
        entries = [
            {
                "object": "embedding",
                "index": position,
                "embedding": [component * self.scale for component in self._vector(text, width)],
            }
            for position, text in enumerate(texts)
        ]
        if self.out_of_order:
            entries = list(reversed(entries))
        payload: dict = {
            "object": "list",
            "data": entries,
            "model": self.served_model if self.served_model is not None else body["model"],
            "usage": {
                "prompt_tokens": sum(token_count(text) for text in texts),
                "total_tokens": sum(token_count(text) for text in texts),
            },
        }
        if self.served_provider:
            payload["provider"] = self.served_provider
        return httpx.Response(200, json=payload, request=request)

    def _vector(self, text: str, width: int) -> list[float]:
        vector = list(vector_for(text, self.dimension))
        if width == len(vector):
            return vector
        # A width the caller asked for and the fake was not built around: cut
        # or pad, so a test can make the provider disagree with the store.
        return (vector + [0.0] * width)[:width]


# ── A whole ingest pipeline, wired to the four fakes ──────────────────────
#
# The pipeline feature is the one that cannot be proven a step at a time: its
# criteria are about which steps ran and which did not, and "the reader was
# never called" is only observable from the far end. So these helpers build the
# real `app.ingest.Pipeline` against the same transport-level fakes every other
# feature uses, and a test asserts against what each fake was asked for.

#: The one Google credential a deployment's Google-run stages name. Spelled
#: once because the source, the reader and the bucket really are opened by one
#: service-account key — and spelled into each section rather than shared
#: between them, which is the point of the setting being per stage.
GOOGLE_CREDENTIAL_REFERENCE = "env:GOOGLE_SERVICE_ACCOUNT_KEY"

#: A complete deployment whose far ends are all faked. The dimension and the
#: bucket match the fixtures rather than production, and nothing else differs
#: from `apps/example/config.yaml`.
INGEST_DEPLOYMENT: dict[str, dict] = {
    "source": {
        "type": "gdrive",
        "folder_ids": ["root-folder"],
        "credentials_secret": GOOGLE_CREDENTIAL_REFERENCE,
    },
    # One module reads and cuts, because it is one call — so one `type` and one
    # credential, named here for both halves (see CLAUDE.md).
    "process": {
        "type": "docai",
        "credentials_secret": GOOGLE_CREDENTIAL_REFERENCE,
        "read": {
            "processor_id": DOCAI_PROCESSOR,
            "processor_version": DOCAI_VERSION,
            "artifact_bucket": ARTIFACT_BUCKET,
        },
        "chunk": {"size": 800},
    },
    "embed": {
        "model": EMBED_MODEL,
        "dimension": TEST_DIMENSION,
        "context_window": 32768,
        "base_url": EMBED_BASE_URL,
        "allowed_hosts": ["openrouter.ai"],
        "credentials_secret": "env:OPENROUTER_KEY",
    },
    "db": {"url": "postgresql://localhost/rag"},
    "telemetry": {"client": TEST_CLIENT},
    "api": {
        "credentials_secret": API_CREDENTIAL_REFERENCE,
        "allowed_hosts": [ALLOWED_HOST],
    },
    # The tool surface names the credential that opens it (AC-297). The same
    # reference as the routes' here, which is what the first deployment writes:
    # the point of repeating it is the line that changes the day the
    # assistant's token has to be revocable on its own.
    "mcp": {"credentials_secret": API_CREDENTIAL_REFERENCE},
}

#: The same deployment on the free route. One value differs from the one above
#: in what a person reads — `process.type` — and what follows from it is the
#: whole of the second module: no processor to name, no version to pin, no
#: cutting knob, and no region, because nothing leaves this process.
LOCAL_DEPLOYMENT: dict[str, dict] = {
    **INGEST_DEPLOYMENT,
    # One line, and every absence is the same fact: this module reaches no
    # provider, so it names no credential, no bucket, no processor, no region
    # and no cutting knob. What it read is kept in the archive itself (AC-272).
    "process": {"type": "pdftext"},
}

#: A production-shaped deployment: real widths, real endpoint, nothing faked.
#: What `test_config_contract` and the embedding tests build a boot from, and
#: deliberately not the same document as INGEST_DEPLOYMENT — that one's width
#: and bucket match the fixtures.
FULL_DEPLOYMENT: dict[str, dict] = {
    "source": {
        "type": "gdrive",
        "folder_ids": ["folder-123"],
        "credentials_secret": GOOGLE_CREDENTIAL_REFERENCE,
    },
    "process": {
        "type": "docai",
        "credentials_secret": GOOGLE_CREDENTIAL_REFERENCE,
        "read": {
            "processor_id": "projects/x/processors/y",
            "processor_version": DOCAI_VERSION,
            "artifact_bucket": "modular-rag-artifacts",
        },
        "chunk": {"size": 800},
    },
    "embed": {
        "model": EMBED_MODEL,
        "dimension": 2048,
        "context_window": 32768,
        "base_url": EMBED_BASE_URL,
        "allowed_hosts": ["openrouter.ai"],
        "credentials_secret": "env:OPENROUTER_KEY",
    },
    "db": {"url": "postgresql://localhost/rag"},
    "telemetry": {"client": "example"},
    "api": {
        "credentials_secret": "env:API_TOKEN",
        "allowed_hosts": ["rag.example.internal"],
    },
    "mcp": {"credentials_secret": "env:API_TOKEN"},
}

#: Where a test's configuration file is written. One directory for the whole
#: session: the file is rewritten per call, and a test that wanted to keep one
#: would be asserting over a path rather than over a deployment.
_CONFIG_DIR = pathlib.Path(tempfile.mkdtemp(prefix="modular-rag-config-"))

#: Every section name, for turning a legacy setting name back into a path.
_SECTIONS: tuple[str, ...] = (
    "source", "process", "embed",
    "db", "search", "rerank", "api", "usage", "telemetry", "schedule",
)

#: What a legacy name's first word used to be, and the path it names now. The
#: settings of a module moved under the stage that runs it — `gdrive` into
#: `source`, `docai` into `read` — and then the read and cut halves moved again,
#: under the one `process` that runs both. Roughly a hundred call sites spell
#: the old names. They say exactly the same thing, so they are translated here
#: rather than rewritten: this helper is a convenience for writing a test, and
#: the service itself knows only the sections above.
_MOVED_SECTIONS: dict[str, tuple[str, ...]] = {
    "gdrive": ("source",),
    "docai": ("process", "read"),
    "read": ("process", "read"),
    "chunk": ("process", "chunk"),
}

#: The settings that sit on `process` itself rather than in one of its halves,
#: whichever legacy word a test spells them by. `RAG_READ_CREDENTIALS_SECRET`
#: and `RAG_CHUNK_TYPE` now name the same two settings they always meant — one
#: credential and one module — which is the whole point of the nesting.
_PROCESS_OWN: frozenset[str] = frozenset({"type", "credentials_secret", "region"})

#: The settings that MOVED when every window went under the surface that serves
#: it. A test still spells them the way the environment once did — the legacy
#: convenience `_path_of` exists for — and each one now names more than one
#: path, because one flat setting used to mean several things at once:
#: `max_document_pieces` was the window AND its ceiling, and a window named for
#: "the deployment" has to reach both surfaces or the boot's cross-surface check
#: refuses a pair the test never meant to invert.
_MOVED_SETTINGS: "dict[str, tuple[tuple[str, ...], ...]]" = {
    # `extra_params` became the body of a request the deployment shapes.
    "embed_extra_params": (("embed", "request", "body"),),
    "embed_request": (("embed", "request"),),
    # `query_instruction` became one half of a two-sided `prompts`.
    "embed_query_instruction": (("embed", "prompts", "query"),),
    "embed_prompts": (("embed", "prompts"),),
    "search_result_count": (
        ("api", "defaults", "search", "limit"),
        ("mcp", "defaults", "search", "limit"),
    ),
    "search_max_result_count": (
        ("api", "limits", "search", "limit"),
        ("mcp", "limits", "search", "limit"),
    ),
    "search_brief_length": (
        ("api", "defaults", "search", "brief_length"),
        ("mcp", "defaults", "search", "brief_length"),
    ),
    "api_max_document_pieces": (
        ("api", "defaults", "fetch", "limit"),
        ("api", "limits", "fetch", "limit"),
        ("mcp", "defaults", "fetch", "limit"),
        ("mcp", "limits", "fetch", "limit"),
    ),
    "api_assistant_document_pieces": (
        ("mcp", "defaults", "fetch", "limit"),
        ("mcp", "limits", "fetch", "limit"),
    ),
    "api_passage_window": (
        ("api", "defaults", "fetch", "passage_window"),
        ("mcp", "defaults", "fetch", "passage_window"),
    ),
}



def _paths_of(name: str) -> "tuple[tuple[str, ...], ...]":
    """Every path one legacy override name now writes to.

    One name, several paths, because the flat setting it spells was one word
    for what are now a default, a ceiling, and sometimes both surfaces. The
    broader name is applied before the narrower one (see ``write_config``), so
    a test that sets the deployment's window and then the assistant's gets the
    pair it asked for rather than whichever kwarg came last.
    """
    bare = name.removeprefix("RAG_").lower()
    if bare in _MOVED_SETTINGS:
        return _MOVED_SETTINGS[bare]
    return (_path_of(name),)


def _path_of(name: str) -> "tuple[str, ...]":
    """``RAG_CHUNK_SIZE`` as the path it names — ``("process", "chunk", "size")``.

    The overrides these helpers accept are spelled the way the environment
    spelled them, and that is a convenience for writing a test rather than the
    service's vocabulary. The service itself knows only sections and settings.
    """
    bare = name.removeprefix("RAG_").lower()
    for section in sorted((*_SECTIONS, *_MOVED_SECTIONS), key=len, reverse=True):
        if not bare.startswith(f"{section}_"):
            continue
        setting = bare[len(section) + 1 :]
        path = _MOVED_SECTIONS.get(section, (section,))
        if path[0] == "process" and setting in _PROCESS_OWN:
            return ("process", setting)
        return (*path, setting)
    raise KeyError(f"{name!r} names no section; the sections are {', '.join(_SECTIONS)}")


def _fields_of(path: "tuple[str, ...]") -> dict:
    """Every field one section can carry, a pluggable stage's variants merged.

    A section is either one model or a discriminated union of them. Merged
    rather than picked, because these helpers spell a setting by name and do
    not know which module the document under construction selected.
    """
    from typing import get_args

    from src.config import _MODULES

    declared = dict(_MODULES)[path[0]]
    members = get_args(get_args(declared)[0]) if get_args(declared) else (declared,)
    fields: dict = {}
    for member in members:
        if hasattr(member, "model_fields"):
            fields.update(member.model_fields)
    # Walk into the sub-section the path names, if it names one: `process` holds
    # a `read` and a `chunk`, and a setting's own annotation lives on those.
    for step in path[1:-1]:
        declared = fields.get(step)
        if declared is None or not hasattr(declared.annotation, "model_fields"):
            return {}
        fields = dict(declared.annotation.model_fields)
    return fields


def _coerce(path: "tuple[str, ...]", value: object) -> object:
    """A string standing in for a mapping or a list, decoded as the file would hold it.

    The environment could only carry `{"a":"b"}` as text, and five call sites
    still write it that way. The file carries a real mapping, so the string is
    decoded here — in the test helper, never in the service, which must not
    learn to accept configuration as JSON.
    """
    field = _fields_of(path).get(path[-1])
    if field is None or not isinstance(value, str):
        return value
    annotation = str(field.annotation)
    if value[:1] in "{[" and ("dict" in annotation or "list" in annotation):
        return json.loads(value)
    return value


def write_config(document: "dict[str, dict]", **overrides) -> pathlib.Path:
    """One deployment written as the file the service reads, and its path.

    ``overrides`` are legacy setting names — `RAG_CHUNK_SIZE=1200` — merged over
    ``document`` per setting, so overriding one member of a section keeps the
    rest of it. ``None`` removes a setting instead of writing it, which is what
    a test used to say with `delenv`.
    """
    merged = _deep_copy(document)
    # Broadest first: a name that writes four paths is the deployment-wide word
    # and a name that writes two is the surface's own, so applying them in that
    # order lets a test say "this deployment windows at five, and the assistant
    # at three" without depending on the order it passed its keyword arguments.
    ordered = sorted(overrides.items(), key=lambda item: -len(_paths_of(item[0])))
    for name, value in ordered:
        for path in _paths_of(name):
            if value is None:
                # What `delenv` used to say: the file does not name this, so the
                # default applies. A section left empty is dropped, because an
                # empty mapping and an absent one must mean the same thing.
                _unset(merged, path)
                continue
            _set(merged, path, _coerce(path, value))
    merged = _pruned(merged)
    location = _CONFIG_DIR / "deployment.yaml"
    location.write_text(yaml.safe_dump(merged, sort_keys=True, allow_unicode=True))
    return location


def without(document: "dict[str, dict]", *paths: str) -> "dict[str, dict]":
    """``document`` with the named settings removed — ``without(d, "chunk.size")``.

    A whole section goes with a bare section name. This is how a test says "the
    file omits this", which used to be a `delenv` and now has to be an edit to
    the document, because there is nothing else to unset.
    """
    pruned = _deep_copy(document)
    for path in paths:
        _unset(pruned, tuple(path.split(".")))
    return pruned


def _deep_copy(document: "dict[str, dict]") -> "dict[str, dict]":
    return {
        key: (_deep_copy(value) if isinstance(value, dict) else value)
        for key, value in document.items()
    }


def _set(document: dict, path: "tuple[str, ...]", value: object) -> None:
    """Write one setting at ``path``, creating the sections above it."""
    node = document
    for step in path[:-1]:
        node = node.setdefault(step, {})
    node[path[-1]] = value


def _unset(document: dict, path: "tuple[str, ...]") -> None:
    """Remove whatever ``path`` names — a setting, a sub-section or a section."""
    node = document
    for step in path[:-1]:
        node = node.get(step)
        if not isinstance(node, dict):
            return
    node.pop(path[-1], None)


def _pruned(document: "dict[str, dict]") -> "dict[str, dict]":
    """The document with every empty SECTION dropped, and no deeper.

    An empty section and an absent one have to mean the same thing, or a test
    that unset the last setting of a section would be writing a section the
    file does not otherwise contain.

    A sub-section is different and deliberately kept: `process: {chunk: {}}`
    says the cut half is present and states nothing, which is what makes the
    refusal name the setting inside it rather than the half it sits in. Pruning
    it would turn "you did not give me a size" into "you did not give me a
    chunk", which sends somebody looking one level too high.
    """
    return {key: value for key, value in document.items() if value or not isinstance(value, dict)}


def deployment_config(document: "dict[str, dict] | None" = None, **overrides):
    """One deployment's configuration, out of one file and nothing else."""
    return load_config(write_config(INGEST_DEPLOYMENT if document is None else document, **overrides))


def ingest_config(monkeypatch, **overrides):
    """One deployment's configuration, out of one file and nothing else.

    ``monkeypatch`` stays in the signature because every caller passes it. A
    test that boots the whole application hands the written path to
    ``main_app(path=...)``: a deployment names itself with ``$APP`` and reads
    ``apps/<app>/config.yaml``, and a test's document lives outside that tree.

    The list of settings to clear between tests is gone: a document written per
    call cannot leak a value into the next test the way a variable could.
    """
    return load_config(write_config(INGEST_DEPLOYMENT, **overrides))


def pdf_file(external_id: str, *, name: str = "Contrato", content: bytes = b"%PDF-1.7\n", **kwargs):
    """An uploaded PDF in the fake Drive, checksummed so its marker is its content.

    A PDF rather than a Drive-authored document because the reprocessing
    decision is about content changing, and only an uploaded file has a
    checksum the source itself computes over the bytes (ASM-012).
    """
    entry = dict(
        id=external_id,
        name=name,
        mimeType="application/pdf",
        md5Checksum=hashlib.md5(content).hexdigest(),
        size=len(content),
        parents=("root-folder",),
        content=content,
    )
    entry.update(kwargs)
    return DriveFile(**entry)


def contract_payload(*, clauses: "Sequence[str]" = ("Cláusula primeira: do objeto.",), pages=(1, 1)):
    """The reader's output for one short contract: a heading and its clauses.

    Built through the same helpers every mapping test uses, so what the
    pipeline maps is the shape the provider actually returns rather than a
    convenient stand-in.
    """
    blocks = [text_block("h1", "Contrato", pages=pages, kind="heading-1")]
    chunks = []
    for position, clause in enumerate(clauses, start=1):
        block_id = f"p{position}"
        blocks.append(text_block(block_id, clause, pages=pages))
        chunks.append(reader_chunk(f"c{position}", clause, [block_id], pages=pages))
    return reader_payload(blocks, chunks)


@dataclass
class IngestWorld:
    """Every far end the pipeline talks to, and the pipeline itself."""

    config: object
    drive: FakeDrive
    storage: FakeObjectStorage
    docai: FakeDocumentAI
    embeddings: FakeEmbeddings
    pipeline: object

    @property
    def reads(self) -> int:
        """How many documents the reader was actually asked to take.

        The number three of the four criteria in this feature are about, and it
        is counted at the far end on purpose: a run that skipped the reader and
        a run that called it and threw the answer away are indistinguishable
        from the caller's own results.
        """
        return sum(
            len(body["inputDocuments"]["gcsDocuments"]["documents"])
            for body in self.docai.submissions
        )

    @property
    def embedded(self) -> int:
        """How many texts were sent to the embedding provider."""
        return sum(len(body["input"]) for body in self.embeddings.bodies)


def ingest_world(
    monkeypatch,
    *,
    files: "Sequence[DriveFile]" = (),
    outputs: "dict[str, dict] | None" = None,
    prices: "PriceBook | None" = None,
    hooks=None,
    embeddings: "FakeEmbeddings | None" = None,
    storage: "FakeObjectStorage | None" = None,
    deployment: "dict[str, dict] | None" = None,
    extract=None,
    **env,
) -> IngestWorld:
    """A real pipeline over fake far ends, ready to be handed a connection.

    ``deployment`` is which document the configuration comes out of, so a test
    can build the same world on the free route by passing
    :data:`LOCAL_DEPLOYMENT`. The reader step follows the module the document
    names, the way ``build_pipeline`` does — a test that named the free module
    and got the paid reader would prove nothing about either.

    ``extract`` replaces the free module's own reading of a PDF's text layer,
    for the tests that are about what the pipeline does with the answer rather
    than about `pypdf`.
    """
    from src.embedding.pinning import pinned_client
    from src.ingest import DocumentAIReader, LocalTextReader, Pipeline
    from src.ingest_hooks import NO_HOOKS
    from src.processing.artifact import ArchiveStore
    from src.processing.docai import ChunkingOptions

    config = load_config(write_config(deployment or INGEST_DEPLOYMENT, **env))
    drive = FakeDrive(files)
    # One bucket outlives a deployment: the kept reader output is what makes a
    # re-cut cost two cents (principle P-009), so a test that reconfigures and
    # runs again passes the same bucket back in.
    storage = storage or FakeObjectStorage(
        getattr(config.read, "artifact_bucket", ARTIFACT_BUCKET)
    )
    docai = FakeDocumentAI(rounds_until_done=0, storage=storage, outputs=outputs or {})
    provider = embeddings or FakeEmbeddings(
        model=config.embed.model, dimension=config.embed.dimension
    )
    ticks = iter(range(0, 10_000))
    # The store follows the module, the way `build_pipeline` does: the free
    # route keeps what it read in the archive itself and never touches the
    # bucket, so a test that named it and got the fake bucket would prove
    # nothing about either.
    store = ArchiveStore() if config.process.type == "pdftext" else storage.store()
    reader = (
        LocalTextReader(store=store, **({"extract": extract} if extract else {}))
        if config.process.type == "pdftext"
        else _paid_reader_for(config, docai=docai, storage=storage, ticks=ticks)
    )
    pipeline = Pipeline(
        config=config,
        source=drive_source(drive, folder_ids=config.source.folder_ids),
        reader=reader,
        store=store,
        embedder=pinned_client(config, http=provider.client(), token=lambda: "fake-key"),
        prices=prices if prices is not None else PriceBook(configured_prices()),
        hooks=hooks if hooks is not None else NO_HOOKS,
        clock=lambda: next(ticks),
    )
    return IngestWorld(
        config=config,
        drive=drive,
        storage=storage,
        docai=docai,
        embeddings=provider,
        pipeline=pipeline,
    )


def _paid_reader_for(config, *, docai, storage, ticks):
    """The paid module's reader step, wired from the same document."""
    from src.ingest import DocumentAIReader
    from src.processing.docai import ChunkingOptions

    return DocumentAIReader(
        reader=docai.reader(
            processor_id=config.read.processor_id,
            processor_version=config.read.processor_version,
            region=config.process.region,
            chunking=ChunkingOptions(
                size=config.chunk.size,
                cutter=config.process.type,
                include_ancestor_headings=config.chunk.include_headings,
            ),
            enable_table_annotation=config.read.enable_table_annotation,
            enable_image_annotation=config.read.enable_image_annotation,
            max_concurrent_batches=config.read.max_concurrent_batches,
        ),
        store=storage.store(),
        max_documents_per_batch=config.read.max_documents_per_batch,
        page_limit=config.read.page_limit,
        max_document_bytes=config.read.max_document_bytes,
        clock=lambda: next(ticks),
    )


# ── The surface, over a real archive ──────────────────────────────────────
#
# The two surfaces are proven against the real application: the real routes,
# the real middleware, the real tool server, and a real PostgreSQL underneath.
# What is faked is what a test cannot be allowed to reach — the embedding
# provider and the reranking model — and both are faked at the seam the
# deployment itself injects, never by stubbing a use case. A test that stubbed
# the search would prove that the surface calls something, which is not what
# any of these criteria say.

def fake_resolver(reference: str) -> str:
    """A resolver that knows this deployment's two references and no others.

    Injected rather than reaching the process's own environment, which is the
    seam ``resolver_over`` exists for — see src/secrets.py.
    """
    known = {
        API_CREDENTIAL_REFERENCE: API_TOKEN,
        "env:OPENROUTER_KEY": EMBED_TOKEN,
    }
    if reference not in known:
        raise KeyError(f"no such secret: {reference}")
    return known[reference]


class FakeQueryEmbedder:
    """Turns a question into the same deterministic vector the corpus uses.

    ``degrees`` pins the direction so a test can say which pieces the question
    should be near; ``fails_with`` is how the credential-rejected and
    provider-unreachable halves of AC-132 are produced without a network.
    """

    def __init__(self, *, degrees: float = 0.0, fails_with: Exception | None = None) -> None:
        self.degrees = degrees
        self.fails_with = fails_with
        self.questions: list[str] = []

    def embed_query(self, question: str) -> tuple[float, ...]:
        self.questions.append(question)
        if self.fails_with is not None:
            raise self.fails_with
        return tuple(direction(self.degrees))


class FakeScorer:
    """A reranking model that scores by a table the test writes.

    Scores are looked up by the passage's text; anything unnamed scores zero, so
    a test says only what it cares about.
    """

    def __init__(self, model: str, scores: "dict[str, float] | None" = None) -> None:
        self.model = model
        self.scores = scores or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, question: str, passages: "Sequence[str]") -> list[float]:
        self.calls.append((question, tuple(passages)))
        return [float(self.scores.get(passage, 0.0)) for passage in passages]


def api_config(monkeypatch, dsn: str, **overrides):
    """A deployment configured to serve, pointed at the test archive.

    ``RAG_DB_URL`` defaults to the test archive and is still overridable, which
    is how the unreachable-archive halves of AC-132 and AC-138 are produced.
    """
    return ingest_config(monkeypatch, **{"RAG_DB_URL": dsn, **overrides})


def api_config_path(monkeypatch, dsn: str, **overrides):
    """The same deployment, and the path to the file it was written to.

    For the tests that boot the real application: `main_app` reads the file
    itself, so it needs the path rather than the loaded configuration.
    """
    return write_config(INGEST_DEPLOYMENT, **{"RAG_DB_URL": dsn, **overrides})


def api_service(config, *, embedder=None, scorer=None, checks=None):
    """The wiring one process holds, with the far ends faked.

    Built through ``app.main.build_service`` so the real secret resolution and
    the real pinned embedding client are exercised, then the embedder is
    replaced — the deployment's own construction is what a test wants proven,
    and a live provider is what it cannot have.
    """
    from dataclasses import replace as _replace

    from src.main import build_service

    service = build_service(
        config,
        resolver=fake_resolver,
        http=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
        scorer=scorer,
    )
    changes: dict[str, object] = {"embedder": embedder or FakeQueryEmbedder()}
    if checks is not None:
        changes["checks"] = checks
    return _replace(service, **changes)


@pytest.fixture
def api_world(monkeypatch, schema_dsn: str):
    """Build the real application over the test archive, and a client for it.

    Returns a callable so a test can choose its own knobs, its own embedder and
    its own reranking model without a fixture per combination.
    """
    from fastapi.testclient import TestClient

    from src.main import create_app

    built: list[TestClient] = []

    def build(*, embedder=None, scorer=None, checks=None, **overrides):
        config = api_config(monkeypatch, schema_dsn, **overrides)
        service = api_service(config, embedder=embedder, scorer=scorer, checks=checks)
        client = TestClient(
            create_app(service),
            base_url=f"http://{ALLOWED_HOST}",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        )
        # Entered, not merely constructed: the lifespan is what starts the tool
        # transport's session manager (ASM-063), and a client that skips it
        # tests a surface whose mounted half was never switched on.
        client.__enter__()
        built.append(client)
        return client, service

    yield build
    for client in built:
        client.__exit__(None, None, None)


@pytest.fixture
def served_corpus(schema_dsn: str) -> Iterator[dict[str, int]]:
    """A small archive, committed, with its model version activated.

    Committed rather than rolled back because the application under test opens
    its own connections: a corpus living inside another transaction is a corpus
    the routes cannot see. Every row is removed afterwards, in one statement per
    table, so the next test still starts from nothing.
    """
    from src.store_versions import activate_version

    with psycopg.connect(schema_dsn) as setup:
        written = index_corpus(setup, API_CORPUS)
        activate_version(setup, TEST_STAMP)
        setup.commit()
    try:
        yield written
    finally:
        with psycopg.connect(schema_dsn) as cleanup, cleanup.cursor() as cursor:
            cursor.execute("DELETE FROM search_events")
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM active_model_version")
            cleanup.commit()
        # Deleting the rows does not give the approximate index back. pgvector's
        # HNSW keeps a graph entry per dead tuple, so a shared schema that has
        # served a few hundred of these fixtures hands the next dense search a
        # graph it spends its whole `ef_search` budget walking — the leg then
        # returns nothing, MAX_MATCH_DISTANCE drops what little it found, and a
        # test that passes alone fails in the suite. Rebuilding here is what
        # keeps "nothing accumulates" true of the index as well as the tables.
        with psycopg.connect(schema_dsn) as rebuild:
            rebuild.autocommit = True
            with rebuild.cursor() as cursor:
                cursor.execute("REINDEX INDEX chunks_embedding_idx")


#: The corpus every surface test searches. Two contracts and one that is gone,
#: with directions far enough apart that a question aimed at one does not
#: accidentally find the other.
API_CORPUS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        "drive-rescisao",
        "Contrato de locação — Sala 402",
        (
            ChunkSpec(
                0,
                "Cláusula décima: a rescisão antecipada exige aviso prévio de trinta dias.",
                degrees=0,
                pages=(4, 5),
                headings=("Da rescisão",),
            ),
            ChunkSpec(
                1,
                "Cláusula décima primeira: a multa por rescisão corresponde a três aluguéis.",
                degrees=5,
                pages=(5, 5),
                headings=("Da rescisão", "Da multa"),
            ),
        ),
    ),
    DocumentSpec(
        "drive-reajuste",
        "Contrato de locação — Loja térrea",
        (
            ChunkSpec(
                0,
                "Cláusula quinta: o reajuste anual seguirá o índice acordado entre as partes.",
                # Sixty degrees, not ninety: the dense leg drops anything at or
                # past a right angle as pointing away from the question
                # (``MAX_MATCH_DISTANCE``), and a corpus whose second document is
                # invisible to every search would make "the defaults returned
                # everything" true by accident.
                degrees=60,
                pages=(2, 2),
                headings=("Do reajuste",),
            ),
        ),
    ),
    DocumentSpec(
        "drive-removido",
        "Contrato rescindido",
        (
            ChunkSpec(
                0,
                "Cláusula décima: a rescisão antecipada exige aviso prévio de trinta dias.",
                degrees=0,
                pages=(1, 1),
                headings=("Da rescisão",),
            ),
        ),
        gone=True,
    ),
)
