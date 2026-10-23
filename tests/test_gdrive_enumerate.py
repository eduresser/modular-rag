"""Unit-level coverage for src/sources/gdrive.py.

AC-010 and AC-015 are proven in tests/test_spec_source_gdrive.py — see
CLAUDE.md for why the proofs live there. This file covers what the criteria do
not name and what silently breaks anyway: the field projection on every call,
the shared-drive parameters, paging, the folder walk, and which status code
means which thing.
"""

import base64
import functools
import json
import urllib.parse

import httpx
import pytest

from src.sources.base import DocumentUnavailable, SourceError
from src.sources.gdrive import (
    ALL_DRIVES,
    FILE_FIELDS,
    SCOPE_READONLY,
    GoogleDriveSource,
    service_account_token_provider,
)
from src.sources.gdrive_export import KIND_DOCUMENT, KIND_FOLDER, KIND_SHORTCUT, KIND_SPREADSHEET
from src.sources.gdrive_throttle import QuotaBudget, RateLimited
from tests.conftest import DRIVE_FOLDER_KIND, DriveFile, FakeDrive, drive_source

FOLDER = "root-folder"


def _doc(identifier, *, parent=FOLDER, version="1", **kwargs):
    return DriveFile(identifier, identifier, KIND_DOCUMENT, version=version, parents=(parent,), **kwargs)


def test_every_call_asks_for_its_fields_and_opts_into_shared_drives():
    # ASM-014's failure mode is silence: a field nobody asked for arrives as
    # absent, not as an error. And a corpus on a shared drive is invisible to a
    # client that does not opt in — invisible reads as empty.
    drive = FakeDrive([_doc("doc-1")])
    source = drive_source(drive, folder_ids=[FOLDER])

    list(source.sweep())
    source.start_position()
    source.probe("doc-1")

    assert drive.requests
    for request in drive.requests:
        params = dict(request.url.params)
        if request.url.path.endswith("/changes/startPageToken"):
            # This one has no projection to ask for, but still needs the drives.
            assert params["supportsAllDrives"] == "true"
            continue
        assert "fields" in params, request.url
        for key, value in ALL_DRIVES.items():
            assert params[key] == value, request.url


def test_the_projection_names_every_field_the_decision_depends_on():
    # A forgotten md5Checksum looks exactly like a document with no checksum,
    # which is a document that gets re-read every night forever.
    for field in (
        "id", "name", "mimeType", "version", "md5Checksum", "trashed",
        "shortcutDetails", "webViewLink",
    ):
        assert field in FILE_FIELDS, field


def _handler_for(entry: dict):
    """A minimal Drive transport answering one file, in the API's own shape.

    Built directly rather than through ``FakeDrive``/``DriveFile`` (which does
    not carry ``webViewLink``), the same way the rate-limit-as-403 test below
    already bypasses them to control exactly what the raw JSON contains.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/drive/v3")
        if path == "/files":
            return httpx.Response(200, json={"files": [entry]}, request=request)
        if path == f"/files/{FOLDER}":
            # The watched folder is resolved against the source before anything
            # is enumerated, so even a hand-built transport has to answer for it.
            return httpx.Response(
                200,
                json={"id": FOLDER, "name": "Pasta", "mimeType": DRIVE_FOLDER_KIND},
                request=request,
            )
        return httpx.Response(200, json=entry, request=request)

    return handler


def test_a_documents_own_link_is_asked_of_the_source():
    # ASM-107: the link is a field the source hands over, not a string this
    # service composes from the identifier.
    entry = {
        "id": "doc-1", "name": "Contrato", "mimeType": KIND_DOCUMENT,
        "version": "1", "webViewLink": "https://drive.google.com/file/d/doc-1/view",
    }
    source = GoogleDriveSource(
        folder_ids=[FOLDER], http=httpx.Client(transport=httpx.MockTransport(_handler_for(entry))),
        token=lambda: "t",
    )

    found = next(iter(source.sweep()))

    assert found.link == "https://drive.google.com/file/d/doc-1/view"


def test_a_document_the_source_gives_no_link_for_carries_none():
    # A field nobody asked for arrives as absent (ASM-014); one the source
    # simply has none for must arrive absent too, never guessed at (AC-185).
    entry = {"id": "doc-1", "name": "Contrato", "mimeType": KIND_DOCUMENT, "version": "1"}
    source = GoogleDriveSource(
        folder_ids=[FOLDER], http=httpx.Client(transport=httpx.MockTransport(_handler_for(entry))),
        token=lambda: "t",
    )

    found = next(iter(source.sweep()))

    assert found.link is None


def test_a_checksum_is_the_marker_when_there_is_one():
    # For an uploaded file the checksum IS the content hash, so an unmoved
    # marker genuinely means unchanged and nothing is downloaded.
    drive = FakeDrive([_doc("uploaded", md5Checksum="deadbeef", version="9")])

    document = next(iter(drive_source(drive, folder_ids=[FOLDER]).sweep()))

    assert document.revision_marker == "deadbeef"


def test_the_version_counter_is_the_marker_when_there_is_no_checksum():
    # A document authored in the editor has no checksum and its counter moves
    # when somebody merely opens the file (ASM-012).
    drive = FakeDrive([_doc("authored", version="12")])

    document = next(iter(drive_source(drive, folder_ids=[FOLDER]).sweep()))

    assert document.revision_marker == "12"


def test_a_document_with_neither_has_no_marker_rather_than_a_fake_one():
    drive = FakeDrive([DriveFile("bare", "bare", KIND_DOCUMENT, parents=(FOLDER,))])

    document = next(iter(drive_source(drive, folder_ids=[FOLDER]).sweep()))

    assert document.revision_marker is None


def test_the_sweep_walks_into_subfolders():
    drive = FakeDrive(
        [
            DriveFile("sub", "Subpasta", KIND_FOLDER, parents=(FOLDER,)),
            DriveFile("deep", "Mais fundo", KIND_FOLDER, parents=("sub",)),
            _doc("top"),
            _doc("middle", parent="sub"),
            _doc("bottom", parent="deep"),
        ]
    )

    found = {document.external_id for document in drive_source(drive, folder_ids=[FOLDER]).sweep()}

    assert found == {"top", "middle", "bottom"}


def test_a_folder_is_never_yielded_as_a_document():
    drive = FakeDrive([DriveFile("sub", "Subpasta", KIND_FOLDER, parents=(FOLDER,)), _doc("doc-1")])

    found = [document.external_id for document in drive_source(drive, folder_ids=[FOLDER]).sweep()]

    assert found == ["doc-1"]


def test_the_sweep_follows_every_page():
    drive = FakeDrive([_doc(f"doc-{n}") for n in range(5)], page_size=2)

    found = {document.external_id for document in drive_source(drive, folder_ids=[FOLDER]).sweep()}

    assert len(found) == 5
    listings = [r for r in drive.requests if r.url.path.endswith("/files")]
    assert len(listings) >= 3, "a paged listing was not followed"


def test_a_trashed_document_is_reported_as_removed_not_as_present():
    # As far as the archive is concerned a trashed file and a removed one are
    # the same event, and neither is proof of anything yet.
    drive = FakeDrive([_doc("doc-1", trashed=True)])
    drive.push_change("doc-1")
    source = drive_source(drive, folder_ids=[FOLDER])

    page = source.changes_since("t0")

    assert page.documents == ()
    assert page.removed == ("doc-1",)


def test_the_feed_reports_the_position_to_resume_from_and_whether_it_is_exhausted():
    drive = FakeDrive([_doc(f"doc-{n}") for n in range(3)], page_size=2)
    for n in range(3):
        drive.push_change(f"doc-{n}")
    source = drive_source(drive, folder_ids=[FOLDER])

    first = source.changes_since("t0")
    assert first.exhausted is False
    assert first.next_position == "t2"

    second = source.changes_since(first.next_position)
    assert second.exhausted is True
    assert second.next_position == "t3"


def test_an_incremental_run_without_a_position_is_refused():
    # A first run takes one with start_position() before sweeping, so that a
    # document changed mid-sweep does not fall in the gap.
    source = drive_source(FakeDrive(), folder_ids=[FOLDER])

    with pytest.raises(SourceError) as excinfo:
        source.changes_since(None)

    assert "start_position()" in str(excinfo.value)


def test_a_source_without_a_folder_is_refused():
    with pytest.raises(SourceError):
        GoogleDriveSource(folder_ids=[], http=httpx.Client(), token=lambda: "t")


def test_a_shortcut_whose_target_the_projection_dropped_blames_the_projection():
    # Which is the honest diagnosis: a shortcut without a target id is a
    # forgotten field, not a broken shortcut (ASM-014).
    drive = FakeDrive([DriveFile("shortcut", "Atalho", KIND_SHORTCUT, parents=(FOLDER,))])

    with pytest.raises(SourceError) as excinfo:
        list(drive_source(drive, folder_ids=[FOLDER]).sweep())

    assert "ASM-014" in str(excinfo.value)


def test_a_missing_document_and_a_denied_one_raise_the_same_type_with_different_flags():
    drive = FakeDrive([_doc("absent", missing=True), _doc("denied", denied=True)])
    source = drive_source(drive, folder_ids=[FOLDER])

    with pytest.raises(DocumentUnavailable) as absent:
        source.probe("absent")
    with pytest.raises(DocumentUnavailable) as denied:
        source.probe("denied")

    assert absent.value.denied is False
    assert denied.value.denied is True


def test_a_trashed_document_probed_directly_is_proof_of_deletion():
    """A document a probe still gets a 200 for, but which the source marks
    trashed, is treated as gone rather than as present — trash is the second
    proof of deletion, distinct from a bare permission refusal.
    """
    drive = FakeDrive([_doc("doc-1", trashed=True)])
    source = drive_source(drive, folder_ids=[FOLDER])

    with pytest.raises(DocumentUnavailable) as excinfo:
        source.probe("doc-1")

    assert excinfo.value.denied is False
    assert "trash" in str(excinfo.value).lower()


def test_probing_something_that_is_not_a_document_says_so():
    drive = FakeDrive([DriveFile("sub", "Subpasta", KIND_FOLDER, parents=(FOLDER,))])
    source = drive_source(drive, folder_ids=[FOLDER])

    with pytest.raises(DocumentUnavailable) as excinfo:
        source.probe("sub")

    assert "not a document" in str(excinfo.value)


def test_a_rate_refusal_is_retried_rather_than_failing_the_sweep():
    drive = FakeDrive([_doc("doc-1")])
    drive.rate_limit_first = 2
    clock = {"now": 0.0}
    budget = QuotaBudget(
        now=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    source = drive_source(drive, folder_ids=[FOLDER], budget=budget)

    found = [document.external_id for document in source.sweep()]

    assert found == ["doc-1"]
    assert clock["now"] > 0, "a refusal was retried without backing off"


def test_a_rate_limit_that_never_lets_up_raises_rather_than_looping():
    drive = FakeDrive([_doc("doc-1")])
    drive.rate_limit_first = 99
    clock = {"now": 0.0}
    budget = QuotaBudget(
        now=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)
    )
    source = drive_source(drive, folder_ids=[FOLDER], budget=budget, max_attempts=3)

    with pytest.raises(RateLimited):
        list(source.sweep())


def test_a_rate_limit_reported_as_forbidden_is_not_read_as_access_denied():
    # Drive reports its own rate limits as 403 with a reason, not only as 429.
    # Reading that as "access denied" would make a busy sweep look like a
    # permission problem and leave documents indexed forever.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"errors": [{"reason": "userRateLimitExceeded"}]}},
            request=request,
        )

    clock = {"now": 0.0}
    source = GoogleDriveSource(
        folder_ids=[FOLDER],
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        token=lambda: "t",
        budget=QuotaBudget(
            now=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)
        ),
        max_attempts=2,
    )

    with pytest.raises(RateLimited):
        source.probe("doc-1")


def test_the_bearer_token_is_asked_for_on_every_request():
    # So a token that expires mid-sweep is refreshed by whatever supplies it,
    # rather than being captured once at construction.
    asked = {"n": 0}

    def token() -> str:
        asked["n"] += 1
        return f"token-{asked['n']}"

    # A shortcut forces a second call, so there is more than one request to
    # compare.
    drive = FakeDrive(
        [
            DriveFile("shortcut", "Atalho", KIND_SHORTCUT, parents=(FOLDER,), targetId="target"),
            _doc("target", parent="elsewhere"),
        ]
    )
    drive.ensure_folder(FOLDER)
    source = GoogleDriveSource(folder_ids=[FOLDER], http=drive.client(), token=token)

    list(source.sweep())

    assert len(drive.requests) > 1
    assert asked["n"] == len(drive.requests)
    tokens = [request.headers["Authorization"] for request in drive.requests]
    assert tokens[0] == "Bearer token-1"
    assert len(set(tokens)) == len(tokens), "the token was captured once instead of asked for"


def test_a_spreadsheet_and_a_document_are_both_enumerated():
    drive = FakeDrive(
        [_doc("doc-1"), DriveFile("sheet-1", "Planilha", KIND_SPREADSHEET, version="1", parents=(FOLDER,))]
    )

    kinds = {d.external_id: d.kind for d in drive_source(drive, folder_ids=[FOLDER]).sweep()}

    assert kinds == {"doc-1": KIND_DOCUMENT, "sheet-1": KIND_SPREADSHEET}


# ── The bearer token, and the transport it goes out on ────────────────────
#
# Nothing else in this suite mints a token: every other test injects one, which
# is precisely how the refresh path came to need a package the project does not
# declare and stay green for eight features. What is asserted here is that the
# exchange goes out on **the client the caller passed in** — a transport of the
# library's own would leave the fake endpoint below with nothing recorded, and
# would need `requests` to get that far.

TOKEN_URI = "https://oauth2.googleapis.com/token"
SERVICE_ACCOUNT_EMAIL = "rag-ingest@modular-rag.iam.gserviceaccount.com"
SCOPE_STORAGE = "https://www.googleapis.com/auth/devstorage.read_write"


@functools.lru_cache(maxsize=1)
def _key_json() -> str:
    """A real, throwaway service-account key, generated rather than committed.

    Generated because the signing is real — google-auth signs the grant
    assertion with this key — and because a private key sitting in a
    version-controlled file is the one thing principle P-012 forbids, test-only
    or not. Cached, because generating one costs more than every assertion
    below put together.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "modular-rag",
            "private_key_id": "kid-1",
            "private_key": pem,
            "client_email": SERVICE_ACCOUNT_EMAIL,
            "client_id": "1",
            "token_uri": TOKEN_URI,
        }
    )


def _claims(assertion: str) -> dict:
    """The grant assertion's payload, decoded without a JWT library."""
    payload = assertion.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class _TokenEndpoint:
    """The provider's token endpoint, answering without a network."""

    def __init__(self, *, access_token: str = "minted", expires_in: int = 3600) -> None:
        self.access_token = access_token
        self.expires_in = expires_in
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) != TOKEN_URI:
            return httpx.Response(404, json={"error": "not the token endpoint"})
        return httpx.Response(
            200,
            json={
                "access_token": self.access_token,
                "expires_in": self.expires_in,
                "token_type": "Bearer",
            },
        )


def test_the_token_exchange_goes_out_on_the_client_it_was_given():
    endpoint = _TokenEndpoint(access_token="ya29.minted")
    provider = service_account_token_provider(_key_json(), http=endpoint.client())
    assert provider() == "ya29.minted"
    assert len(endpoint.requests) == 1


def test_the_grant_carries_exactly_the_scopes_that_were_asked_for():
    endpoint = _TokenEndpoint()
    scopes = (SCOPE_READONLY, SCOPE_STORAGE)
    service_account_token_provider(_key_json(), http=endpoint.client(), scopes=scopes)()
    form = urllib.parse.parse_qs(endpoint.requests[0].content.decode())
    assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
    claims = _claims(form["assertion"][0])
    assert claims["scope"] == " ".join(scopes)
    assert claims["iss"] == SERVICE_ACCOUNT_EMAIL
    assert "sub" not in claims  # no domain-wide delegation (ASM-010)


def test_a_token_that_is_still_valid_is_not_minted_twice():
    endpoint = _TokenEndpoint()
    provider = service_account_token_provider(_key_json(), http=endpoint.client())
    assert provider() == provider()
    assert len(endpoint.requests) == 1


def test_a_token_past_its_life_is_minted_again():
    # One second, which is inside google-auth's own clock skew, so the token is
    # never considered live. A first sweep of this corpus outlives a real token,
    # which is the case this covers.
    endpoint = _TokenEndpoint(expires_in=1)
    provider = service_account_token_provider(_key_json(), http=endpoint.client())
    provider()
    provider()
    assert len(endpoint.requests) == 2


def test_the_watched_folders_are_resolved_once_and_then_remembered():
    # ASM-129: the resolution pass is what makes the access check and the
    # nesting answer affordable, and it is affordable because it is paid once
    # per source rather than once per sweep.
    drive = FakeDrive(
        [
            DriveFile(FOLDER, "Pasta", DRIVE_FOLDER_KIND),
            DriveFile("other", "Outra pasta", DRIVE_FOLDER_KIND),
            _doc("doc-1", parent=FOLDER),
            _doc("doc-2", parent="other"),
        ]
    )
    source = drive_source(drive, folder_ids=[FOLDER, "other"])

    assert source.folders == (FOLDER, "other")
    lookups = [r for r in drive.requests if r.url.path.endswith(f"/files/{FOLDER}")]
    assert len(lookups) == 1

    before = len(drive.requests)
    list(source.sweep())
    list(source.sweep())
    again = [
        r for r in drive.requests[before:] if r.url.path.endswith(f"/files/{FOLDER}")
    ]
    assert again == [], "the folders were resolved again instead of being remembered"


def test_the_listed_order_is_kept_and_a_repeat_costs_nothing():
    # The deployment's own order, deduplicated: listing a folder twice is a
    # redundancy (ASM-128), and nothing downstream should have to care which
    # order somebody wrote them in.
    drive = FakeDrive(
        [
            DriveFile(FOLDER, "Pasta", DRIVE_FOLDER_KIND),
            DriveFile("other", "Outra pasta", DRIVE_FOLDER_KIND),
        ]
    )
    source = drive_source(drive, folder_ids=["other", FOLDER, "other"])

    assert source.configured_folders == ("other", FOLDER)
    assert source.folders == ("other", FOLDER)
