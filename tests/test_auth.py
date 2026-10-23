"""The credential and the host allowlist (T-135).

AC-136 and AC-137 are proven end to end in tests/test_spec_api_and_mcp.py — see
CLAUDE.md. What this file covers is the two gates on their own: what counts as a
credential, what counts as a host, and the ways a request can try to name a
user.
"""

from __future__ import annotations

import logging

import pytest
from starlette.datastructures import Headers
from starlette.requests import Request

from src.auth import (
    END_USER_HEADERS,
    Credential,
    IdentityNotAccepted,
    Unauthenticated,
    authorise,
    host_allowed,
    host_of,
    host_refusal,
    presented_token,
    refuse_end_user_identity,
)

TOKEN = "an-api-token-nobody-guesses"


def _request(path: str = "/v1/search", **headers: str) -> Request:
    raw = Headers({key.replace("_", "-"): value for key, value in headers.items()})
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": raw.raw,
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


def test_an_empty_credential_is_refused_at_construction():
    """An empty token would accept every request that sends an empty one."""
    with pytest.raises(ValueError):
        Credential(token="   ")


def test_the_comparison_accepts_the_token_and_nothing_near_it():
    credential = Credential(token=TOKEN)

    assert credential.matches(TOKEN) is True
    assert credential.matches(TOKEN + " ") is False
    assert credential.matches(TOKEN[:-1]) is False
    assert credential.matches("") is False


def test_a_bearer_token_is_read_whatever_the_case_of_the_scheme():
    assert presented_token(_request(authorization=f"Bearer {TOKEN}")) == TOKEN
    assert presented_token(_request(authorization=f"bearer {TOKEN}")) == TOKEN
    assert presented_token(_request(authorization=f"BEARER  {TOKEN} ")) == TOKEN


def test_a_missing_or_unusable_authorization_header_is_refused_with_a_reason():
    for header in ("", "Basic dXNlcjpwYXNz", f"Token {TOKEN}", "Bearer", "Bearer    "):
        with pytest.raises(Unauthenticated):
            presented_token(_request(authorization=header))


def test_a_token_in_the_query_string_is_not_a_credential():
    """Proxies log query strings; one misrouted request would leak the token."""
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/search",
            "headers": [],
            "query_string": f"token={TOKEN}".encode(),
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )

    with pytest.raises(Unauthenticated):
        presented_token(request)


@pytest.mark.parametrize("header", END_USER_HEADERS)
def test_every_way_of_naming_an_end_user_is_refused(header):
    """Principle P-014: refused rather than ignored, so nobody builds on it."""
    with pytest.raises(IdentityNotAccepted) as excinfo:
        refuse_end_user_identity(_request(**{header: "user@example.com"}))

    assert "P-014" in str(excinfo.value)


def test_naming_a_user_is_refused_before_the_credential_is_even_looked_at():
    """The problem worth reporting is the one a valid credential would not fix."""
    with pytest.raises(IdentityNotAccepted):
        authorise(_request(x_user_id="someone"), Credential(token=TOKEN))


def test_a_valid_credential_with_no_identity_header_passes():
    authorise(_request(authorization=f"Bearer {TOKEN}"), Credential(token=TOKEN))


def test_the_host_is_read_without_its_port():
    assert host_of(_request(host="rag.example.internal")) == "rag.example.internal"
    assert host_of(_request(host="rag.example.internal:8080")) == "rag.example.internal"
    assert host_of(_request(host="[::1]:8080")) == "[::1]"
    assert host_of(_request(host="[::1]")) == "[::1]"
    assert host_of(_request()) == ""


def test_only_the_first_host_of_a_folded_header_is_considered():
    """Two hosts in one header is a smuggling attempt, not a deployment with two names."""
    assert host_of(_request(host="rag.example.internal, evil.example")) == "rag.example.internal"


def test_the_allowlist_matches_exactly_and_never_by_suffix():
    allowed = ["rag.example.internal"]

    assert host_allowed("rag.example.internal", allowed) is True
    assert host_allowed("evil-rag.example.internal", allowed) is False
    assert host_allowed("rag.example.internal.evil.example", allowed) is False
    assert host_allowed("", allowed) is False


def test_a_wildcard_entry_opens_the_deployment_and_has_to_be_written_down():
    assert host_allowed("anything.example", ["*"]) is True


def test_a_refused_host_produces_a_reason_and_a_log_line(caplog):
    with caplog.at_level(logging.WARNING, logger="src.auth"):
        reason = host_refusal(_request(host="evil.example"), ["rag.example.internal"])

    assert reason is not None
    assert "evil.example" in reason and "rag.example.internal" in reason
    assert any("evil.example" in record.getMessage() for record in caplog.records)


def test_an_allowed_host_produces_no_reason_and_no_log_line(caplog):
    with caplog.at_level(logging.WARNING, logger="src.auth"):
        assert host_refusal(_request(host="rag.example.internal"), ["rag.example.internal"]) is None

    assert caplog.records == []
