"""The application skeleton and its startup gate (T-130).

AC-138 is proven in tests/test_spec_api_and_mcp.py — see CLAUDE.md. What this
file covers is what has to be true before the process serves anything: which
failures stop it, which ones it starts degraded with, and the one line whose
absence produces nothing at startup and a broken tool call afterwards (ASM-063).
"""

from __future__ import annotations

import httpx
import pytest
from conftest import (
    ALLOWED_HOST,
    API_TOKEN,
    FakeQueryEmbedder,
    api_config,
    api_config_path,
    fake_resolver,
)

from src.boot import BootCompatibilityError
from src.main import MCP_PATH, build_service, create_app, main_app, startup_checks
from src.routes.health import StartupChecks
from src.secrets import SecretResolutionError


def _http() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))


# ── The gate ──────────────────────────────────────────────────────────────


def test_the_gate_passes_against_the_archive_it_is_configured_for(monkeypatch, schema_dsn):
    config = api_config(monkeypatch, schema_dsn)

    assert startup_checks(config) == StartupChecks(passed=True)


def test_a_configuration_the_archive_disagrees_with_stops_the_process(monkeypatch, schema_dsn):
    """A dimension that disagrees with the live column is a migration, not a restart."""
    config = api_config(monkeypatch, schema_dsn, RAG_EMBED_DIMENSION="2048")

    with pytest.raises(BootCompatibilityError) as excinfo:
        startup_checks(config)

    assert "2048" in str(excinfo.value)


def test_an_archive_that_cannot_be_reached_is_a_degraded_start_and_not_a_refusal(
    monkeypatch,
):
    """It resolves without a redeploy; refusing to start would keep the service down."""
    config = api_config(monkeypatch, "postgresql://127.0.0.1:1/nowhere")

    checks = startup_checks(config)

    assert checks.passed is False
    assert "could not reach the archive" in checks.detail


# ── The wiring ────────────────────────────────────────────────────────────


def test_both_credentials_are_resolved_once_and_the_reference_never_reaches_the_client(
    monkeypatch, schema_dsn
):
    config = api_config(monkeypatch, schema_dsn)

    service = build_service(config, resolver=fake_resolver, http=_http())

    assert service.credential.token == API_TOKEN
    assert service.credential.token != config.api.credentials_secret
    # The embedding client got the resolved key, not the reference it came from.
    assert service.embedder.token() != config.embed.credentials_secret


def test_a_reference_that_cannot_be_resolved_is_a_boot_failure_naming_the_setting(
    monkeypatch, schema_dsn
):
    """Never a 401 on the first question somebody asks."""
    config = api_config(monkeypatch, schema_dsn, RAG_API_CREDENTIALS_SECRET="env:A_VARIABLE_NOBODY_SET")

    with pytest.raises(SecretResolutionError) as excinfo:
        build_service(config, resolver=fake_resolver, http=_http())

    # The setting as the deployment's file spells it. `RAG_API_CREDENTIALS_SECRET`
    # above is the test harness's own legacy override vocabulary (see
    # `write_config`), not a name this build reads.
    assert "api.credentials_secret" in str(excinfo.value)
    # And nothing it resolved is echoed into the message.
    assert API_TOKEN not in str(excinfo.value)


def test_the_embedding_client_a_deployment_gets_is_the_pinned_one(monkeypatch, schema_dsn):
    """A client built without asking whether its endpoint routes can be substituted."""
    config = api_config(monkeypatch, schema_dsn)

    service = build_service(config, resolver=fake_resolver, http=_http())

    assert service.embedder.pinning is not None


# ── The mounted sub-application (ASM-063) ─────────────────────────────────


def test_the_tool_transport_answers_once_the_application_has_started(api_world, served_corpus):
    """The mounted sub-application's own lifespan never runs; ours drives it.

    Without that one line this passes nothing: the session manager is never
    started and the first tool call fails at runtime with a healthy startup
    behind it. So this asks the protocol a real question over the real mount.
    """
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    response = client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 200
    assert "search" in response.text and "fetch" in response.text


def test_the_tool_transport_is_stateless_and_needs_no_handshake(api_world, served_corpus):
    """ASM-060: no session, so replicas scale with no sticky routing."""
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    first = client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={"Accept": "application/json, text/event-stream"},
    )
    second = client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert first.status_code == second.status_code == 200
    # Neither answer asked the caller to carry a session on to the next one.
    assert "mcp-session-id" not in {name.lower() for name in second.headers}


# ── From the environment outward ──────────────────────────────────────────


def test_the_whole_application_boots_from_one_file_alone(monkeypatch, schema_dsn):
    from fastapi.testclient import TestClient

    location = api_config_path(monkeypatch, schema_dsn)

    app = main_app(resolver=fake_resolver, http=_http(), path=str(location))

    with TestClient(app, base_url=f"http://{ALLOWED_HOST}") as client:
        assert client.get("/healthz").json() == {
            "status": "ok",
            "checks_passed": True,
            "detail": "",
        }


def test_the_application_carries_its_wiring_where_a_route_can_reach_it(
    monkeypatch, schema_dsn
):
    config = api_config(monkeypatch, schema_dsn)
    service = build_service(config, resolver=fake_resolver, http=_http())

    app = create_app(service)

    assert app.state.service is service
    assert app.state.startup_checks == service.checks


def test_a_tool_call_answers_over_the_real_transport(api_world, served_corpus):
    """Not only the listing: a call has to reach the use case through the mount."""
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    response = client.post(
        MCP_PATH,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "rescisão antecipada"}},
        },
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 200
    assert "drive-rescisao" in response.text


def test_an_unexpected_host_is_refused_by_our_gate_and_not_by_the_transport(
    api_world, served_corpus
):
    """The transport's own check answers a bare 421 whose reason lives in a log.

    That is the failure AC-137 exists to prevent, so the gate has to reach the
    mounted surface first — and the two allowlists have to be the same list.
    """
    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))

    response = client.post(
        MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        headers={"Host": "rag.someone-else.example", "Accept": "application/json"},
    )

    assert response.status_code == 400
    assert "rag.someone-else.example" in response.json()["error"]
