"""Health and readiness (T-138).

AC-138 is proven in tests/test_spec_api_and_mcp.py — see CLAUDE.md. What this
file covers is the verdict object itself and the one case the criterion does not
name: a process that somehow recorded no verdict at all.
"""

from __future__ import annotations

from conftest import FakeQueryEmbedder

from src.routes.health import StartupChecks


def test_a_passing_verdict_reads_as_ok_and_a_failed_one_as_degraded():
    assert StartupChecks(passed=True).status == "ok"
    assert StartupChecks(passed=False, detail="archive unreachable").status == "degraded"


def test_a_process_with_no_verdict_at_all_is_reported_as_not_passing(api_world):
    """"Nobody checked" and "the checks passed" are different, and one is not safe."""
    client, _ = api_world()
    del client.app.state.startup_checks

    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["checks_passed"] is False
    assert "left no verdict" in response.json()["detail"]


def test_the_health_route_answers_before_anything_has_ever_been_searched(api_world):
    """An orchestrator calls it on a cold process with an empty archive."""
    client, _ = api_world(embedder=FakeQueryEmbedder())

    assert client.get("/healthz").json()["status"] == "ok"


def test_the_health_route_names_no_document_and_no_count(api_world, served_corpus):
    """It is unauthenticated, so anything of the archive in it is a leak."""
    client, _ = api_world()

    body = client.get("/healthz").text

    assert "drive-" not in body
    assert set(client.get("/healthz").json()) == {"status", "checks_passed", "detail"}


def test_a_degraded_process_still_serves_the_route_rather_than_refusing_to_exist(api_world):
    """An archive that was down at startup resolves without a redeploy."""
    client, _ = api_world(checks=StartupChecks(passed=False, detail="connection refused"))

    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["detail"] == "connection refused"
