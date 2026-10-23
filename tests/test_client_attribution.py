"""Unit-level coverage for src/otel.py.

AC-168 is proven in tests/test_spec_usage_observability.py — see CLAUDE.md for
why the proof lives there. This file covers the exporter being inert until it
is configured, and the promise that telemetry never takes down the thing it is
measuring.
"""

import pytest

from src.config import TelemetrySettings
from src.otel import CLIENT_ATTRIBUTE, CollectedSpans, Telemetry, telemetry_from

CLIENT = "example"
ENDPOINT = "http://collector.internal:4318"


def test_a_deployment_without_an_endpoint_exports_nothing():
    # This is how every deployment ships (ASM-072): the instrumentation is in
    # place and switched off, so turning it on later is a variable.
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, exporter=collected)

    assert telemetry.exporting is False
    with telemetry.span("search"):
        pass

    assert collected.spans == []


def test_an_endpoint_without_a_transport_also_exports_nothing():
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT)

    assert telemetry.exporting is False
    with telemetry.span("search"):
        pass  # must not raise


def test_whitespace_is_not_an_endpoint():
    assert Telemetry(client=CLIENT, endpoint="   ", exporter=CollectedSpans()).exporting is False


def test_a_configured_deployment_emits_one_span_per_block():
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with telemetry.span("search"):
        pass
    with telemetry.span("read"):
        pass

    assert [span.name for span in collected.spans] == ["search", "read"]
    assert all(span.client == CLIENT for span in collected.spans)
    assert all(span.attributes[CLIENT_ATTRIBUTE] == CLIENT for span in collected.spans)


def test_the_client_attribute_cannot_be_overwritten_by_the_caller():
    # One collector serving every deployment groups by this attribute, so a
    # caller passing its own value would silently merge two clients' signals.
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with telemetry.span("search", client="somebody-else"):
        pass

    assert collected.spans[0].attributes[CLIENT_ATTRIBUTE] == CLIENT


def test_attributes_discovered_during_the_work_reach_the_span():
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with telemetry.span("read", document="rescisao") as attributes:
        attributes["pages"] = 412

    span = collected.spans[0]
    assert span.attributes["document"] == "rescisao"
    assert span.attributes["pages"] == 412


def test_a_span_is_emitted_for_a_failure_and_the_exception_still_reaches_the_caller():
    # A failed operation is exactly the one worth having a span for, and the
    # failure belongs to the caller.
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with pytest.raises(TimeoutError):
        with telemetry.span("read"):
            raise TimeoutError("Document AI took too long")

    span = collected.spans[0]
    assert span.failed
    assert "TimeoutError" in span.error
    assert "took too long" in span.error


def test_a_successful_span_carries_no_error():
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with telemetry.span("search"):
        pass

    assert collected.spans[0].failed is False
    assert collected.spans[0].error is None


def test_a_span_measures_the_block():
    collected = CollectedSpans()
    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=collected)

    with telemetry.span("search"):
        sum(range(50_000))

    assert collected.spans[0].duration_ms > 0


def test_a_collector_that_throws_does_not_take_down_the_operation():
    # The failure mode that makes operators switch instrumentation off
    # entirely: a search failing because the observability endpoint moved.
    def broken(span):
        raise ConnectionRefusedError("collector moved")

    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=broken)

    with telemetry.span("search") as attributes:
        attributes["hits"] = 3  # must not raise


def test_a_collector_that_throws_on_a_failing_operation_keeps_the_real_exception():
    def broken(span):
        raise ConnectionRefusedError("collector moved")

    telemetry = Telemetry(client=CLIENT, endpoint=ENDPOINT, exporter=broken)

    # The caller sees the timeout it caused, not the collector's problem.
    with pytest.raises(TimeoutError):
        with telemetry.span("read"):
            raise TimeoutError("Document AI took too long")


def test_telemetry_without_a_client_is_refused():
    for blank in ("", "   "):
        with pytest.raises(ValueError) as excinfo:
            Telemetry(client=blank)
        assert "AC-168" in str(excinfo.value)


def test_it_is_built_from_the_deployments_settings():
    collected = CollectedSpans()

    inert = telemetry_from(TelemetrySettings(client=CLIENT), collected)
    live = telemetry_from(TelemetrySettings(client=CLIENT, export_endpoint=ENDPOINT), collected)

    assert inert.client == live.client == CLIENT
    assert inert.exporting is False
    assert live.exporting is True
