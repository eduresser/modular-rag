from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

from src.config import TelemetrySettings

CLIENT_ATTRIBUTE = "client"


@dataclass(frozen=True, slots=True)
class Span:
    name: str
    client: str
    duration_ms: float
    attributes: Mapping[str, object] = field(default_factory=dict)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


Exporter = Callable[[Span], None]


class Telemetry:
    __slots__ = ("_client", "_endpoint", "_exporter")

    def __init__(self, *, client: str, endpoint: str = "", exporter: Exporter | None = None) -> None:
        if not client.strip():
            raise ValueError(
                "telemetry has to name the client this deployment serves (AC-168); "
                "an unattributed span cannot be grouped with anything"
            )
        self._client = client
        self._endpoint = endpoint.strip()
        self._exporter = exporter

    @property
    def client(self) -> str:
        return self._client

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def exporting(self) -> bool:
        return bool(self._endpoint) and self._exporter is not None

    @contextmanager
    def span(self, name: str, **attributes: object) -> Iterator[dict[str, object]]:
        carried: dict[str, object] = dict(attributes)
        started = perf_counter()
        error: str | None = None
        try:
            yield carried
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._emit(
                Span(
                    name=name,
                    client=self._client,
                    duration_ms=(perf_counter() - started) * 1000.0,
                    attributes={**carried, CLIENT_ATTRIBUTE: self._client},
                    error=error,
                )
            )

    def _emit(self, span: Span) -> None:
        if not self.exporting:
            return
        try:
            self._exporter(span)  # type: ignore[misc]
        except Exception:
            return


class CollectedSpans:
    __slots__ = ("spans",)

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def __call__(self, span: Span) -> None:
        self.spans.append(span)


def telemetry_from(settings: TelemetrySettings, exporter: Exporter | None = None) -> Telemetry:
    return Telemetry(
        client=settings.client,
        endpoint=settings.export_endpoint,
        exporter=exporter,
    )
