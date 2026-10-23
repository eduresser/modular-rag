from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from src.boot import RESERVED_BODY_KEYS, RESERVED_HEADERS

EMBEDDINGS_PATH = "embeddings"

ROLE_DOCUMENT = "document"
ROLE_QUERY = "query"


NORM_TOLERANCE = 1e-3

DIMENSION_SETTING = "embed.dimension"


class EmbeddingError(Exception):

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RequestPinning(Protocol):

    def apply(self, body: dict[str, Any]) -> dict[str, Any]:
        ...

    def verify(self, payload: Mapping[str, Any], *, model: str) -> None:
        ...


@dataclass(frozen=True, slots=True)
class EmbeddingResult:

    vectors: tuple[tuple[float, ...], ...]
    model: str
    tokens: int

    def __len__(self) -> int:
        return len(self.vectors)


class Embedder(Protocol):

    model: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        ...


@dataclass(frozen=True)
class EmbeddingClient:

    model: str
    dimension: int
    base_url: str
    http: httpx.Client
    token: Callable[[], str]
    query_prompt: str = ""
    document_prompt: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, str] = field(default_factory=dict)
    body_extras: Mapping[str, Any] = field(default_factory=dict)
    pinning: RequestPinning | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise EmbeddingError("an embedding client has to name the model it calls")
        if self.dimension < 1:
            raise EmbeddingError(
                f"an embedding client has to name a positive width, got {self.dimension}"
            )
        if not self.base_url.strip():
            raise EmbeddingError("an embedding client has to name the endpoint it calls")
        self._reject_unpermitted_params()

    def _reject_unpermitted_params(self) -> None:
        if "dimensions" in self.body_extras:
            raise EmbeddingError(
                f"extra parameter 'dimensions' is set to "
                f"{self.body_extras['dimensions']!r} while the width is already "
                f"{self.dimension}; the width is a first-class setting because it "
                f"decides the store's column type (ASM-031) — set {DIMENSION_SETTING} "
                "and remove it from the request body"
            )
        for key in sorted(self.body_extras):
            if key in RESERVED_BODY_KEYS:
                raise EmbeddingError(
                    f"request body key {key!r} is set and this service decides it: "
                    f"{RESERVED_BODY_KEYS[key]}"
                )
        for header in sorted(self.headers):
            if header.lower() in RESERVED_HEADERS:
                raise EmbeddingError(
                    f"request header {header!r} is set and this service decides it: "
                    f"{RESERVED_HEADERS[header.lower()]}"
                )


    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{EMBEDDINGS_PATH}"

    def render(self, text: str, role: str) -> str:
        prompt = self.query_prompt if role == ROLE_QUERY else self.document_prompt
        return f"{prompt}{text}" if prompt else text

    def body(self, texts: Sequence[str], role: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [self.render(text, role) for text in texts],
            "dimensions": self.dimension,
            **dict(self.body_extras),
        }
        if self.pinning is not None:
            payload = self.pinning.apply(payload)
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token()}",
            "Content-Type": "application/json",
            **dict(self.headers),
        }


    def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._embed(texts, ROLE_DOCUMENT)

    def embed_query(self, question: str) -> tuple[float, ...]:
        return self._embed([question], ROLE_QUERY).vectors[0]

    def _embed(self, texts: Sequence[str], role: str) -> EmbeddingResult:
        if not texts:
            raise EmbeddingError("an embedding request with no text is a request nobody has to make")
        try:
            response = self.http.post(
                self.url,
                json=self.body(texts, role),
                headers=self._headers(),
                params=dict(self.params) or None,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"the embedding provider could not be reached: {exc}") from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"the embedding provider refused a batch of {len(texts)}: {_message(response)}",
                status=response.status_code,
            )
        return self._read(response.json(), len(texts))

    def _read(self, payload: Mapping[str, Any], expected: int) -> EmbeddingResult:
        served = str(payload.get("model") or "")
        if served and not _same_model(served, self.model):
            raise EmbeddingError(
                f"asked {self.model!r} for {expected} vector(s) and {served!r} answered; "
                "a vector from another model in this index is a search silently comparing "
                "things that are not comparable (AC-052)"
            )
        if self.pinning is not None:
            self.pinning.verify(payload, model=self.model)
        entries = payload.get("data")
        if not isinstance(entries, list) or len(entries) != expected:
            raise EmbeddingError(
                f"asked for {expected} vector(s) and the provider returned "
                f"{len(entries) if isinstance(entries, list) else 'no list of'} embeddings"
            )
        vectors = tuple(
            self._vector(entry) for entry in sorted(entries, key=lambda item: item.get("index", 0))
        )
        usage = payload.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        return EmbeddingResult(vectors=vectors, model=served or self.model, tokens=tokens)

    def _vector(self, entry: Mapping[str, Any]) -> tuple[float, ...]:
        raw = entry.get("embedding")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            raise EmbeddingError("the provider returned an entry carrying no embedding")
        if len(raw) != self.dimension:
            raise EmbeddingError(
                f"asked {self.model!r} for {self.dimension} dimensions and it returned "
                f"{len(raw)}; the store's column is {self.dimension} wide and a schema "
                "migration is not a retry"
            )
        return unit(tuple(float(component) for component in raw))


def unit(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(component) * float(component) for component in vector))
    if norm == 0.0:
        raise EmbeddingError(
            "the provider returned an all-zero vector, which has no direction and no "
            "meaningful distance to anything"
        )
    if abs(norm - 1.0) <= NORM_TOLERANCE:
        return tuple(float(component) for component in vector)
    return tuple(float(component) / norm for component in vector)


def _same_model(served: str, asked: str) -> bool:
    served = served.split(":", 1)[0].casefold()
    asked = asked.split(":", 1)[0].casefold()
    if served == asked:
        return True
    if "/" in served and "/" in asked:
        return False
    return served.rsplit("/", 1)[-1] == asked.rsplit("/", 1)[-1]


def _message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return f"HTTP {response.status_code}: {error['message']}"
    return f"HTTP {response.status_code}"
