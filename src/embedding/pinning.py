from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from src.config import AppConfig
from src.embedding.client import EmbeddingClient, EmbeddingError


FIELD_ROUTING = "provider"
FIELD_ALLOW_FALLBACKS = "allow_fallbacks"
FIELD_ORDER = "order"


class ProviderSubstituted(EmbeddingError):
    ...


def is_routed(base_url: str, routers: Sequence[str]) -> bool:
    return urlparse(base_url).hostname in routers


@dataclass(frozen=True, slots=True)
class RouterPinning:

    provider: str = ""

    def apply(self, body: dict[str, Any]) -> dict[str, Any]:
        routing: dict[str, Any] = {FIELD_ALLOW_FALLBACKS: False}
        if self.provider:
            routing[FIELD_ORDER] = [self.provider]
        return {**body, FIELD_ROUTING: routing}

    def verify(self, payload: Mapping[str, Any], *, model: str) -> None:
        served = str(payload.get(FIELD_ROUTING) or "")
        if not self.provider or not served:
            return
        if served.casefold() != self.provider.casefold():
            raise ProviderSubstituted(
                f"the request pinned {self.provider!r} for {model!r} and {served!r} "
                "answered; a vector from a substituted provider joins an index that "
                "records only the model and the width, so nothing downstream could "
                "ever tell the two apart"
            )


def pinned_client(
    config: AppConfig,
    *,
    http: httpx.Client,
    token: Callable[[], str],
) -> EmbeddingClient:
    routers = config.embed.allowed_hosts
    routed = is_routed(config.embed.base_url, routers)
    if config.embed.provider and not routed:
        raise EmbeddingError(
            f"a provider to pin ({config.embed.provider!r}) is configured while "
            f"{config.embed.base_url!r} is not a router, so there is nothing to pin and "
            "nothing that would read the constraint; 'embed.allowed_hosts' names "
            f"{', '.join(routers) or '(none)'}"
        )
    return EmbeddingClient(
        model=config.embed.model,
        dimension=config.embed.dimension,
        base_url=config.embed.base_url,
        http=http,
        token=token,
        query_prompt=config.embed.prompts.query,
        document_prompt=config.embed.prompts.document,
        headers=config.embed.request.headers,
        params=config.embed.request.params,
        body_extras=config.embed.request.body,
        pinning=RouterPinning(provider=config.embed.provider) if routed else None,
    )
