"""Provider pinning (T-051): the request that refuses a substitute.

AC-052 is proven in tests/test_spec_embedding_provider.py — see CLAUDE.md for
why the proof lives there. This file covers the routing block itself, the check
on the answer, and the constructor that is the only sane way to build a client
for a router.
"""

import pytest

from src.config import load_config
from src.embedding.client import EmbeddingError
from src.embedding.pinning import (
    FIELD_ALLOW_FALLBACKS,
    FIELD_ORDER,
    FIELD_ROUTING,
    ProviderSubstituted,
    RouterPinning,
    is_routed,
    pinned_client,
)
from tests.conftest import FULL_DEPLOYMENT, FakeEmbeddings, deployment_config


def _config(monkeypatch, **overrides):
    """One deployment out of one file. ``monkeypatch`` is vestigial and kept
    because every caller passes it."""
    return deployment_config(FULL_DEPLOYMENT, **overrides)


def test_the_gateway_is_a_router_and_a_direct_provider_is_not():
    # The routers are `embed.allowed_hosts`, passed in rather than kept here:
    # this module used to hold its own copy of the one string `src.boot` also
    # held, which is one list of hosts that can disagree with itself.
    routers = ["openrouter.ai"]

    assert is_routed("https://openrouter.ai/api/v1", routers) is True
    assert is_routed("https://api.openai.com/v1", routers) is False


def test_fallback_is_disabled_even_when_no_provider_is_named():
    # Not a setting, and there is nothing that turns it off: an empty provider
    # still means exactly one provider, the router's own first choice.
    body = RouterPinning().apply({"model": "qwen/qwen3-embedding-8b"})

    assert body[FIELD_ROUTING] == {FIELD_ALLOW_FALLBACKS: False}
    assert FIELD_ORDER not in body[FIELD_ROUTING]


def test_a_named_provider_becomes_the_order_the_router_must_follow():
    body = RouterPinning(provider="DeepInfra").apply({"model": "qwen/qwen3-embedding-8b"})

    assert body[FIELD_ROUTING] == {FIELD_ALLOW_FALLBACKS: False, FIELD_ORDER: ["DeepInfra"]}


def test_applying_the_routing_block_leaves_the_rest_of_the_body_alone():
    body = RouterPinning(provider="DeepInfra").apply({"model": "m", "input": ["a"], "dimensions": 8})

    assert body["model"] == "m"
    assert body["input"] == ["a"]
    assert body["dimensions"] == 8


def test_an_answer_from_the_pinned_provider_passes():
    RouterPinning(provider="DeepInfra").verify({"provider": "deepinfra"}, model="m")


def test_an_answer_from_another_provider_is_refused():
    with pytest.raises(ProviderSubstituted) as excinfo:
        RouterPinning(provider="DeepInfra").verify({"provider": "Novita"}, model="m")

    message = str(excinfo.value)
    assert "DeepInfra" in message
    assert "Novita" in message


def test_an_answer_that_names_no_provider_is_not_evidence_of_a_substitution():
    # The constraint went out on the request. Refusing every answer that omits
    # an optional field would fail every batch against a gateway that simply
    # does not report it.
    RouterPinning(provider="DeepInfra").verify({"model": "m"}, model="m")


def test_a_substitution_is_a_failed_batch_like_any_other():
    # A run has to treat it as one (AC-057): the pieces stay unembedded and
    # counted. It keeps its own name because a run hitting it repeatedly is
    # saying something quite different from a rate limit.
    assert issubclass(ProviderSubstituted, EmbeddingError)


def test_the_client_a_deployment_gets_is_pinned(monkeypatch):
    provider = FakeEmbeddings()
    config = _config(monkeypatch, RAG_EMBED_DIMENSION="8", RAG_EMBED_PROVIDER="DeepInfra")

    client = pinned_client(config, http=provider.client(), token=lambda: "fake-key")
    client.embed_documents(["texto"])

    assert provider.bodies[0][FIELD_ROUTING] == {
        FIELD_ALLOW_FALLBACKS: False,
        FIELD_ORDER: ["DeepInfra"],
    }


def test_the_client_carries_the_configured_model_width_and_request(monkeypatch):
    provider = FakeEmbeddings()
    config = _config(
        monkeypatch,
        RAG_EMBED_DIMENSION="8",
        RAG_EMBED_EXTRA_PARAMS='{"encoding_format": "float"}',
    )

    client = pinned_client(config, http=provider.client(), token=lambda: "fake-key")

    assert client.model == "qwen/qwen3-embedding-8b"
    assert client.dimension == 8
    # The three places a call has, carried through from `embed.request` rather
    # than the one mapping `extra_params` used to be.
    assert client.body_extras == {"encoding_format": "float"}
    assert client.headers == {}
    assert client.params == {}


def test_pinning_a_provider_at_an_endpoint_that_routes_nowhere_is_refused(monkeypatch):
    # There would be nothing to pin and nothing that reads the constraint, and
    # the deployment would believe it was pinned.
    provider = FakeEmbeddings()
    config = _config(
        monkeypatch,
        RAG_EMBED_BASE_URL="https://api.openai.com/v1",
        RAG_EMBED_PROVIDER="DeepInfra",
    )

    with pytest.raises(EmbeddingError) as excinfo:
        pinned_client(config, http=provider.client(), token=lambda: "fake-key")

    assert "openrouter.ai" in str(excinfo.value)


def test_a_direct_provider_gets_no_routing_block(monkeypatch):
    provider = FakeEmbeddings()
    config = _config(monkeypatch, RAG_EMBED_BASE_URL="https://api.openai.com/v1", RAG_EMBED_DIMENSION="8")

    client = pinned_client(config, http=provider.client(), token=lambda: "fake-key")
    client.embed_documents(["texto"])

    assert client.pinning is None
    assert FIELD_ROUTING not in provider.bodies[0]
