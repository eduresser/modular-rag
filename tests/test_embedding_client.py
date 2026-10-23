"""The provider client (T-050): configuration in, vectors out, nothing hard-coded.

AC-050 and AC-051 are proven in tests/test_spec_embedding_provider.py — see
CLAUDE.md for why the proof lives there. What this file covers is everything
underneath: how the request is built, and every answer the client is right to
refuse.

The provider is faked at the transport, so what these assertions read is the
body that actually went out.
"""

import math

import httpx
import pytest

from src.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    unit,
)
from tests.conftest import TEST_DIMENSION, FakeEmbeddings, token_count, vector_for


def test_the_request_names_the_model_the_width_and_nothing_else_from_code():
    provider = FakeEmbeddings()

    provider.embedder().embed_documents(["cláusula quarta"])

    body = provider.bodies[0]
    assert body["model"] == "qwen/qwen3-embedding-8b"
    assert body["dimensions"] == TEST_DIMENSION
    assert body["input"] == ["cláusula quarta"]
    assert provider.requests[0].url.path.endswith("/embeddings")


def test_the_key_is_asked_for_on_every_request_rather_than_captured_once():
    # A run long enough to matter is a run long enough for a credential to be
    # rotated under it, so the callable is called per request, not per client.
    provider = FakeEmbeddings()
    keys = iter(["first-key", "second-key"])
    client = provider.embedder(token=lambda: next(keys))

    client.embed_documents(["um"])
    client.embed_documents(["dois"])

    assert provider.requests[0].headers["Authorization"] == "Bearer first-key"
    assert provider.requests[1].headers["Authorization"] == "Bearer second-key"


def test_a_configured_body_key_reaches_the_provider_unchanged():
    provider = FakeEmbeddings()

    provider.embedder(body_extras={"encoding_format": "float"}).embed_documents(["texto"])

    assert provider.bodies[0]["encoding_format"] == "float"


def test_a_key_nothing_here_knows_about_now_travels_instead_of_being_refused():
    """What the denylist gives up, asserted rather than left to be discovered.

    `request.body` used to be an allowlist of three keys, so a provider wanting
    a fourth was a deployment this file could not describe. It travels now —
    and so does a typo, which is the cost: `dimension` for `dimensions` is
    found by the far end instead of at boot.
    """
    provider = FakeEmbeddings()

    provider.embedder(body_extras={"input_type": "search_document"}).embed_documents(["texto"])

    assert provider.bodies[0]["input_type"] == "search_document"


def test_a_configured_header_reaches_the_provider_beside_the_ones_we_set():
    # The case that forced the whole shape: a provider wanting its credential
    # in a header it names itself.
    provider = FakeEmbeddings()

    provider.embedder(headers={"x-api-key": "the-key"}).embed_documents(["texto"])

    sent = provider.requests[0].headers
    assert sent["x-api-key"] == "the-key"
    assert sent["Authorization"].startswith("Bearer ")


def test_a_header_this_service_sets_may_not_be_overwritten():
    # A provider wanting the credential elsewhere names ITS header; writing
    # `Authorization` here would be the same credential in two places, and the
    # one that won would depend on merge order.
    provider = FakeEmbeddings()

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(headers={"Authorization": "Basic smuggled"})

    assert "Authorization" in str(excinfo.value)
    assert "embed.credentials_secret" in str(excinfo.value)
    assert provider.requests == []


def test_a_body_key_this_service_guarantees_is_refused_before_any_request():
    provider = FakeEmbeddings()

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(body_extras={"allow_fallbacks": True})

    # AC-052: a substituted provider is invisible from every row, so this one
    # is not a preference a deployment gets to express.
    assert "allow_fallbacks" in str(excinfo.value)
    assert provider.requests == []


def test_the_width_may_not_be_smuggled_in_through_the_request_body():
    # ASM-031: the width decides the store's column type, so it has one home.
    # Two places to say how wide a vector is means one of them disagreeing with
    # a column that can only match one.
    provider = FakeEmbeddings()

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(body_extras={"dimensions": "1024"})

    message = str(excinfo.value)
    assert "embed.dimension" in message
    assert "1024" in message


def test_a_question_and_a_passage_are_not_sent_the_same_way():
    # ASM-033. The chosen model's card asks for a task instruction on the query
    # and nothing on the passage; fixing either to one value would silently
    # degrade half of every operation. The ROLE is still the code's decision —
    # two methods, never one with a parameter — and what a question wears is
    # now the deployment's (`embed.prompts`).
    instruction = "Instruct: retrieve the passages that answer it\nQuery: "
    provider = FakeEmbeddings()
    client = provider.embedder(query_prompt=instruction)

    client.embed_documents(["o locatário pagará"])
    client.embed_query("quem paga o aluguel?")

    assert provider.bodies[0]["input"] == ["o locatário pagará"]
    assert provider.bodies[1]["input"] == [f"{instruction}quem paga o aluguel?"]


def test_a_model_that_wants_both_prefixes_gets_both():
    """The half `query_instruction` could not express, and it is not hypothetical.

    The e5 family prefixes `query: ` on a question and `passage: ` on a passage,
    and its model card says omitting either degrades quality. With one field
    there was no way to write the second, so moving to it would have lost half
    the prefix in silence — ASM-033's failure in a third costume.
    """
    provider = FakeEmbeddings()
    client = provider.embedder(query_prompt="query: ", document_prompt="passage: ")

    client.embed_documents(["o locatário pagará"])
    client.embed_query("quem paga o aluguel?")

    assert provider.bodies[0]["input"] == ["passage: o locatário pagará"]
    assert provider.bodies[1]["input"] == ["query: quem paga o aluguel?"]


def test_a_model_that_makes_no_distinction_gets_no_prefix_on_either():
    # An empty instruction states "this model makes no such distinction through
    # this endpoint" — Gemini's case, which HAS task types the OpenAI-shaped
    # endpoint carries none of. It used to be an ABSENT key in a table, which
    # reads the same as a model nobody got round to registering.
    provider = FakeEmbeddings(model="openai/text-embedding-3-large")
    client = provider.embedder(model="openai/text-embedding-3-large", query_prompt="")

    client.embed_query("quem paga o aluguel?")

    assert provider.bodies[0]["input"] == ["quem paga o aluguel?"]


def test_the_answer_is_read_by_its_index_and_not_by_arrival_order():
    # A provider that answers out of order would otherwise mis-attribute a whole
    # batch, and every piece would still look perfectly valid.
    provider = FakeEmbeddings(out_of_order=True)

    result = provider.embedder().embed_documents(["primeiro", "segundo", "terceiro"])

    assert result.vectors[0] == vector_for("primeiro")
    assert result.vectors[2] == vector_for("terceiro")


def test_a_provider_that_answers_with_another_model_is_refused():
    provider = FakeEmbeddings(served_model="google/gemini-embedding-001")

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder().embed_documents(["texto"])

    assert "gemini" in str(excinfo.value)


def test_a_variant_suffix_on_the_served_slug_is_not_a_substitution():
    # A gateway names how it served a model, not what it served.
    provider = FakeEmbeddings(served_model="qwen/qwen3-embedding-8b:free")

    result = provider.embedder().embed_documents(["texto"])

    assert len(result) == 1


def test_the_vendor_s_own_capitalisation_of_the_slug_is_not_a_substitution():
    # Measured against the live gateway: its catalogue slug is
    # 'qwen/qwen3-embedding-8b' and the answer names the upstream vendor's
    # spelling of the same weights, 'Qwen/Qwen3-Embedding-8B'. Comparing byte
    # for byte refused every real request while holding the right vectors.
    provider = FakeEmbeddings(served_model="Qwen/Qwen3-Embedding-8B")

    result = provider.embedder(model="qwen/qwen3-embedding-8b").embed_documents(["texto"])

    assert len(result) == 1


def test_a_missing_vendor_prefix_is_not_a_substitution():
    # The same lesson as the capitalisation above, learned a second time and
    # measured on the same gateway: asked for 'openai/text-embedding-3-large'
    # it answers 'text-embedding-3-large'. The prefix is the router's own and
    # whether it comes back is the upstream vendor's habit — refusing that
    # refused every OpenAI-family model on this router, with the right vectors
    # in hand and a whole re-embedding run failing document by document.
    provider = FakeEmbeddings(served_model="text-embedding-3-large")

    result = provider.embedder(model="openai/text-embedding-3-large").embed_documents(["texto"])

    assert len(result) == 1


def test_two_vendors_of_the_same_name_are_still_two_models():
    # The line the rule above must not cross. One prefix missing is a naming
    # habit; two different prefixes are two different routes, which is exactly
    # the substitution AC-052 exists to catch.
    provider = FakeEmbeddings(served_model="azure/text-embedding-3-large")

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(model="openai/text-embedding-3-large").embed_documents(["texto"])

    assert "AC-052" in str(excinfo.value)


def test_a_different_model_is_still_a_substitution_however_it_is_spelled():
    provider = FakeEmbeddings(served_model="Qwen/Qwen3-Embedding-4B")

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(model="qwen/qwen3-embedding-8b").embed_documents(["texto"])

    assert "AC-052" in str(excinfo.value)


def test_a_vector_of_the_wrong_width_is_refused_here_and_not_by_the_column():
    # The store's column would report a type error and the fault would be
    # blamed on the wrong module three steps away.
    provider = FakeEmbeddings(honour_dimensions=False, dimension=TEST_DIMENSION)

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder(dimension=TEST_DIMENSION + 1).embed_documents(["texto"])

    assert str(TEST_DIMENSION) in str(excinfo.value)


def test_an_unnormalised_answer_comes_back_as_a_unit_vector():
    # Q-031: one provider in play returns unnormalised vectors at any width but
    # its native one, and documents renormalising as the caller's job. Cosine is
    # scale-invariant, which is exactly why this would sit in the index for
    # months without a symptom.
    provider = FakeEmbeddings(scale=17.0)

    vector = provider.embedder().embed_documents(["texto"]).vectors[0]

    assert math.isclose(math.sqrt(sum(component**2 for component in vector)), 1.0, abs_tol=1e-9)


def test_an_all_zero_vector_is_refused_rather_than_stored():
    provider = FakeEmbeddings(scale=0.0)

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder().embed_documents(["texto"])

    assert "direction" in str(excinfo.value)


def test_a_refusal_carries_the_providers_own_message():
    provider = FakeEmbeddings(fail_batches=(0,), status=503)

    with pytest.raises(EmbeddingError) as excinfo:
        provider.embedder().embed_documents(["texto"])

    message = str(excinfo.value)
    assert "503" in message
    assert "unavailable" in message


def test_a_transport_failure_is_an_embedding_error_and_not_an_httpx_one():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = EmbeddingClient(
        model="qwen/qwen3-embedding-8b",
        dimension=TEST_DIMENSION,
        base_url="https://openrouter.ai/api/v1",
        http=httpx.Client(transport=httpx.MockTransport(refuse)),
        token=lambda: "fake-key",
    )

    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_documents(["texto"])

    assert "no route to host" in str(excinfo.value)


def test_the_tokens_the_provider_reports_are_carried_back():
    provider = FakeEmbeddings()

    result = provider.embedder().embed_documents(["duas palavras", "três palavras aqui"])

    assert result.tokens == token_count("duas palavras") + token_count("três palavras aqui")


def test_a_client_without_a_model_a_width_or_an_endpoint_is_refused():
    provider = FakeEmbeddings()

    for broken in ({"model": "  "}, {"dimension": 0}, {"base_url": ""}):
        with pytest.raises(EmbeddingError):
            provider.embedder(**broken)


def test_an_empty_request_is_refused_before_it_is_sent():
    provider = FakeEmbeddings()

    with pytest.raises(EmbeddingError):
        provider.embedder().embed_documents([])

    assert provider.requests == []


def test_unit_leaves_an_already_normalised_vector_alone():
    already = vector_for("texto")

    assert unit(already) == already
