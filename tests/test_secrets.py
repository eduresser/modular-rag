import pytest

from src.secrets import (
    REGISTERED_SCHEMES,
    SECRET_SUFFIX,
    SecretResolutionError,
    is_reference,
    resolve_secrets,
    resolver_over,
    scheme_of,
    target_of,
)


def test_resolves_a_reference_whatever_the_setting_is_called():
    settings = {"x-api-key": "env:AZURE_KEY", "embed.credentials_secret": "env:OPENROUTER_KEY"}

    resolved = resolve_secrets(settings, resolver=lambda ref: f"resolved:{ref}")

    assert resolved["x-api-key"] == "resolved:env:AZURE_KEY"
    assert resolved["embed.credentials_secret"] == "resolved:env:OPENROUTER_KEY"


def test_a_reference_resolves_inside_a_longer_string():
    settings = {"Authorization": "Bearer env:OPENROUTER_KEY"}

    resolved = resolve_secrets(settings, resolver=lambda ref: "sk-or-the-key")

    assert resolved["Authorization"] == "Bearer sk-or-the-key"


def test_doubling_the_separator_escapes_it_and_resolves_nothing():
    settings = {"process.read.artifact_bucket": "env::NOT_A_REFERENCE"}

    resolved = resolve_secrets(settings, resolver=lambda ref: pytest.fail("resolver must not run"))

    assert resolved["process.read.artifact_bucket"] == "env:NOT_A_REFERENCE"


def test_passes_through_settings_that_are_not_secret_references():
    settings = {"RAG_CHUNK_SIZE": "800"}

    resolved = resolve_secrets(settings, resolver=lambda ref: pytest.fail("resolver must not run"))

    assert resolved == {"RAG_CHUNK_SIZE": "800"}


def test_resolves_a_mix_of_references_and_plain_settings():
    settings = {"chunk.size": "800", "process.credentials_secret": "env:GOOGLE_KEY"}

    resolved = resolve_secrets(settings, resolver=lambda ref: "the-actual-value")

    assert resolved == {"chunk.size": "800", "process.credentials_secret": "the-actual-value"}


def test_a_resolution_failure_stops_the_boot_naming_the_setting_and_reference():
    settings = {"embed.credentials_secret": "env:MISSING_KEY"}

    def _resolver(reference: str) -> str:
        raise LookupError("secret not found")

    with pytest.raises(SecretResolutionError) as excinfo:
        resolve_secrets(settings, resolver=_resolver)

    message = str(excinfo.value)
    assert "embed.credentials_secret" in message
    assert "env:MISSING_KEY" in message


def test_a_resolution_failure_does_not_echo_a_value_resolved_earlier_in_the_same_run():
    settings = {
        "process.credentials_secret": "env:GOOD_KEY",
        "Authorization": "Bearer env:GOOD_KEY env:MISSING_KEY",
    }

    def _resolver(reference: str) -> str:
        if reference == "env:MISSING_KEY":
            raise LookupError("secret not found")
        return "super-secret-value"

    with pytest.raises(SecretResolutionError) as excinfo:
        resolve_secrets(settings, resolver=_resolver)

    assert "super-secret-value" not in str(excinfo.value)


def test_the_secret_suffix_constant_matches_the_convention_it_documents():
    assert SECRET_SUFFIX == "_secret"
    assert "embed.credentials_secret".lower().endswith(SECRET_SUFFIX)
    assert "RAG_EMBED_API_KEY_SECRET".lower().endswith(SECRET_SUFFIX)
    assert not "chunk.size".lower().endswith(SECRET_SUFFIX)


@pytest.mark.parametrize(
    "value,scheme",
    [
        ("env:OPENROUTER_KEY", "env"),
        ("gcp-sm:projects/x/secrets/y/versions/latest", "gcp-sm"),
        ("sk-or-abc123", ""),
        ("vault:secret/openrouter", "vault"),
        (800, ""),
        (None, ""),
    ],
)
def test_the_scheme_is_whatever_stands_before_the_first_separator(value, scheme):
    assert scheme_of(value) == scheme
    assert is_reference(value) is (scheme in REGISTERED_SCHEMES)


def test_a_reference_keeps_what_it_names_when_the_target_contains_separators():
    assert target_of("env:A_KEY:1") == "A_KEY:1"


def test_the_environment_backend_reads_the_variable_the_reference_names(monkeypatch):
    monkeypatch.setenv("A_KEY", "the-value")

    assert resolver_over()("env:A_KEY") == "the-value"


def test_an_unset_variable_is_a_failure_that_names_it_and_not_its_value(monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)

    with pytest.raises(SecretResolutionError) as excinfo:
        resolve_secrets({"RAG_X_SECRET": "env:A_KEY"}, resolver_over())

    assert "RAG_X_SECRET" in str(excinfo.value)
    assert "env:A_KEY" in str(excinfo.value)


def test_an_unregistered_scheme_is_refused_where_it_is_read_and_not_where_it_is_resolved():
    assert not is_reference("gcp-sm:projects/x")

    untouched = resolve_secrets({"embed.credentials_secret": "gcp-sm:projects/x"}, resolver_over())

    assert untouched["embed.credentials_secret"] == "gcp-sm:projects/x"


def test_a_backend_can_still_be_registered_by_the_caller():
    resolve = resolver_over({"vault": lambda target: f"from-vault:{target}"})

    assert resolve("vault:secret/x") == "from-vault:secret/x"
    assert resolve("env:PATH")
