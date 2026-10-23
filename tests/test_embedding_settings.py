"""The embedding module's environment contract (T-056).

AC-050 — the provider is selected entirely by configuration — is proven in
tests/test_spec_embedding_provider.py; see CLAUDE.md. What this file covers is
the settings themselves: which of them a deployment must state, what the rest
default to, that the names give nothing away about which vendor is behind them,
and whether the committed files a new deployment is copied from actually declare
them.

A missing key in ``apps/_template/.env.example`` is the kind of defect that
surfaces months later as the second client's boot failing on a Friday, so the
committed files are read here rather than described.
"""

import pathlib

import pytest
import yaml

from src.config import ConfigError, EmbedSettings
from tests.conftest import FULL_DEPLOYMENT, deployment_config

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_EXAMPLE = ROOT / "src" / "embedding" / "config.example.yaml"
COMMITTED_DEPLOYMENTS = (
    ROOT / "apps" / "base" / "config.yaml",
    ROOT / "apps" / "example" / "config.yaml",
    ROOT / "apps" / "_template" / "config.yaml",
)

# The spelling the committed example files still use. It lives here rather
# than in src/config.py because the service no longer knows an environment
# name for a setting — T-173 turns those files into the base configuration and
# this constant goes with them.
_LEGACY_PREFIX = "RAG_"
EMBED_PREFIX = f"{_LEGACY_PREFIX}EMBED_"

# Names that would tie a variable to one vendor. AC-050 is that two deployments
# pointed at different providers differ by values and by nothing else — a
# variable called RAG_EMBED_OPENROUTER_KEY would make changing provider a
# rename, which is a release.
VENDOR_WORDS = ("openrouter", "openai", "qwen", "voyage", "google", "gemini", "cohere", "anthropic")


def _declared(path: pathlib.Path, section: str = "embed") -> set[str]:
    """The settings one committed file names for a section.

    Read from the file rather than described, for the reason it always was: a
    key missing from what a new deployment is copied from surfaces months later
    as the second client's boot failing on a Friday.
    """
    document = yaml.safe_load(path.read_text()) or {}
    return dict(document.get(section) or {})


def _expected() -> set[str]:
    return set(EmbedSettings.model_fields)


def _env(monkeypatch, **overrides):
    """One deployment out of one file, with ``None`` meaning "the file omits it".

    ``monkeypatch`` stays in the signature because every caller passes it.
    """
    return deployment_config(FULL_DEPLOYMENT, **overrides)


def test_a_deployment_must_name_the_model_the_width_the_endpoint_and_the_credential(monkeypatch):
    for setting in ("MODEL", "DIMENSION", "BASE_URL", "CREDENTIALS_SECRET"):
        with pytest.raises(ConfigError) as excinfo:
            _env(monkeypatch, **{f"{EMBED_PREFIX}{setting}": None})

        message = str(excinfo.value)
        assert "[embed]" in message
        assert setting.lower() in message
        assert "missing required setting" in message


def test_the_credential_is_a_reference_and_the_suffix_is_what_resolves_it():
    # Principle P-012: a committed file names which secrets exist and where they
    # live, never one of them. app/secrets.py resolves by the _SECRET suffix.
    assert "credentials_secret" in EmbedSettings.model_fields


def test_fallback_is_not_a_setting(monkeypatch):
    # AC-052 is not a preference. There is a setting for *which* provider to
    # pin to and none for whether to pin at all, because a deployment that could
    # turn it off is a deployment that could mix two vector spaces in one index.
    assert not [field for field in EmbedSettings.model_fields if "fallback" in field]

    with pytest.raises(ConfigError):
        _env(monkeypatch, RAG_EMBED_ALLOW_FALLBACKS="true")


def test_the_provider_is_empty_unless_a_deployment_has_a_reason(monkeypatch):
    config = _env(monkeypatch)

    assert config.embed.provider == ""
    assert _env(monkeypatch, RAG_EMBED_PROVIDER="DeepInfra").embed.provider == "DeepInfra"


def test_the_batch_size_ships_far_below_the_request_ceiling(monkeypatch):
    # Bounded by what a failure throws away (AC-057), not by what the provider
    # accepts: every piece of a failed batch is left unembedded.
    config = _env(monkeypatch)

    assert config.embed.batch_size == 64
    assert config.embed.batch_size < config.embed.max_inputs_per_request


def test_the_request_defaults_to_adding_nothing_in_any_of_its_three_places(monkeypatch):
    request = _env(monkeypatch).embed.request

    assert (request.headers, request.params, request.body) == ({}, {}, {})


def test_a_typo_in_an_embedding_setting_stops_the_boot(monkeypatch):
    # AC-002, in this namespace: a setting nobody reads and nobody reports is a
    # corpus embedded at a width the deployment does not think it chose.
    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_EMBED_DIMENSON="1024")

    message = str(excinfo.value)
    assert "[embed]" in message
    assert "dimenson" in message


def test_no_setting_name_gives_away_which_vendor_is_behind_it():
    for field in EmbedSettings.model_fields:
        assert not [word for word in VENDOR_WORDS if word in field.lower()], field


def test_the_module_env_example_declares_every_setting_and_no_extra():
    assert set(_declared(MODULE_EXAMPLE)) == _expected()


def test_every_required_setting_is_supplied_by_exactly_one_layer():
    """The invariant the two layers rest on, and it is finer than it looks.

    A required setting is either shared — the base names it, and a client that
    says nothing gets it — or it is the client's own, and the base must NOT
    name it: a placeholder would let a client that forgot to override one start
    against the wrong corpus, while an omission stops the boot and names the
    setting (AC-001, AC-170).

    So neither "the base names them all" nor "the base names none" is true. What
    is true is that each one is supplied by exactly one layer: absent from both
    is every new client's boot failing, and present in both is two answers to
    what a client is configured as.
    """
    required = {field for field, info in EmbedSettings.model_fields.items() if info.is_required()}
    assert required, "a section with no required setting would make this vacuous"

    template = set(_declared(ROOT / "apps" / "_template" / "config.yaml"))
    base = set(_declared(ROOT / "apps" / "base" / "config.yaml"))

    assert required <= (template | base), sorted(required - template - base)
    assert not (required & template & base), sorted(required & template & base)

def test_no_committed_file_carries_a_resolved_credential():
    # The weaker but useful neighbour of the mechanical P-012 check: the
    # reference is left empty for the provisioning step rather than filled with
    # somebody's sandbox key.
    for path in COMMITTED_DEPLOYMENTS:
        for line in path.read_text().splitlines():
            if line.startswith(f"{EMBED_PREFIX}CREDENTIALS_SECRET"):
                assert line.split("=", 1)[1] == "", f"{path}: {line}"


def test_the_model_and_the_width_the_files_ship_are_the_ones_q_030_chose():
    # The width is the store's column type (ASM-031), so a file shipping
    # another one is a schema migration nobody asked for.
    base = _declared(ROOT / "apps" / "base" / "config.yaml")

    assert base["model"] == "qwen/qwen3-embedding-8b"
    assert base["dimension"] == 2048