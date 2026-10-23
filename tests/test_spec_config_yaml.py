
import pathlib
from decimal import Decimal

import pytest
import yaml

from src.config import CONFIG_FILENAME, ConfigError, load_config, resolve_references

ROOT = pathlib.Path(__file__).resolve().parent.parent
from src.prices import PriceBook
from src.secrets import REGISTERED_SCHEMES, resolver_over
from tests.conftest import FULL_DEPLOYMENT, deployment_config, without, write_config

def test_ac_169(monkeypatch):
    """AC-169: The file is the only place a setting comes from @spec:AC-169"""
    location = write_config(FULL_DEPLOYMENT)
    assert "size: 800" in location.read_text()
    monkeypatch.setenv("RAG_CHUNK_SIZE", "99")

    config = load_config(location)

    assert config.chunk.size == 800

    for name in ("RAG_EMBED_DIMENSION", "RAG_TELEMETRY_CLIENT", "RAG_SOURCE_TYPE"):
        monkeypatch.setenv(name, "0")
    assert load_config(location) == config

def _tree(apps: pathlib.Path, **documents: dict) -> None:
    for app, document in documents.items():
        (apps / app).mkdir(parents=True)
        (apps / app / CONFIG_FILENAME).write_text(yaml.safe_dump(document, sort_keys=True))


def test_ac_170(monkeypatch, tmp_path):
    """AC-170: A client's file carries only what differs from the base @spec:AC-170"""
    _tree(
        tmp_path,
        base=FULL_DEPLOYMENT,
        example={
            "source": {"folder_ids": ["the-firm-s-own-folder"]},
            "process": {"read": {"artifact_bucket": "modular-rag-artifacts-example"}},
        },
    )
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    config = load_config(app="example")

    assert config.source.folder_ids == ["the-firm-s-own-folder"]
    assert config.read.artifact_bucket == "modular-rag-artifacts-example"
    assert config.chunk.size == FULL_DEPLOYMENT["process"]["chunk"]["size"]
    assert config.embed.dimension == FULL_DEPLOYMENT["embed"]["dimension"]
    assert config.telemetry.client == FULL_DEPLOYMENT["telemetry"]["client"]
    assert config.source.credentials_secret == FULL_DEPLOYMENT["source"]["credentials_secret"]
    assert config.read.processor_id == FULL_DEPLOYMENT["process"]["read"]["processor_id"]

    thin = without(FULL_DEPLOYMENT, "process.chunk.size")
    other = tmp_path / "acme"
    other.mkdir()
    (other / CONFIG_FILENAME).write_text(
        yaml.safe_dump({"source": {"folder_ids": ["f"]}, "process": {"type": "docai"}})
    )
    (tmp_path / "base" / CONFIG_FILENAME).write_text(yaml.safe_dump(thin))

    with pytest.raises(ConfigError) as excinfo:
        load_config(app="acme")

    message = str(excinfo.value)
    assert "[process]" in message
    assert "missing required setting" in message
    assert "size" in message

def test_ac_171():
    """AC-171: A configuration that cannot be honoured names the file and the place in it @spec:AC-171"""
    document = {
        **{name: values for name, values in FULL_DEPLOYMENT.items() if name != "telemetry"},
        "process": {**FULL_DEPLOYMENT["process"], "chunk": {"size": "not-a-number"}},
        "chunck": {"size": 800},
    }

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)
    lines = message.splitlines()

    assert any("[process]" in line and "size" in line for line in lines)
    assert any("[telemetry]" in line and "client" in line for line in lines)

    assert all("deployment.yaml" in line for line in lines), message
    assert all(line.startswith("[") for line in lines), message

    assert any("'chunck'" in line for line in lines)
    assert len(lines) == 3, message

def test_ac_172():
    """AC-172: A credential value in the file stops the boot, and is not echoed @spec:AC-172"""
    pasted = "sk-or-v1:9f3c8ab27e4d1560"
    document = {**FULL_DEPLOYMENT, "embed": {**FULL_DEPLOYMENT["embed"], "credentials_secret": pasted}}

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)

    assert "[embed]" in message
    assert "credentials_secret" in message

    assert pasted not in message
    for fragment in pasted.replace(":", " ").split():
        assert fragment not in message, fragment

def test_ac_173():
    """AC-173: A reference names its backend, and an unknown backend is listed against @spec:AC-173"""
    document = {
        **FULL_DEPLOYMENT,
        "embed": {**FULL_DEPLOYMENT["embed"], "credentials_secret": "vault:secret/openrouter"},
    }

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)

    assert "[embed]" in message
    assert "credentials_secret" in message
    for scheme in REGISTERED_SCHEMES:
        assert f"{scheme}:" in message

    assert "vault" not in message

def test_ac_174():
    """AC-174: A reference resolves wherever it appears, and a literal that looks like one is written as one @spec:AC-174"""
    document = {
        **FULL_DEPLOYMENT,
        "process": {**FULL_DEPLOYMENT["process"], "read": {**FULL_DEPLOYMENT["process"]["read"], "artifact_bucket": "env:ARTIFACT_BUCKET"}},
    }

    config = resolve_references(
        deployment_config(document),
        resolver_over(env=lambda target: {"ARTIFACT_BUCKET": "modular-rag-artifacts"}.get(target, "·")),
    )

    assert config.read.artifact_bucket == "modular-rag-artifacts"

    escaped = {
        **FULL_DEPLOYMENT,
        "process": {**FULL_DEPLOYMENT["process"], "read": {**FULL_DEPLOYMENT["process"]["read"], "artifact_bucket": "env::ARTIFACT_BUCKET"}},
    }

    literal = resolve_references(
        deployment_config(escaped),
        resolver_over(
            env=lambda target: pytest.fail(f"resolved {target!r}; the escape resolves nothing")
            if target == "ARTIFACT_BUCKET"
            else "·"
        ),
    )

    assert literal.read.artifact_bucket == "env:ARTIFACT_BUCKET"
