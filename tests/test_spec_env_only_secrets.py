
import pathlib

import pytest

from src.config import ConfigError
from src.secrets import (
    REGISTERED_SCHEMES,
    SCHEME_ENV,
    SecretResolutionError,
    resolve_secrets,
    resolver_over,
)
from tests.conftest import FULL_DEPLOYMENT, deployment_config

SECRETS_MODULE = pathlib.Path(__file__).resolve().parent.parent / "src" / "secrets.py"


def test_ac_242(monkeypatch):
    """AC-242: Every credential resolves from the environment, and an absent one is named @spec:AC-242"""
    references = {
        "RAG_SOURCE_CREDENTIALS_SECRET": "env:GOOGLE_SERVICE_ACCOUNT_KEY",
        "RAG_EMBED_CREDENTIALS_SECRET": "env:OPENROUTER_KEY",
        "RAG_API_CREDENTIALS_SECRET": "env:API_TOKEN",
        "RAG_DB_URL_SECRET": "env:DATABASE_URL",
    }
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_KEY", '{"client_email": "a@b.iam"}')
    monkeypatch.setenv("OPENROUTER_KEY", "sk-or-the-key")
    monkeypatch.setenv("API_TOKEN", "the-api-token")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/rag")

    resolved = resolve_secrets(references, resolver_over())

    assert resolved["RAG_EMBED_CREDENTIALS_SECRET"] == "sk-or-the-key"
    assert resolved["RAG_API_CREDENTIALS_SECRET"] == "the-api-token"
    assert resolved["RAG_DB_URL_SECRET"] == "postgresql://localhost/rag"

    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    with pytest.raises(SecretResolutionError) as unset:
        resolve_secrets(references, resolver_over())

    message = str(unset.value)
    assert "RAG_EMBED_CREDENTIALS_SECRET" in message
    assert "env:OPENROUTER_KEY" in message
    assert "the-api-token" not in message, "a value resolved earlier in the same run leaked"
    assert "the-key" not in message

    monkeypatch.setenv("OPENROUTER_KEY", "")
    with pytest.raises(SecretResolutionError) as blank:
        resolve_secrets(references, resolver_over())
    assert "RAG_EMBED_CREDENTIALS_SECRET" in str(blank.value)


def test_ac_243():
    """AC-243: A reference to a secret store is refused, and there is nothing behind it to reach @spec:AC-243"""
    for retired in ("gcp-sm:projects/x/secrets/y/versions/latest", "vault:secret/openrouter"):
        document = {
            **FULL_DEPLOYMENT,
            "embed": {**FULL_DEPLOYMENT["embed"], "credentials_secret": retired},
        }

        with pytest.raises(ConfigError) as excinfo:
            deployment_config(document)

        message = str(excinfo.value)
        assert "[embed]" in message
        assert "credentials_secret" in message
        assert f"{SCHEME_ENV}:" in message
        assert REGISTERED_SCHEMES == (SCHEME_ENV,)
        assert "vault" not in message
        assert "gcp-sm" not in message

    import src.secrets as secrets

    for name in ("secret_manager_backend", "SECRET_MANAGER_API_ROOT", "SCHEME_SECRET_MANAGER"):
        assert not hasattr(secrets, name), f"{name} is still reachable"

    source = SECRETS_MODULE.read_text()
    body = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for reachable in ("secretmanager.googleapis.com", "google.auth", "base64", ":access"):
        assert reachable not in body, f"{reachable} still has a path through this module"
