import pytest

from src.boot import BootCompatibilityError, check_boot_compatibility
from src.config import ConfigError
from tests.conftest import FULL_DEPLOYMENT, deployment_config, without
from src.fingerprint import compute_fingerprint

def test_ac_001(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(without(FULL_DEPLOYMENT, "process.chunk.size"))

    message = str(excinfo.value)
    assert "[process]" in message
    assert "size" in message
    assert "missing required setting" in message


def test_ac_002(monkeypatch):
    document = {
        **FULL_DEPLOYMENT,
        "process": {
            **FULL_DEPLOYMENT["process"],
            "chunk": {"size": 800, "sizee": 800},
        },
    }

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)
    assert "[process]" in message
    assert "unrecognised setting" in message
    assert "sizee" in message


def test_ac_002_typo_that_damages_the_section_name(monkeypatch):
    document = {**FULL_DEPLOYMENT, "chunck": {"size": 800}}

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)
    assert "'chunck'" in message
    assert "no module claims it" in message
    assert "chunck" in message


def test_ac_003(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(FULL_DEPLOYMENT, RAG_SOURCE_TYPE="dropbox")

    message = str(excinfo.value)
    assert "[source]" in message
    assert "'dropbox'" in message
    assert "gdrive" in message

def test_ac_004(monkeypatch):
    config = deployment_config(FULL_DEPLOYMENT, RAG_CHUNK_SIZE=40000)

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config)

    message = str(excinfo.value)
    assert "40000" in message
    assert "32768" in message
    assert "reduce the chunk size" in message


def test_ac_005(monkeypatch):
    config = deployment_config(FULL_DEPLOYMENT)
    assert config.embed.dimension == 2048

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config, stored_dimension=768)

    message = str(excinfo.value)
    assert "2048" in message
    assert "768" in message
    assert "schema migration" in message


def test_ac_006(monkeypatch):
    config = deployment_config(FULL_DEPLOYMENT, RAG_EMBED_BASE_URL="https://evil.example.com/v1")

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config)

    message = str(excinfo.value)
    assert "evil.example.com" in message


def test_ac_007(monkeypatch):
    document = {
        **FULL_DEPLOYMENT,
        "embed": {
            **FULL_DEPLOYMENT["embed"],
            "request": {
                "headers": {"Authorization": "Basic smuggled", "x-api-key": "fine"},
                "body": {
                    "model": "something-else",
                    "dimensions": 1024,
                    "allow_fallbacks": True,
                    "encoding_format": "float",
                },
            },
        },
    }
    config = deployment_config(document)

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config)

    message = str(excinfo.value)
    assert "request.body.model" in message and "embed.model" in message
    assert "request.body.dimensions" in message and "ASM-031" in message
    assert "request.body.allow_fallbacks" in message and "AC-052" in message
    assert "request.headers.Authorization" in message and "embed.credentials_secret" in message

    assert "encoding_format" not in message
    assert "x-api-key" not in message


_AC_008_OPTIONAL_KEYS = (
    "RAG_DOCAI_ENABLE_TABLE_ANNOTATION",
    "RAG_DOCAI_ENABLE_IMAGE_ANNOTATION",
    "RAG_DOCAI_REGION",
    "RAG_CHUNK_INCLUDE_HEADINGS",
    "RAG_EMBED_EXTRA_PARAMS",
    "RAG_EMBED_PROVIDER",
    "RAG_EMBED_BATCH_SIZE",
    "RAG_SEARCH_RESULT_COUNT",
    "RAG_SEARCH_DENSE_WEIGHT",
    "RAG_SEARCH_LEXICAL_WEIGHT",
    "RAG_SEARCH_INDEX_SEARCH_EFFORT",
    "RAG_SEARCH_CANDIDATE_DEPTH",
    "RAG_SEARCH_MAX_RESULT_COUNT",
    "RAG_SEARCH_MAX_CANDIDATE_DEPTH",
    "RAG_SEARCH_MAX_INDEX_SEARCH_EFFORT",
    "RAG_SEARCH_BRIEF_LENGTH",
    "RAG_SEARCH_MAX_FILTER_CLAUSES",
    "RAG_SEARCH_MAX_FILTER_DEPTH",
    "RAG_SEARCH_MAX_PATTERN_LENGTH",
    "RAG_SEARCH_FILTER_TIMEOUT_MS",
    "RAG_RERANK_ENABLED",
    "RAG_RERANK_MODEL",
    "RAG_API_MAX_DOCUMENT_PIECES",
    "RAG_API_ASSISTANT_DOCUMENT_PIECES",
    "RAG_API_PASSAGE_WINDOW",
    "RAG_USAGE_RAW_RETENTION_DAYS",
    "RAG_USAGE_RECORD_QUESTION_TEXT",
    "RAG_USAGE_PRICES",
    "RAG_TELEMETRY_EXPORT_ENDPOINT",
    "RAG_SCHEDULE_FULL_SWEEP",
    "RAG_SCHEDULE_INCREMENTAL",
    "RAG_DB_URL_SECRET",
)


def _config_with(monkeypatch=None, **overrides):
    if "RAG_DB_URL_SECRET" in overrides:
        overrides["RAG_DB_URL"] = ""
    return deployment_config(FULL_DEPLOYMENT, **overrides)


def test_ac_008(monkeypatch):
    recorded = compute_fingerprint(_config_with(monkeypatch))

    write_time_changes = {
        "RAG_DOCAI_PROCESSOR_VERSION": "pretrained-layout-parser-v1.5",
        "RAG_DOCAI_ENABLE_TABLE_ANNOTATION": "false",
        "RAG_DOCAI_ENABLE_IMAGE_ANNOTATION": "false",
        "RAG_CHUNK_SIZE": "400",
        "RAG_CHUNK_INCLUDE_HEADINGS": "false",
        "RAG_EMBED_MODEL": "google/gemini-embedding-001",
        "RAG_EMBED_DIMENSION": "3072",
        "RAG_EMBED_EXTRA_PARAMS": '{"user": "example-user"}',
        "RAG_EMBED_PROMPTS": {"query": "", "document": "passage: "},
    }
    for key, value in write_time_changes.items():
        changed = compute_fingerprint(_config_with(monkeypatch, **{key: value}))
        assert changed != recorded, f"{key} did not move the fingerprint"

    monkeypatch.setattr("src.fingerprint.cut_version", lambda _module: "a-different-cut")
    assert compute_fingerprint(_config_with()) != recorded, "the cut version did not move it"
    monkeypatch.undo()

    read_time_changes = {
        "RAG_SEARCH_RESULT_COUNT": "50",
        "RAG_SEARCH_DENSE_WEIGHT": "0.9",
        "RAG_SEARCH_LEXICAL_WEIGHT": "0.1",
        "RAG_SEARCH_INDEX_SEARCH_EFFORT": "100",
        "RAG_SEARCH_CANDIDATE_DEPTH": "200",
        "RAG_RERANK_ENABLED": "true",
        "RAG_RERANK_MODEL": "cross-encoder/mmarco-mMiniLMv2",
        "RAG_SEARCH_MAX_RESULT_COUNT": "25",
        "RAG_SEARCH_MAX_CANDIDATE_DEPTH": "300",
        "RAG_SEARCH_MAX_INDEX_SEARCH_EFFORT": "500",
        "RAG_EMBED_QUERY_INSTRUCTION": "Instruct: something else\nQuery: ",
        "RAG_SEARCH_BRIEF_LENGTH": "200",
        "RAG_SEARCH_MAX_FILTER_CLAUSES": "32",
        "RAG_SEARCH_MAX_FILTER_DEPTH": "5",
        "RAG_SEARCH_MAX_PATTERN_LENGTH": "100",
        "RAG_SEARCH_FILTER_TIMEOUT_MS": "5000",
        "RAG_API_MAX_DOCUMENT_PIECES": "20",
        "RAG_API_ASSISTANT_DOCUMENT_PIECES": "5",
        "RAG_API_PASSAGE_WINDOW": "2",
        "RAG_USAGE_RAW_RETENTION_DAYS": "90",
        "RAG_USAGE_RECORD_QUESTION_TEXT": "true",
        "RAG_USAGE_PRICES": [
            {"operation": "read", "provider": "google", "unit": "page", "amount": "0.02"}
        ],
        "RAG_TELEMETRY_EXPORT_ENDPOINT": "http://collector.internal:4318",
        "RAG_DOCAI_REGION": "eu",
        "RAG_EMBED_BATCH_SIZE": "128",
        "RAG_EMBED_PROVIDER": "DeepInfra",
        "RAG_SCHEDULE_FULL_SWEEP": "0 4 * * *",
        "RAG_SCHEDULE_INCREMENTAL": "*/15 * * * *",
        "RAG_DB_URL_SECRET": "env:DATABASE_URL",
    }
    for key, value in read_time_changes.items():
        unchanged = compute_fingerprint(_config_with(monkeypatch, **{key: value}))
        assert unchanged == recorded, f"{key} unexpectedly moved the fingerprint"
