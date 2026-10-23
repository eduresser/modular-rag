import pathlib

import pytest
import yaml

from src.config import ConfigError, DocaiReadSettings, PdfTextReadSettings
from src.processing.docai import MAX_CONCURRENT_BATCHES, MAX_DOCUMENTS_PER_BATCH
from src.processing.preflight import BATCH_BYTE_LIMIT
from tests.conftest import FULL_DEPLOYMENT, deployment_config

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE_EXAMPLE = ROOT / "src" / "processing" / "config.example.yaml"
COMMITTED_DEPLOYMENTS = (
    ROOT / "apps" / "base" / "config.yaml",
    ROOT / "apps" / "example" / "config.yaml",
    ROOT / "apps" / "_template" / "config.yaml",
)

_LEGACY_PREFIX = "RAG_"
DOCAI_PREFIX = f"{_LEGACY_PREFIX}DOCAI_"


def _declared(path: pathlib.Path, section: str = "process.read") -> set[str]:
    node = yaml.safe_load(path.read_text()) or {}
    for step in section.split("."):
        node = (node or {}).get(step) or {}
    return dict(node)


def _expected() -> set[str]:
    return set(DocaiReadSettings.model_fields)


def _env(monkeypatch, **overrides):
    return deployment_config(FULL_DEPLOYMENT, **overrides)


def test_a_deployment_must_name_the_processor_and_the_version_it_pins(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_DOCAI_PROCESSOR_VERSION=None)

    message = str(excinfo.value)
    assert "[process]" in message
    assert "processor_version" in message
    assert "missing required setting" in message


def test_a_deployment_without_object_storage_is_refused_at_boot(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_DOCAI_ARTIFACT_BUCKET=None)

    assert "artifact_bucket" in str(excinfo.value)


def test_the_region_is_a_closed_pair_and_a_third_value_is_named(monkeypatch):
    assert _env(monkeypatch, RAG_DOCAI_REGION="eu").process.region == "eu"

    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_DOCAI_REGION="southamerica-east1")

    message = str(excinfo.value)
    assert "'southamerica-east1'" in message
    assert "accepted values are" in message
    assert "'us'" in message and "'eu'" in message


def test_the_module_selection_message_still_says_registered_modules(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_SOURCE_TYPE="dropbox")

    assert "registered modules are" in str(excinfo.value)


def test_the_reader_selection_is_a_closed_literal_too(monkeypatch):
    assert _env(monkeypatch).process.type == "docai"

    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_READ_TYPE="docling")

    message = str(excinfo.value)
    assert "[process]" in message
    assert "'docling'" in message
    assert "registered modules are" in message


def test_the_kept_output_is_the_stages_requirement_and_the_bucket_is_one_modules(monkeypatch):
    from src.config import DocaiReadSettings, PdfTextReadSettings, ReadSettings
    from src.processing.artifact import ArchiveStore, ArtifactStore, ObjectStore

    assert "artifact_bucket" not in ReadSettings.model_fields
    assert "type" not in ReadSettings.model_fields

    assert "artifact_bucket" in DocaiReadSettings.model_fields
    assert not PdfTextReadSettings.model_fields

    for store in (ObjectStore, ArchiveStore):
        assert issubclass(store, ArtifactStore), store.__name__


def test_the_description_switches_are_on_unless_a_deployment_turns_them_off(monkeypatch):
    config = _env(monkeypatch)

    assert config.read.enable_table_annotation is True
    assert config.read.enable_image_annotation is True
    assert _env(monkeypatch, RAG_DOCAI_ENABLE_TABLE_ANNOTATION="false").read.enable_table_annotation is False


def test_no_setting_opens_a_second_read_route(monkeypatch):
    from src.config import DocaiReadSettings

    assert not [name for name in DocaiReadSettings.model_fields if name.startswith("allow_")]


def test_the_limits_default_to_the_providers_own(monkeypatch):
    config = _env(monkeypatch)

    assert config.read.max_concurrent_batches == MAX_CONCURRENT_BATCHES
    assert config.read.max_documents_per_batch == MAX_DOCUMENTS_PER_BATCH
    assert config.read.max_document_bytes == BATCH_BYTE_LIMIT
    assert config.read.page_limit == 200


def test_a_typo_in_a_processing_setting_stops_the_boot(monkeypatch):
    with pytest.raises(ConfigError) as excinfo:
        _env(monkeypatch, RAG_DOCAI_PROCESSOR_VERSON="v1")

    message = str(excinfo.value)
    assert "[process]" in message
    assert "processor_verson" in message


def test_the_module_env_example_declares_every_setting_and_no_extra():
    declared = set(_declared(MODULE_EXAMPLE))

    assert declared == _expected()


def test_the_module_example_documents_the_second_module_too():
    text = MODULE_EXAMPLE.read_text(encoding="utf-8")

    assert "type: pdftext" in text
    free = {
        field for field, info in PdfTextReadSettings.model_fields.items()
    } | {"type", "credentials_secret"}
    for field in free:
        assert field in text, field
    free_block = text[text.index("type: pdftext"):]
    for borrowed in ("processor_id", "processor_version", "region", "size", "include_headings"):
        assert borrowed not in free_block, borrowed


def test_every_required_setting_is_supplied_by_exactly_one_layer():
    required = {field for field, info in DocaiReadSettings.model_fields.items() if info.is_required()}
    assert required, "a section with no required setting would make this vacuous"

    template = set(_declared(ROOT / "apps" / "_template" / "config.yaml"))
    base = set(_declared(ROOT / "apps" / "base" / "config.yaml"))

    assert required <= (template | base), sorted(required - template - base)
    assert not (required & template & base), sorted(required & template & base)

def test_no_committed_file_carries_a_resolved_value_for_the_bucket_or_the_processor():
    template = _declared(ROOT / "apps" / "_template" / "config.yaml")

    assert template["processor_id"] is None
    assert template["artifact_bucket"] is None


def test_the_pinned_version_the_files_ship_is_the_one_with_bounding_boxes():
    for path in COMMITTED_DEPLOYMENTS:
        pinned = _declared(path).get("processor_version")
        if pinned is not None:
            assert pinned == "pretrained-layout-parser-v1.0-2024-06-03", path


def test_the_process_names_its_own_credential_rather_than_borrowing_the_sources():
    from src.config import DocaiProcessSettings, ProcessSettings

    assert DocaiProcessSettings.model_fields["credentials_secret"].is_required()
    assert "credentials_secret" not in ProcessSettings.model_fields


def test_neither_half_carries_a_credential_because_there_is_one_call():
    from src.config import ChunkSettings, DocaiChunkSettings, DocaiReadSettings, ReadSettings

    for model in (DocaiReadSettings, ReadSettings, DocaiChunkSettings, ChunkSettings):
        assert "credentials_secret" not in model.model_fields, model.__name__


def test_the_committed_deployment_states_the_reference_in_every_stage_that_spends_it():
    example = yaml.safe_load((ROOT / "apps" / "example" / "config.yaml").read_text())
    spent = {
        section: example[section].get("credentials_secret")
        for section in ("source", "process")
        if example[section].get("credentials_secret") is not None
    }

    assert "source" in spent, "the source always reaches somebody"
    assert set(spent.values()) == {"env:GOOGLE_SERVICE_ACCOUNT_KEY"}, spent

    from src.config import PROCESS_MODULES

    declares = {
        module.model_fields["type"].annotation.__args__[0]: "credentials_secret" in module.model_fields
        for module in PROCESS_MODULES
    }
    assert declares == {"docai": True, "pdftext": False}
    assert declares[example["process"]["type"]] is ("process" in spent)


def test_the_base_layer_names_no_credential_for_the_two_stages_either():
    base = yaml.safe_load((ROOT / "apps" / "base" / "config.yaml").read_text())

    for section in ("source", "process", "embed"):
        if section not in base:
            continue
        assert "credentials_secret" not in base[section], section
