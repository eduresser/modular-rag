"""Unit-level coverage for src/config.py that is not itself an acceptance criterion.

The acceptance criteria (AC-001, AC-002, AC-003, AC-169, AC-171) are proven in
tests/test_spec_config_and_boot.py and tests/test_spec_config_yaml.py — see
CLAUDE.md for why the proofs live there. This file covers the happy path and
the per-section edge cases underneath them.

A test says "the file omits this" with ``without(document, "chunk.size")``,
because there is no longer anything to unset: the deployment is a document, and
an omission is an edit to it.
"""

import ast
import pathlib
import re
from decimal import Decimal

import pytest
from pydantic import ValidationError

from typing import get_args

from src.config import (
    CHUNK_MODULES,
    PROCESS_MODULES,
    REGISTERED_CHUNK_MODULES,
    REGISTERED_PROCESS_MODULES,
    _MODULES,
    AppConfig,
    ConfigError,
    SourceConfig,
)
from src.prices import PriceBook
from tests.conftest import FULL_DEPLOYMENT, deployment_config, without

def test_load_config_builds_appconfig_on_a_complete_file():
    config = deployment_config(FULL_DEPLOYMENT)

    assert isinstance(config, AppConfig)
    assert config.source.type == "gdrive"
    assert config.chunk.size == 800
    assert config.embed.dimension == 2048
    # Defaults apply where the file says nothing.
    assert config.read.page_limit == 200
    assert config.rerank.enabled is False


def test_load_config_reports_every_offending_section_at_once():
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(
            without(FULL_DEPLOYMENT, "process.chunk.size", "source.folder_ids")
        )

    message = str(excinfo.value)
    assert "[process]" in message
    assert "[source]" in message
    # The sub-section the setting sits in is named inside the section's own
    # report, so a missing `size` sends somebody to the right half of the
    # process rather than to the process at large (AC-171).
    assert "chunk.size" in message
    # Sections that were valid are not mentioned.
    assert "[embed]" not in message


def _every_settings_class():
    """Every concrete section model, a pluggable stage's variants included.

    A section is either one model or a discriminated union of them, and the
    union itself is not callable — so the guarantee is asserted over what a
    file actually validates against, which is the variants.
    """
    seen: list = []
    for name, declared in _MODULES:
        members = get_args(get_args(declared)[0]) if get_args(declared) else (declared,)
        for member in members:
            if not hasattr(member, "model_fields"):
                continue
            seen.append((name, member))
            # A section with sub-sections has to be walked into: `process` holds
            # a read half and a cut half, and `extra="forbid"` has to hold on
            # each of them or a typo one level down passes silently (AC-002).
            for field in member.model_fields.values():
                nested = field.annotation
                if hasattr(nested, "model_fields"):
                    seen.append((name, nested))
    return seen


@pytest.mark.parametrize(
    "name,cls", _every_settings_class(), ids=lambda v: getattr(v, "__name__", v)
)
def test_every_section_refuses_a_setting_it_does_not_define(name, cls):
    """No section is exempt, and `extra="forbid"` on the base is what makes that true."""
    with pytest.raises(ValidationError) as excinfo:
        cls(not_a_real_setting="x")

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())


def test_every_stage_registers_exactly_the_modules_it_declares():
    # The process stage has two modules, so it is a real discriminated union —
    # the shape every stage takes the day it gains a second. What has to hold
    # either way is that the registry and the union agree: a module declared and
    # not registered is one a file cannot select, and the failure is a closed
    # Literal refusing a slug that exists.
    assert REGISTERED_CHUNK_MODULES == ("docai",)
    assert REGISTERED_PROCESS_MODULES == ("docai", "pdftext")
    # The halves are reached through the process, which is the only thing a
    # file selects — so they are what `process` validates against, one level
    # down, rather than sections of their own.
    reachable = {cls for name, cls in _every_settings_class() if name == "process"}
    assert set(CHUNK_MODULES) <= reachable
    assert set(PROCESS_MODULES) <= reachable


def test_a_section_chosen_by_type_names_the_setting_when_it_is_missing():
    # Pydantic reports a missing discriminator against the section rather than
    # the field, and "setting ''" would send somebody looking for a setting
    # with no name (AC-171).
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(without(FULL_DEPLOYMENT, "process.type"))

    message = str(excinfo.value)
    assert "[process]" in message
    assert "missing required setting 'type'" in message
    assert "'docai'" in message


def test_source_settings_rejects_an_unregistered_module_directly():
    with pytest.raises(ValidationError) as excinfo:
        SourceConfig(
            type="sharepoint",
            folder_ids=["folder-123"],
            credentials_secret="env:GOOGLE_SERVICE_ACCOUNT_KEY",
        )

    errors = excinfo.value.errors()
    assert any(error["type"] == "literal_error" for error in errors)


def test_no_variable_in_the_environment_reaches_the_configuration(monkeypatch):
    """Regression, and now a structural one.

    Build systems (BuildKit, dpkg, Nix) set SOURCE_DATE_EPOCH on their own, and
    with bare namespace prefixes the boot was refused with "[source]
    unrecognised setting 'date_epoch'" — a variable this service does not own,
    blamed on a setting that does not exist. The `RAG_` prefix fixed that by
    narrowing the scan; reading no environment at all removes the scan, so the
    variables below cannot be read as anything, and neither can one named
    exactly after a setting.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    monkeypatch.setenv("PARSE_MODE", "strict")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "99")

    config = deployment_config(FULL_DEPLOYMENT)

    assert config.source.type == "gdrive"
    assert config.chunk.size == 800


def test_a_price_reaches_the_price_book_as_an_exact_decimal():
    """A price read through a binary float rounds where money must not."""
    config = deployment_config(
        {**FULL_DEPLOYMENT,
         "usage": {"prices": [{"operation": "embed", "provider": "openai",
                              "model": "text-embedding-3-large", "unit": "token",
                              "amount": "0.000000123"}]}}
    )

    (entry,) = config.usage.prices
    assert entry.amount == Decimal("0.000000123")
    book = PriceBook(config.usage.prices)
    assert book.price_for(
        operation="embed", provider="openai", model="text-embedding-3-large", unit="token"
    ).amount == Decimal("0.000000123")


def test_the_usage_and_telemetry_sections_have_working_defaults():
    """Only the client is required; a deployment that says nothing else still boots."""
    config = deployment_config(FULL_DEPLOYMENT)

    assert config.usage.raw_retention_days == 30
    assert config.usage.record_question_text is False
    assert config.usage.prices == []
    assert config.telemetry.client == "example"
    # Empty endpoint is what leaves the trace exporter inert (ASM-072).
    assert config.telemetry.export_endpoint == ""


def test_the_brief_limit_leaves_room_for_text_and_its_cut_mark():
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(FULL_DEPLOYMENT, RAG_SEARCH_BRIEF_LENGTH=4)

    message = str(excinfo.value)
    assert "[api]" in message
    assert "brief_length" in message


def test_a_deployment_that_names_no_client_is_stopped_at_boot():
    """AC-168 depends on it, and a default would hide the gap until a second deployment."""
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(without(FULL_DEPLOYMENT, "telemetry"))

    message = str(excinfo.value)
    assert "[telemetry]" in message
    assert "client" in message


def test_a_typo_in_a_usage_setting_stops_the_boot_like_any_other():
    with pytest.raises(ConfigError) as excinfo:
        deployment_config({**FULL_DEPLOYMENT, "usage": {"retention_days": 90}})  # missing raw_

    assert "[usage]" in str(excinfo.value)
    assert "retention_days" in str(excinfo.value)


def test_no_variant_redeclares_a_field_of_its_stage():
    """The one thing a `params:` nesting would have bought, as a test instead.

    A stage's base says what every module of it has; a variant that redeclared
    one would be answering the same question twice, and pydantic allows it in
    silence. This is that guarantee — and it is why the settings sit flat
    beside `type` rather than inside a nested bag: the shape pydantic can
    discriminate on is the flat one, and the nested one needs the tag written
    in two places that nothing checks agree.
    """
    from src.config import (
        CHUNK_MODULES,
        READ_MODULES,
        SOURCE_MODULES,
        ChunkSettings,
        ReadSettings,
        SourceSettings,
    )

    stages = (
        (SourceSettings, SOURCE_MODULES),
        (ReadSettings, READ_MODULES),
        (ChunkSettings, CHUNK_MODULES),
    )
    for base, variants in stages:
        assert variants, f"{base.__name__} has no registered module"
        for variant in variants:
            # `__annotations__` is what this class writes, not what it inherits.
            redeclared = set(variant.__annotations__) & set(base.model_fields)
            assert not redeclared, f"{variant.__name__} redeclares {sorted(redeclared)}"


# ── schedule: two settings, both optional, validated at boot (AC-178) ────────


def test_a_deployment_naming_no_schedule_runs_nothing_on_its_own():
    """The third clause of AC-178: empty is the ordinary state, not a gap."""
    config = deployment_config(FULL_DEPLOYMENT)

    assert config.schedule.full_sweep == ""
    assert config.schedule.incremental == ""


def test_a_valid_five_field_schedule_is_accepted():
    config = deployment_config(FULL_DEPLOYMENT, RAG_SCHEDULE_FULL_SWEEP="0 3 * * *")

    assert config.schedule.full_sweep == "0 3 * * *"


def test_both_schedules_are_set_independently():
    config = deployment_config(
        FULL_DEPLOYMENT,
        RAG_SCHEDULE_FULL_SWEEP="0 3 * * *",
        RAG_SCHEDULE_INCREMENTAL="*/15 * * * *",
    )

    assert config.schedule.full_sweep == "0 3 * * *"
    assert config.schedule.incremental == "*/15 * * * *"


def test_an_unreadable_schedule_stops_the_boot_naming_the_setting_and_the_value():
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(FULL_DEPLOYMENT, RAG_SCHEDULE_FULL_SWEEP="not a schedule")

    message = str(excinfo.value)
    assert "[schedule]" in message
    assert "full_sweep" in message
    assert "not a schedule" in message


def test_a_schedule_outside_its_valid_range_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(FULL_DEPLOYMENT, RAG_SCHEDULE_INCREMENTAL="60 * * * *")

    message = str(excinfo.value)
    assert "[schedule]" in message
    assert "incremental" in message
    assert "60 * * * *" in message


def test_a_schedule_with_the_wrong_field_count_is_rejected():
    """ASM-091 is a FIVE-field expression; croniter itself accepts 6, 7 or a macro."""
    with pytest.raises(ConfigError) as excinfo:
        deployment_config(FULL_DEPLOYMENT, RAG_SCHEDULE_FULL_SWEEP="0 0 3 * * *")

    message = str(excinfo.value)
    assert "[schedule]" in message
    assert "five-field" in message


@pytest.mark.parametrize("reference", ["", "plain-password", "vault:db"])
def test_database_secret_field_only_accepts_registered_references(reference):
    with pytest.raises(ConfigError) as error:
        deployment_config({**FULL_DEPLOYMENT, "db": {"url_secret": reference}})
    assert "url_secret" in str(error.value)
    if reference:
        assert reference not in str(error.value)


def test_database_secret_resolution_is_an_explicit_boot_step(monkeypatch):
    from src.config import resolve_database_config

    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres@localhost/ignored")
    config = deployment_config({**FULL_DEPLOYMENT, "db": {"url_secret": "env:DATABASE_URL"}})
    assert config.db.url == ""
    seen = []

    def resolver(reference):
        seen.append(reference)
        return "postgresql://postgres@localhost/modular_rag"

    resolved = resolve_database_config(config, resolver)
    assert resolved.db.url == "postgresql://postgres@localhost/modular_rag"
    assert resolve_database_config(resolved, resolver) is resolved
    assert seen == ["env:DATABASE_URL"]
    assert config.db.url_secret == "env:DATABASE_URL"
    assert "postgresql://" not in repr(resolved.db)


# --- The names a message gives a setting -----------------------------------
#
# A message that names the wrong setting is a defect no type checker can see:
# the string is well formed, the code runs, and the person reading it goes
# looking for something that does not exist. It happened wholesale — AC-169
# took the environment out of the configuration and the `RAG_` prefix with it,
# and fourteen messages across eight modules kept sending operators to
# variables this build never reads.

#: The one place the retired prefix may still be written: the boot check whose
#: whole job is to refuse a deployment that still carries it.
RETIRED_PREFIX_EXCEPTIONS = {("src/config.py", "RAG_CONFIG")}

_SOURCE_FILES = sorted(pathlib.Path("src").rglob("*.py"))


def _string_literals(path: pathlib.Path) -> list[str]:
    """Every string constant in a module, docstrings included."""
    tree = ast.parse(path.read_text())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_no_message_sends_an_operator_to_a_variable_this_build_never_reads():
    """AC-169: the environment carries secret values and nothing else."""
    offenders: list[tuple[str, str]] = []
    for path in _SOURCE_FILES:
        for literal in _string_literals(path):
            for name in re.findall(r"RAG_[A-Z0-9_]+", literal):
                if (path.as_posix(), name) not in RETIRED_PREFIX_EXCEPTIONS:
                    offenders.append((path.as_posix(), name))

    assert offenders == [], (
        "these name an environment variable that went with AC-169; a setting is named "
        f"by its path in the deployment's file, as 'embed.dimension' is: {offenders}"
    )


def _resolves(path: str) -> bool:
    """Whether a dotted name reaches a real field of :class:`AppConfig`.

    A union chosen by ``type`` is resolved through every variant, because a
    message naming ``process.read`` is right whichever module is configured.
    """
    models = [AppConfig]
    for segment in path.split("."):
        nxt = []
        for model in models:
            field = getattr(model, "model_fields", {}).get(segment)
            annotation = field.annotation if field is not None else None
            if annotation is None and isinstance(getattr(model, segment, None), property):
                # `read` and `chunk` are properties over `process` (ASM-125).
                annotation = getattr(AppConfig, "_property_targets", {}).get(segment)
                if annotation is None:
                    return True
            if annotation is None:
                continue
            nxt += [
                candidate
                for candidate in (get_args(annotation) or (annotation,))
                if hasattr(candidate, "model_fields")
            ] or [type(None)]
        if not nxt:
            return False
        models = nxt
    return True


def test_every_setting_a_message_names_is_a_setting_the_file_can_carry():
    """The other half of the same drift, and the half that survives a rename.

    A dotted name whose first segment is a section of the deployment's file is
    read as a setting and checked; anything else — a module path, a table and
    its column — is not a setting and is left alone.
    """
    # A fingerprint key is a HASH INPUT and not a field path: it keeps the
    # spelling it was first hashed under, deliberately, through every rename of
    # the setting behind it (`docai.processor_version`, `parse.version`).
    # Renaming one would move a stage digest and reprocess a corpus to arrive at
    # identical output, which is the one thing `src/fingerprint.py` exists to
    # avoid. `tests/test_fingerprint.py` is what keeps these honest.
    hash_inputs = {("src/fingerprint.py", "embed.extra_params")}
    sections = set(AppConfig.model_fields) | {"read", "chunk"}
    # A file name and a hostname are dotted too, and their first label collides
    # with a section often enough to matter: `search.py`, `api.openai.com`.
    not_settings = {"py", "yaml", "yml", "json", "md", "sql", "com", "ai", "org", "net", "io"}
    offenders: list[tuple[str, str]] = []
    for path in _SOURCE_FILES:
        for literal in _string_literals(path):
            for name in re.findall(r"\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b", literal):
                if name.split(".")[-1] in not_settings:
                    continue
                if (path.as_posix(), name) in hash_inputs:
                    continue
                if name.split(".")[0] in sections and not _resolves(name):
                    offenders.append((path.as_posix(), name))

    assert offenders == [], f"named as settings and not carried by any section: {offenders}"
