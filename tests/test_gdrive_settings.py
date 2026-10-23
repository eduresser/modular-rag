"""Unit-level coverage for the source module's environment contract.

AC-010 is proven in tests/test_spec_source_gdrive.py — see CLAUDE.md for why
the proof lives there. This file covers the naming rule the contract rests on
and the committed example file that documents it: a setting in the wrong half
is a setting that has to be renamed the day a second source arrives.

The two halves used to be two sections — `source` beside `gdrive`. They are a
base class and its variant now, under the one section named after the stage,
because a section named after the vendor wrote the same decision twice with
nothing checking that the two agreed, and left a deployment on another source
still obliged to fill in this one's required fields.
"""

import pathlib

import yaml

import pytest
from pydantic import ValidationError

from src.config import (
    _MODULES,
    REGISTERED_SOURCE_MODULES,
    SOURCE_MODULES,
    GDriveSourceSettings,
    SourceConfig,
    SourceSettings,
)
from src.secrets import SECRET_SUFFIX
from src.sources.base import registered_slugs

MODULE_EXAMPLE = pathlib.Path(__file__).resolve().parent.parent / "src" / "sources" / "config.example.yaml"

CREDENTIALS = "env:GOOGLE_SERVICE_ACCOUNT_KEY"


def _declared(section: str) -> set[str]:
    """The settings the module's own example documents for one section."""
    document = yaml.safe_load(MODULE_EXAMPLE.read_text())
    return set(document.get(section) or {})


def test_what_every_source_has_lives_on_the_shared_base():
    # The naming rule: a setting any source has goes on the base, so a second
    # source inherits it rather than restating it. Asserted over the fields the
    # base owns — which is the split itself, rather than a prefix string that
    # only ever described it.
    assert set(SourceSettings.model_fields) == set()

    # And the base is not a thing a file can select: every concrete source
    # declares its own slug, so there is no sourceless source.
    assert "type" not in SourceSettings.model_fields


def test_what_only_this_source_has_lives_on_its_own_variant():
    own = set(GDriveSourceSettings.model_fields) - set(SourceSettings.model_fields)

    assert own == {"type", "folder_ids", "credentials_secret", "units_per_minute"}


def test_the_section_is_named_after_the_stage_not_after_the_vendor():
    # One section, named for what every deployment has. With one registered
    # source the alias is that source; a second makes it a discriminated union
    # on `type`, and nothing about the shape of the file moves.
    assert dict(_MODULES)["source"] is SourceConfig
    assert SOURCE_MODULES == (GDriveSourceSettings,)
    assert "gdrive" not in dict(_MODULES)


def test_the_credential_is_a_reference_and_its_name_says_so():
    # app/secrets.py resolves a setting by its suffix, so the suffix is not
    # cosmetic — it is what makes the value a reference rather than a secret in
    # a committed file (principle P-012).
    assert "credentials_secret".lower().endswith(SECRET_SUFFIX)


def test_the_module_example_declares_every_setting_the_models_read():
    assert set(GDriveSourceSettings.model_fields) <= _declared("source")


def test_the_module_example_declares_nothing_the_models_do_not_read():
    # A setting in the example that no model reads is a setting the boot would
    # reject as unrecognised (AC-002), which is a worse first experience than
    # a missing one.
    assert _declared("source") <= set(GDriveSourceSettings.model_fields)


def test_the_example_carries_no_credential_value():
    text = MODULE_EXAMPLE.read_text()

    assert yaml.safe_load(text)["source"]["credentials_secret"] is None
    assert "PRIVATE KEY" not in text
    assert "-----BEGIN" not in text


def test_the_module_selection_is_a_closed_literal():
    # Principle P-011: never a symbol resolved from configuration.
    valid = GDriveSourceSettings(
        type="gdrive", folder_ids=["folder-123"], credentials_secret=CREDENTIALS
    )
    assert valid.type == "gdrive"

    with pytest.raises(ValidationError):
        GDriveSourceSettings(type="dropbox", folder_ids=["f"], credentials_secret=CREDENTIALS)


def test_the_configuration_literal_and_the_registry_are_one_closed_set():
    # Two views of the same thing, in two files. This test is what keeps them
    # from drifting apart — a slug the configuration accepts and the registry
    # cannot serve is a boot that passes and a run that fails.
    assert registered_slugs() == tuple(sorted(REGISTERED_SOURCE_MODULES))


def test_the_defaults_are_the_measured_figures():
    settings = GDriveSourceSettings(
        type="gdrive", folder_ids=["folder-123"], credentials_secret=CREDENTIALS
    )

    # 325,000 units a minute, measured 2026-09-08, over 200 per download.
    assert settings.units_per_minute == 325_000


def test_the_folder_and_the_credential_are_both_required():
    # A deployment missing either cannot read anything, and finding that out at
    # boot is the whole point of AC-001.
    with pytest.raises(ValidationError):
        GDriveSourceSettings(type="gdrive", credentials_secret=CREDENTIALS)
    with pytest.raises(ValidationError):
        GDriveSourceSettings(type="gdrive", folder_ids=["folder-123"])


def test_an_unknown_setting_on_either_half_is_refused():
    with pytest.raises(ValidationError):
        GDriveSourceSettings(
            type="gdrive", folder_ids=["f"], credentials_secret=CREDENTIALS, ceiling=0.5
        )
    with pytest.raises(ValidationError):
        GDriveSourceSettings(
            type="gdrive", folder_ids=["f"], credentials_secret=CREDENTIALS, folder="other"
        )


def test_a_deployment_naming_the_retired_ceiling_fails_boot():
    """A document is removed only on positive proof of deletion — never on a
    ratio a deployment could still tune. `removal_ceiling` is not a field
    on either half of the source settings any more, so a deployment that
    still names it is refused at boot with the unknown-key message, the
    same way any other unrecognised setting is.
    """
    with pytest.raises(ValidationError):
        GDriveSourceSettings(
            type="gdrive",
            folder_ids=["folder-123"],
            credentials_secret=CREDENTIALS,
            removal_ceiling=0.10,
        )


def test_the_watched_corpus_is_always_a_list_and_never_empty():
    """One setting, always plural — and a list of none watches nothing.

    A deployment with one folder writes a list of one. The empty list is the
    shape that would otherwise pass validation and then sweep nothing at all,
    reporting a corpus that finished with zero documents rather than a
    configuration that names none.
    """
    one = GDriveSourceSettings(
        type="gdrive", folder_ids=["folder-123"], credentials_secret=CREDENTIALS
    )
    assert one.folder_ids == ["folder-123"]

    several = GDriveSourceSettings(
        type="gdrive", folder_ids=["folder-123", "folder-456"], credentials_secret=CREDENTIALS
    )
    assert several.folder_ids == ["folder-123", "folder-456"]

    with pytest.raises(ValidationError):
        GDriveSourceSettings(type="gdrive", folder_ids=[], credentials_secret=CREDENTIALS)


def test_an_entry_that_names_no_folder_is_refused_with_its_own_message():
    # A list item left without a value parses as a null and would otherwise
    # reach the source as a folder identifier, where the refusal is the
    # provider's and unreadable.
    with pytest.raises(ValidationError) as excinfo:
        GDriveSourceSettings(
            type="gdrive", folder_ids=["folder-123", "   "], credentials_secret=CREDENTIALS
        )

    assert "empty" in str(excinfo.value)

    # And surrounding whitespace is not part of an identifier.
    settings = GDriveSourceSettings(
        type="gdrive", folder_ids=[" folder-123 "], credentials_secret=CREDENTIALS
    )
    assert settings.folder_ids == ["folder-123"]


def test_a_deployment_naming_the_retired_singular_folder_fails_boot():
    """`folder_id` was renamed rather than widened (Q-119).

    Two ways of naming the watched corpus — one folder here, a list there —
    is the defect the stage-named sections exist to avoid, so the old key is
    refused as an unrecognised setting (AC-002) instead of being quietly
    accepted as a list of one.
    """
    with pytest.raises(ValidationError):
        GDriveSourceSettings(
            type="gdrive", folder_id="folder-123", credentials_secret=CREDENTIALS
        )
