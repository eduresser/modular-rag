"""Where a deployment's configuration is found, and what a bad file says.

AC-169 and AC-171 are proven in tests/test_spec_config_yaml.py — see CLAUDE.md
for why the proofs live there. This file covers what the criteria do not name
and what silently breaks anyway: how a path is derived from a deployment's
name, which names are refused, and what a file the loader cannot read reports.
"""

from __future__ import annotations

import pathlib
from decimal import Decimal

import pytest
import yaml

from src.config import (
    APP_VARIABLE,
    APPS_DIRECTORY,
    CONFIG_FILENAME,
    RETIRED_PATH_VARIABLE,
    ConfigError,
    config_path_for,
    load_config,
)
from tests.conftest import FULL_DEPLOYMENT, write_config


def test_a_deployment_s_path_is_derived_from_its_name():
    assert config_path_for("example") == APPS_DIRECTORY / "example" / CONFIG_FILENAME


@pytest.mark.parametrize("name", ["", "   ", "../etc/passwd", "example/../base", ".hidden"])
def test_a_name_that_carries_a_path_is_refused(name):
    # The whole point of naming rather than locating is that there is no way to
    # point at another file; a name that traverses would put it back.
    with pytest.raises(ConfigError) as excinfo:
        config_path_for(name)

    assert "not a deployment name" in str(excinfo.value)


def test_naming_no_deployment_names_the_variable_and_lists_the_ones_there_are(monkeypatch):
    monkeypatch.delenv(APP_VARIABLE, raising=False)
    monkeypatch.delenv(RETIRED_PATH_VARIABLE, raising=False)

    with pytest.raises(ConfigError) as excinfo:
        load_config()

    message = str(excinfo.value)
    assert f"${APP_VARIABLE}" in message
    # The committed deployments, so the operator does not have to guess. The
    # template is excluded: its required fields are blank on purpose.
    assert "example" in message
    assert "_template" not in message


def test_the_retired_path_variable_is_refused_rather_than_ignored(monkeypatch):
    # It was documented in one commit before being replaced, so a shell may
    # still export it — and a stale path beating the app that was asked for is
    # the failure this exists to prevent.
    monkeypatch.setenv(RETIRED_PATH_VARIABLE, "/tmp/somewhere.yaml")

    with pytest.raises(ConfigError) as excinfo:
        load_config(app="example")

    message = str(excinfo.value)
    assert RETIRED_PATH_VARIABLE in message
    assert APP_VARIABLE in message


def test_an_explicit_path_bypasses_the_convention(monkeypatch):
    # How a test boots over a document it wrote: outside apps/, and named.
    monkeypatch.delenv(APP_VARIABLE, raising=False)
    location = write_config(FULL_DEPLOYMENT)

    assert load_config(location).chunk.size == 800


def test_the_environment_names_the_deployment_when_nothing_else_does(monkeypatch, tmp_path):
    # The deployment's own route: $APP and nothing else. Pointed at a temporary
    # tree so the test does not depend on which clients happen to be committed.
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)
    written = write_config(FULL_DEPLOYMENT)
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / CONFIG_FILENAME).write_text(written.read_text())
    monkeypatch.setenv(APP_VARIABLE, "acme")

    assert load_config().chunk.size == 800


def test_a_file_that_is_not_there_says_so_by_name(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)
    monkeypatch.setenv(APP_VARIABLE, "acme")

    with pytest.raises(ConfigError) as excinfo:
        load_config()

    assert "could not be read" in str(excinfo.value)
    assert "acme" in str(excinfo.value)


def test_a_syntax_error_carries_the_line_and_the_column(tmp_path):
    # "The file is invalid" without a position sends somebody reading four
    # hundred lines by eye.
    broken = tmp_path / CONFIG_FILENAME
    broken.write_text("chunk:\n  size: 800\n   bad_indent: 1\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(broken)

    message = str(excinfo.value)
    assert "not valid YAML" in message
    assert "line 3" in message


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    listed = tmp_path / CONFIG_FILENAME
    listed.write_text("- chunk\n- embed\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(listed)

    assert "has to be a mapping" in str(excinfo.value)


def test_an_empty_file_is_refused_rather_than_read_as_every_default(tmp_path):
    # Every required setting missing would report twelve problems; saying the
    # file is empty is the one an operator can act on.
    empty = tmp_path / CONFIG_FILENAME
    empty.write_text("# nothing here yet\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(empty)

    assert "is empty" in str(excinfo.value)


def test_a_section_that_is_not_a_mapping_names_what_it_carries(tmp_path):
    document = write_config(FULL_DEPLOYMENT).read_text()
    wrong = tmp_path / CONFIG_FILENAME
    wrong.write_text(document + "\nrerank: enabled\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(wrong)

    message = str(excinfo.value)
    assert "[rerank]" in message
    assert "has to be a mapping" in message
    assert "str" in message


def test_a_section_the_file_omits_falls_back_to_its_defaults(tmp_path):
    # `search:` absent and `search: {}` have to mean the same thing, or a
    # deployment would have to write an empty section to get the defaults.
    assert "search" not in FULL_DEPLOYMENT

    config = load_config(write_config(FULL_DEPLOYMENT))

    assert config.api.defaults.search.limit == 10
    assert config.rerank.enabled is False


# ── The two layers, and how deep they merge ───────────────────────────────


def _apps(tmp_path: pathlib.Path, **documents: dict) -> pathlib.Path:
    for app, document in documents.items():
        (tmp_path / app).mkdir(parents=True)
        (tmp_path / app / CONFIG_FILENAME).write_text(yaml.safe_dump(document, sort_keys=True))
    return tmp_path


def test_a_section_the_client_does_not_mention_is_the_base_s_entirely(monkeypatch, tmp_path):
    _apps(tmp_path, base=FULL_DEPLOYMENT, example={"process": {"chunk": {"size": 1200}}})
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    config = load_config(app="example")

    assert config.chunk.size == 1200
    assert config.read.processor_id == FULL_DEPLOYMENT["process"]["read"]["processor_id"]


def test_a_setting_whose_value_is_a_mapping_is_replaced_and_not_merged(monkeypatch, tmp_path):
    # Merging inside a setting would leave a client no way to remove an entry
    # the base declared, and "the client's prices are these" is a sentence
    # somebody can check.
    _apps(
        tmp_path,
        base={**FULL_DEPLOYMENT, "usage": {"prices": [
            {"operation": "read", "provider": "google", "unit": "page", "amount": "0.01"},
            {"operation": "chunk", "provider": "internal", "model": "local",
             "unit": "page", "amount": "0"},
        ]}},
        example={"usage": {"prices": [
            {"operation": "read", "provider": "google", "unit": "page", "amount": "0.02"},
        ]}},
    )
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    prices = load_config(app="example").usage.prices

    assert len(prices) == 1
    assert prices[0].operation == "read"
    assert prices[0].amount == Decimal("0.02")


def test_the_base_deployment_is_read_alone(monkeypatch, tmp_path):
    # Reading base over base would be a no-op, and asking for it is how an
    # operator inspects the layer itself.
    _apps(tmp_path, base=FULL_DEPLOYMENT)
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    assert load_config(app="base").chunk.size == 800


def test_a_client_with_no_base_beside_it_stands_alone(monkeypatch, tmp_path):
    _apps(tmp_path, example=FULL_DEPLOYMENT)
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    assert load_config(app="example").chunk.size == 800


def test_an_explicit_path_is_not_layered_over_the_base(monkeypatch, tmp_path):
    # A test and a tool point at a file outside the tree; layering the
    # repository's base into it would surprise both.
    _apps(tmp_path, base={**FULL_DEPLOYMENT, "process": {**FULL_DEPLOYMENT["process"], "chunk": {"size": 4321}}})
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    assert load_config(write_config(FULL_DEPLOYMENT)).chunk.size == 800


def test_a_problem_names_the_client_s_own_file_and_not_the_base(monkeypatch, tmp_path):
    # It is the file somebody edits; a base setting the client did not touch is
    # still the client's to override or to fix upstream.
    _apps(
        tmp_path,
        base=FULL_DEPLOYMENT,
        example={"process": {"chunk": {"size": "not-a-number"}}},
    )
    monkeypatch.setattr("src.config.APPS_DIRECTORY", tmp_path)

    with pytest.raises(ConfigError) as excinfo:
        load_config(app="example")

    message = str(excinfo.value)
    assert str(tmp_path / "example" / CONFIG_FILENAME) in message
    assert str(tmp_path / "base" / CONFIG_FILENAME) not in message
