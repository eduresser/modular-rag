"""Every committed deployment loads, before it reaches one.

Answers Q-081. A bad file stops the boot, which is the same guarantee the
environment gave — but a file can be checked before a merge, and that is the
whole reason for putting configuration in one. Already the house pattern: the
module settings tests read committed files rather than describing what they
should contain.

`_template` is excluded by name, and the exclusion is asserted rather than
implied: its required fields are blank on purpose, and a template that
validated would not be forcing anybody to fill it in.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from src.config import BASE_APP, CONFIG_FILENAME, SECRET_FIELD_SUFFIX, ConfigError, load_config
from src.secrets import is_reference

ROOT = pathlib.Path(__file__).resolve().parent.parent
APPS = ROOT / "apps"
TEMPLATE = "_template"


def _deployments() -> list[str]:
    """Every committed client: a directory with a configuration file, not the
    template and not the base layer."""
    return sorted(
        entry.name
        for entry in APPS.iterdir()
        if entry.is_dir()
        and entry.name not in (TEMPLATE, BASE_APP)
        and (entry / CONFIG_FILENAME).is_file()
    )


def test_there_is_at_least_one_committed_deployment():
    # Without this the parametrised tests below would pass by running nothing,
    # which is the failure mode of every test that iterates a directory.
    assert _deployments(), f"no deployment under {APPS}"


@pytest.mark.parametrize("app", _deployments())
def test_every_committed_deployment_loads(app):
    config = load_config(app=app)

    # A deployment whose records are attributed to another client is worse than
    # one that will not start, so the name in the file and the directory it
    # sits in have to agree.
    assert config.telemetry.client == app


def test_examples_committed_schedule_matches_the_documented_daily_sweep():
    """Q-090's answer, locked in: 03:00 UTC is midnight in Brasília, and the
    incremental schedule ships empty because that freshness has not been
    asked for.
    """
    config = load_config(app="example")

    assert config.schedule.full_sweep == "0 3 * * *"
    assert config.schedule.incremental == ""


def test_every_non_empty_committed_schedule_is_quoted_in_the_source_text():
    """ASM-095, measured: `0 3 * * *` parses bare because it opens with a
    digit, but `*/15 * * * *` opens with YAML's own alias marker and the file
    fails to parse — so the rule cannot be "quote when it starts with a star",
    and this is asserted against the source text rather than trusted.
    """
    checked = 0
    for path in APPS.glob(f"*/{CONFIG_FILENAME}"):
        root = yaml.compose(path.read_text())
        schedule_node = next(
            (value for key, value in root.value if key.value == "schedule"), None
        )
        if schedule_node is None:
            continue
        for key_node, value_node in schedule_node.value:
            if key_node.value not in ("full_sweep", "incremental") or not value_node.value:
                continue
            checked += 1
            assert value_node.style in ("'", '"'), (
                f"{path}: schedule.{key_node.value} ({value_node.value!r}) is not quoted"
            )
    assert checked, "no non-empty committed schedule was found to check"


@pytest.mark.parametrize("app", _deployments() + [BASE_APP, TEMPLATE])
def test_no_committed_file_carries_a_secret_value(app):
    """Principle P-012, over every committed file rather than over the shapes.

    A credential field names where its value is kept; anything else in one is a
    credential in version control. Checked here as well as at boot because this
    is the check that runs on a pull request.
    """
    document = yaml.safe_load((APPS / app / CONFIG_FILENAME).read_text()) or {}

    for section, settings in document.items():
        for name, value in (settings or {}).items():
            if not name.endswith(SECRET_FIELD_SUFFIX) or value is None:
                continue
            assert is_reference(value), f"{app}: {section}.{name} carries a value, not a reference"


def test_the_base_layer_does_not_validate_on_its_own():
    """It is a layer, not a deployment, and that is why it has no default.

    Seven settings are the client's to state; the base naming any of them would
    let a client that forgot to override one start against the wrong corpus.
    Four of them are credentials, which is the reason the number moves here
    rather than in a comment somewhere: the base holds no reference at all
    (ASM-001), so every stage that gains one gains a line a client has to write
    for itself — and every stage that stops being two gives one back.

    It was eleven while the read and cut halves were sibling sections each
    naming the credential it spent, and ten once they became one process making
    one call. The process stage then gained a **second module** and stopped
    supplying anything at all, which took it to eight — a layer merges and never
    removes, so a setting the base named for one module would reach a client
    that named the other (AC-270).

    **Eight** is seven plus the tool surface's own credential: `mcp:` names the
    reference that opens it rather than borrowing the API's, which is what lets
    the assistant's token be revoked without breaking a program's integration
    (AC-297). Five of the eight are now credentials, and that is the shape of
    this number — the base holds no reference at all, so a surface is a line a
    client writes for itself.

    Seven was that, minus `process.type`, which the base names again and may:
    `type` is the one key every module has, so it replaces cleanly and leaks
    nothing. What it buys is a default for the client who never thought about
    it, and the free route is the one to give them — it costs nothing and
    refuses out loud what it cannot read, where the paid one would quietly spend
    ten dollars a thousand pages.
    """
    with pytest.raises(ConfigError) as excinfo:
        load_config(APPS / BASE_APP / CONFIG_FILENAME)

    missing = [line for line in str(excinfo.value).splitlines() if "missing required" in line]
    assert len(missing) == 8, "\n".join(missing)
    # And the process stage reports none of them, having its answer already.
    assert not [line for line in missing if "[process]" in line], missing


def test_the_template_is_excluded_because_it_cannot_validate():
    """The exclusion above is asserted, not assumed.

    A template whose required fields were filled in would stop forcing anybody
    to fill them in, and the day it silently starts validating is the day it
    stops being a template.
    """
    with pytest.raises(ConfigError):
        load_config(APPS / TEMPLATE / CONFIG_FILENAME)

    assert TEMPLATE not in _deployments()


# ── The catalogue of what the environment still carries ───────────────────
#
# Since the configuration moved into a file, the environment carries three
# kinds of variable and nothing else: which deployment this is, the secret
# values a deployment references, and the Makefile's knobs. `.env.example` is
# where that list is written down, and this is what stops it falling behind the
# code — the lesson of every committed example file in this repository.

ENV_CATALOGUE = ROOT / ".env.example"
MAKEFILE = ROOT / "Makefile"


def _catalogued() -> set[str]:
    return {
        line.split("=", 1)[0].strip()
        for line in ENV_CATALOGUE.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }


def _referenced_variables() -> set[str]:
    """Every environment variable a committed deployment points at.

    Walked over the parsed documents rather than grepped: `env:NAME` appears in
    those files' comments too, and a catalogue built from prose would demand a
    variable nothing reads.
    """
    from src.secrets import SCHEME_ENV, scheme_of, target_of

    names: set[str] = set()
    for config in APPS.glob(f"*/{CONFIG_FILENAME}"):
        document = yaml.safe_load(config.read_text()) or {}
        for settings in document.values():
            for value in (settings or {}).values():
                if scheme_of(value) == SCHEME_ENV:
                    names.add(target_of(value))
    return names


def _makefile_knobs() -> set[str]:
    """Every `VAR ?=` in the Makefile — the ones the environment can override."""
    return set(re.findall(r"^([A-Z_]+) \?=", MAKEFILE.read_text(), re.MULTILINE))


def _provision_inputs() -> set[str]:
    return {"DATABASE_URL"}


def test_the_catalogue_names_every_variable_the_build_can_read():
    from src.config import APP_VARIABLE

    readable = {APP_VARIABLE} | _referenced_variables() | _makefile_knobs() | _provision_inputs()

    assert readable, "nothing readable means this test proves nothing"
    assert readable <= _catalogued(), sorted(readable - _catalogued())


def test_the_catalogue_names_nothing_the_build_cannot_read():
    # A variable in the catalogue that nothing reads is worse than a missing
    # one: somebody sets it and wonders why it had no effect.
    from src.config import APP_VARIABLE

    readable = {APP_VARIABLE} | _referenced_variables() | _makefile_knobs() | _provision_inputs()

    assert _catalogued() <= readable, sorted(_catalogued() - readable)


def test_the_catalogue_says_the_retired_variable_is_refused():
    # It named a path for one commit before `APP` replaced it, and a shell may
    # still export it. Mentioned in prose rather than as a key, because it is
    # not a variable anybody should set.
    from src.config import RETIRED_PATH_VARIABLE

    text = ENV_CATALOGUE.read_text()

    assert RETIRED_PATH_VARIABLE in text
    assert RETIRED_PATH_VARIABLE not in _catalogued()


def test_the_catalogue_carries_no_value_for_a_credential():
    # Principle P-012 over the one committed file that names credentials by the
    # variable holding them: the list is real and an operator needs it, and a
    # value pasted here would be a credential in version control.
    referenced = _referenced_variables()
    assert referenced, "no credential is referenced by an environment variable"

    for line in ENV_CATALOGUE.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() in referenced:
            assert value.strip() == "", line
