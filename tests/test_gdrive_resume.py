"""Unit-level coverage for src/sources/gdrive_resume.py — Drive's own rules.

Everything asserted here is true of this source and no other, which is the
reason the module exists apart from ``src.sync_decision``. A SharePoint module
would bring its own file next to its own client, and none of these assertions
would carry over.
"""

import json

import pytest

from src.sources.gdrive_resume import (
    REASON_AHEAD_OF_SOURCE,
    REASON_ANOTHER_IDENTITY,
    REASON_ANOTHER_TREE,
    DriveResume,
    _is_ahead,
    service_account_identity,
)
from tests.conftest import FakeDrive, drive_source

FOLDER = "root-folder"
IDENTITY = "rag@example.iam.gserviceaccount.com"


def _resume(
    drive: FakeDrive, *, identity: str = IDENTITY, folders: tuple[str, ...] = (FOLDER,)
) -> DriveResume:
    return DriveResume(
        source=drive_source(drive, folder_ids=folders), identity=identity, folders=folders
    )


def test_only_digits_are_compared_as_numbers():
    """A position the provider stopped writing as an integer is not read as one.

    Drive documents the position as an opaque string and happens to issue
    integers. If that changed, reading `"abc" > "7"` as an ordering would sweep
    for ever — so the comparison applies only where both sides really are
    numbers, and otherwise declines to have an opinion.
    """
    assert _is_ahead("99", "7") is True
    assert _is_ahead("3", "7") is False
    assert _is_ahead("7", "7") is False
    assert _is_ahead("CAESAR-abc", "CAESAR-def") is False
    assert _is_ahead("99", "CAESAR-def") is False
    assert _is_ahead("CAESAR-abc", "7") is False


def test_the_two_stored_comparisons_come_before_the_one_that_costs_a_call():
    """Drive's last question is an API call, so it is asked last."""
    drive = FakeDrive(start_token="7")
    resume = _resume(drive)

    assert resume.objection("3", "someone-else@example.com", FOLDER).reason == (
        REASON_ANOTHER_IDENTITY
    )
    assert drive.requests == []

    assert resume.objection("3", IDENTITY, "another-folder").reason == REASON_ANOTHER_TREE
    assert drive.requests == []

    assert resume.objection("3", IDENTITY, FOLDER) is None
    assert [r.url.path for r in drive.requests] == ["/drive/v3/changes/startPageToken"]


def test_a_position_past_the_sources_own_is_refused_by_name():
    """The silent failure is the one worth naming in the record.

    Measured 2026-09-11: asking for a position past the credential's own
    returned 200 with an empty page — no error at all.
    """
    resume = _resume(FakeDrive(start_token="7"))
    objection = resume.objection("99", IDENTITY, FOLDER)
    assert objection.reason == REASON_AHEAD_OF_SOURCE
    assert "empty page" in objection.detail


def test_how_old_the_position_is_is_never_asked():
    """A position from the very beginning is as good as yesterday's.

    The provider states that a page token does not expire, and a token from the
    start of this credential's life still answered. There is no interval here
    to configure because there is no interval to observe.
    """
    resume = _resume(FakeDrive(start_token="9999"))
    assert resume.objection("1", IDENTITY, FOLDER) is None


def test_the_origin_is_what_a_position_taken_now_would_belong_to():
    resume = _resume(FakeDrive(start_token="7"))
    assert resume.origin == (IDENTITY, FOLDER)


def test_the_identity_is_read_from_the_credential_rather_than_configured():
    """Two places to say who we are is one place for them to disagree."""
    key = json.dumps({"client_email": IDENTITY, "private_key": "-----BEGIN..."})
    assert service_account_identity(key) == IDENTITY


def test_a_credential_that_names_nobody_is_refused_loudly():
    """Better a refusal at boot than a position stamped with an empty string.

    An empty identity would compare equal to the next empty identity, so every
    credential would look like the one that took the position — which is the
    check inverted into a hazard.
    """
    with pytest.raises(ValueError, match="no client_email"):
        service_account_identity(json.dumps({"private_key": "-----BEGIN..."}))
    with pytest.raises(ValueError, match="not the JSON"):
        service_account_identity("not json at all")
