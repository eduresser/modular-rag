"""Unit-level coverage for src/sources/gdrive_activity.py.

AC-012 and AC-013 are proven where CLAUDE.md says every criterion's proof
lives — the tagged stub in tests/test_spec_source_gdrive.py — once the tasks
that consume this module's events (T-015, T-017, T-020) exist to complete the
reconciliation those tests exercise. This file covers what those tests will
not: the request this module builds, how it turns one activity entry into an
event of each kind, paging, and the one failure mode ASM-098 exists to
prevent — a failed query read as silence.

Faked at the HTTP layer, like the Drive source itself: what has to be proven
is the request that goes out (the ancestor scope, the action filter, the time
window) and the parsing of what comes back, neither of which a stubbed class
would exercise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from src.sources.base import SourceError
from src.sources.gdrive_activity import (
    ActivityHistory,
    DeletionEvent,
    DeletionKind,
    MoveEvent,
    PermissionChangeEvent,
    group_by_document,
)
from src.sources.gdrive_throttle import QuotaBudget, RateLimited

FOLDER = "root-folder"
SINCE = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _deletion(external_id: str, *, permanent: bool = False, when: str = "2026-09-10T12:00:00.000Z") -> dict:
    return {
        "primaryActionDetail": {"delete": {"type": "PERMANENT_DELETE" if permanent else "TRASH"}},
        "targets": [{"driveItem": {"name": f"items/{external_id}"}}],
        "timestamp": when,
    }


def _move_detail(removed_parents: tuple[str, ...]) -> dict:
    return {"removedParents": [{"driveItem": {"name": f"items/{parent}"}} for parent in removed_parents]}


def _move_activity(external_id: str, *, removed_parents: tuple[str, ...], when: str = "2026-09-10T12:00:00.000Z") -> dict:
    return {
        "primaryActionDetail": {"move": _move_detail(removed_parents)},
        "targets": [{"driveItem": {"name": f"items/{external_id}"}}],
        "timestamp": when,
    }


def _permission_change(
    external_id: str, *, removed_roles: tuple[str, ...], when: str = "2026-09-10T12:00:00.000Z"
) -> dict:
    return {
        "primaryActionDetail": {
            "permissionChange": {
                "removedPermissions": [{"role": role} for role in removed_roles],
            }
        },
        "targets": [{"driveItem": {"name": f"items/{external_id}"}}],
        "timestamp": when,
    }


class FakeActivityAPI:
    """The activity endpoint, answering over HTTP without a network."""

    def __init__(self, activities: list[dict] | None = None, *, page_size: int = 100) -> None:
        self.activities = list(activities or [])
        self.page_size = page_size
        self.requests: list[httpx.Request] = []
        self.status_override: int | None = None
        self.rate_limit_first = 0

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.rate_limit_first > 0:
            self.rate_limit_first -= 1
            return httpx.Response(
                429,
                json={"error": {"errors": [{"reason": "userRateLimitExceeded"}]}},
                request=request,
            )
        if self.status_override is not None:
            return httpx.Response(
                self.status_override,
                json={"error": {"message": "internal error"}},
                request=request,
            )
        body = json.loads(request.content or b"{}")
        token = body.get("pageToken")
        start = int(token) if token else 0
        page = self.activities[start : start + self.page_size]
        payload: dict = {"activities": page}
        end = start + len(page)
        if end < len(self.activities):
            payload["nextPageToken"] = str(end)
        return httpx.Response(200, json=payload, request=request)


def _history(api: FakeActivityAPI, *, folder_ids=(FOLDER,), **kwargs) -> ActivityHistory:
    return ActivityHistory(
        folder_ids=folder_ids, http=api.client(), token=lambda: "fake-token", **kwargs
    )


def test_the_query_scopes_to_the_watched_folders_ancestry_and_the_three_kinds_of_event():
    api = FakeActivityAPI()
    list(_history(api).events_since(SINCE))

    assert api.requests, "no query went out"
    body = json.loads(api.requests[0].content)
    assert body["ancestorName"] == f"items/{FOLDER}"
    assert "DELETE" in body["filter"]
    assert "MOVE" in body["filter"]
    assert "PERMISSION_CHANGE" in body["filter"]


def test_the_time_window_is_a_lower_bound_and_an_optional_upper_one():
    api = FakeActivityAPI()
    list(_history(api).events_since(SINCE))
    lower_only = json.loads(api.requests[0].content)["filter"]
    assert "2026-09-01" in lower_only

    until = datetime(2026, 9, 11, tzinfo=timezone.utc)
    api2 = FakeActivityAPI()
    list(_history(api2).events_since(SINCE, until=until))
    both = json.loads(api2.requests[0].content)["filter"]
    assert "2026-09-01" in both
    assert "2026-09-11" in both


def test_a_deletion_event_reports_trashed_or_permanently_deleted():
    api = FakeActivityAPI([_deletion("trashed-doc"), _deletion("purged-doc", permanent=True)])

    events = list(_history(api).events_since(SINCE))

    assert events == [
        DeletionEvent(
            external_id="trashed-doc",
            kind=DeletionKind.TRASHED,
            occurred_at=datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc),
        ),
        DeletionEvent(
            external_id="purged-doc",
            kind=DeletionKind.PERMANENTLY_DELETED,
            occurred_at=datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc),
        ),
    ]


def test_a_move_event_reports_the_parents_it_left():
    api = FakeActivityAPI([_move_activity("moved-doc", removed_parents=(FOLDER, "sub-folder"))])

    events = list(_history(api).events_since(SINCE))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, MoveEvent)
    assert event.external_id == "moved-doc"
    assert event.removed_parents == (FOLDER, "sub-folder")


def test_a_permission_change_event_reports_the_permissions_removed():
    api = FakeActivityAPI([_permission_change("locked-doc", removed_roles=("reader", "writer"))])

    events = list(_history(api).events_since(SINCE))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PermissionChangeEvent)
    assert event.external_id == "locked-doc"
    assert event.removed_permissions == ("reader", "writer")


def test_an_activity_naming_no_drive_item_yields_nothing():
    # A comment or a settings change on the drive itself carries no driveItem
    # target — there is no document here for a reconciliation to act on.
    activity = {
        "primaryActionDetail": {"delete": {"type": "TRASH"}},
        "targets": [{"comment": {"legacyCommentId": "abc"}}],
        "timestamp": "2026-09-10T12:00:00.000Z",
    }
    api = FakeActivityAPI([activity])

    assert list(_history(api).events_since(SINCE)) == []


def test_paging_is_followed_until_there_is_no_next_page_token():
    activities = [_deletion(f"doc-{n}") for n in range(5)]
    api = FakeActivityAPI(activities, page_size=2)

    events = list(_history(api).events_since(SINCE))

    assert {event.external_id for event in events} == {f"doc-{n}" for n in range(5)}
    assert len(api.requests) >= 3, "a paged listing was not followed"


def test_events_can_be_grouped_by_the_document_they_name():
    api = FakeActivityAPI(
        [_deletion("doc-1"), _move_activity("doc-2", removed_parents=(FOLDER,)), _deletion("doc-1", permanent=True)]
    )

    grouped = group_by_document(_history(api).events_since(SINCE))

    assert set(grouped) == {"doc-1", "doc-2"}
    assert len(grouped["doc-1"]) == 2
    assert len(grouped["doc-2"]) == 1


def test_a_failed_query_raises_rather_than_being_read_as_no_activity():
    # ASM-098: an identifier the source does not recognise answers 500
    # INTERNAL, never a not-found. Reading that as silence would turn an
    # outage into proof that nothing happened under the folder.
    api = FakeActivityAPI()
    api.status_override = 500

    with pytest.raises(SourceError) as excinfo:
        list(_history(api).events_since(SINCE))

    assert "ASM-098" in str(excinfo.value)


def test_a_rate_limited_query_is_retried_rather_than_failing_outright():
    api = FakeActivityAPI([_deletion("doc-1")])
    api.rate_limit_first = 2
    clock = {"now": 0.0}
    budget = QuotaBudget(
        now=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)
    )

    events = list(_history(api, budget=budget).events_since(SINCE))

    assert [event.external_id for event in events] == ["doc-1"]
    assert clock["now"] > 0, "a refusal was retried without backing off"


def test_a_rate_limit_that_never_lets_up_raises_rather_than_looping():
    api = FakeActivityAPI([_deletion("doc-1")])
    api.rate_limit_first = 99
    clock = {"now": 0.0}
    budget = QuotaBudget(
        now=lambda: clock["now"], sleep=lambda seconds: clock.update(now=clock["now"] + seconds)
    )

    with pytest.raises(RateLimited):
        list(_history(api, budget=budget, max_attempts=3).events_since(SINCE))


def test_a_history_without_a_folder_is_refused():
    with pytest.raises(SourceError):
        ActivityHistory(folder_ids=[], http=httpx.Client(), token=lambda: "t")


def test_the_bearer_token_is_asked_for_on_every_request():
    asked = {"n": 0}

    def token() -> str:
        asked["n"] += 1
        return f"token-{asked['n']}"

    api = FakeActivityAPI([_deletion(f"doc-{n}") for n in range(3)], page_size=1)
    history = ActivityHistory(folder_ids=[FOLDER], http=api.client(), token=token)

    list(history.events_since(SINCE))

    assert len(api.requests) > 1
    assert asked["n"] == len(api.requests)
    tokens = [request.headers["Authorization"] for request in api.requests]
    assert len(set(tokens)) == len(tokens), "the token was captured once instead of asked for"


def test_one_query_goes_out_per_watched_folder():
    """ASM-130: the provider's filter names a single ancestor.

    So a deployment watching three folders asks three times, and the events of
    all three arrive as one history — which is what makes a deletion in the
    second folder proof exactly as one in the first is (AC-241).
    """
    api = FakeActivityAPI([_deletion("doc-1")])
    history = _history(api, folder_ids=("first-folder", "second-folder"))

    found = list(history.events_since(SINCE))

    ancestors = [json.loads(request.content)["ancestorName"] for request in api.requests]
    assert ancestors == ["items/first-folder", "items/second-folder"]
    # One deletion per folder asked, because this fake answers every ancestor
    # with the same history: the point is that neither answer is dropped.
    assert [event.external_id for event in found] == ["doc-1", "doc-1"]


def test_a_folder_named_twice_is_asked_about_once():
    api = FakeActivityAPI([_deletion("doc-1")])
    history = _history(api, folder_ids=("first-folder", "first-folder"))

    list(history.events_since(SINCE))

    assert len(api.requests) == 1
