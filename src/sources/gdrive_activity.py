from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import httpx

from src.sources.base import (
    ActivityEvent,
    DeletionEvent,
    DeletionKind,
    MoveEvent,
    PermissionChangeEvent,
    SourceError,
    group_by_document,
)

__all__ = [
    "ActivityEvent",
    "ActivityHistory",
    "DeletionEvent",
    "DeletionKind",
    "MoveEvent",
    "PermissionChangeEvent",
    "SCOPE_ACTIVITY_READONLY",
    "group_by_document",
]
from src.sources.gdrive_throttle import UNITS_LIST, QuotaBudget, with_retries

API_ROOT = "https://driveactivity.googleapis.com/v2"
SCOPE_ACTIVITY_READONLY = "https://www.googleapis.com/auth/drive.activity.readonly"

_ACTION_FILTER = "detail.action_detail_case:(DELETE MOVE PERMISSION_CHANGE)"

PAGE_SIZE = 100

TOO_MANY_REQUESTS = 429


class ActivityHistory:

    def __init__(
        self,
        *,
        folder_ids: Sequence[str],
        http: httpx.Client,
        token: Callable[[], str],
        budget: QuotaBudget | None = None,
        max_attempts: int = 6,
    ) -> None:
        folders = tuple(dict.fromkeys(folder for folder in folder_ids if folder))
        if not folders:
            raise SourceError("an activity history has to name the folders it watches")
        self._folder_ids = folders
        self._http = http
        self._token = token
        self._budget = budget or QuotaBudget()
        self._max_attempts = max_attempts

    def events_since(
        self, moment: datetime, *, until: datetime | None = None
    ) -> Iterator[ActivityEvent]:
        for folder_id in self._folder_ids:
            token: str | None = None
            while True:
                payload = self._query(
                    folder_id=folder_id, since=moment, until=until, page_token=token
                )
                for activity in payload.get("activities", []):
                    yield from _events(activity)
                token = payload.get("nextPageToken")
                if not token:
                    break

    def _query(
        self, *, folder_id: str, since: datetime, until: datetime | None, page_token: str | None
    ) -> dict:
        body: dict = {
            "ancestorName": f"items/{folder_id}",
            "filter": _filter(since, until),
            "pageSize": PAGE_SIZE,
        }
        if page_token:
            body["pageToken"] = page_token

        def attempt() -> httpx.Response:
            response = self._http.post(
                f"{API_ROOT}/activity:query",
                json=body,
                headers={"Authorization": f"Bearer {self._token()}"},
            )
            _raise_for(response)
            return response

        response = with_retries(
            attempt,
            is_rate_limited=_is_rate_limited,
            budget=self._budget,
            units=UNITS_LIST,
            max_attempts=self._max_attempts,
        )
        return response.json()  # type: ignore[union-attr]


def _events(activity: dict) -> Iterator[ActivityEvent]:
    detail = activity.get("primaryActionDetail") or {}
    moment = _occurred_at(activity)
    for target in activity.get("targets", []):
        drive_item = target.get("driveItem")
        if not drive_item:
            continue
        external_id = _item_id(drive_item.get("name"))
        if external_id is None:
            continue
        if "delete" in detail:
            permanent = detail["delete"].get("type") == "PERMANENT_DELETE"
            yield DeletionEvent(
                external_id=external_id,
                kind=DeletionKind.PERMANENTLY_DELETED if permanent else DeletionKind.TRASHED,
                occurred_at=moment,
            )
        elif "move" in detail:
            removed = tuple(
                parent_id
                for parent_id in (
                    _item_id((parent.get("driveItem") or {}).get("name"))
                    for parent in detail["move"].get("removedParents", [])
                )
                if parent_id is not None
            )
            yield MoveEvent(external_id=external_id, removed_parents=removed, occurred_at=moment)
        elif "permissionChange" in detail:
            removed_permissions = tuple(
                str(permission.get("role", "unknown"))
                for permission in detail["permissionChange"].get("removedPermissions", [])
            )
            yield PermissionChangeEvent(
                external_id=external_id,
                removed_permissions=removed_permissions,
                occurred_at=moment,
            )


def _item_id(name: str | None) -> str | None:
    if not name:
        return None
    return name.removeprefix("items/")


def _occurred_at(activity: dict) -> datetime | None:
    timestamp = activity.get("timestamp")
    if timestamp:
        return _parse_moment(timestamp)
    time_range = activity.get("timeRange") or {}
    end = time_range.get("endTime")
    return _parse_moment(end) if end else None


def _parse_moment(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _filter(since: datetime, until: datetime | None) -> str:
    clauses = [_ACTION_FILTER, f'time >= "{_iso(since)}"']
    if until is not None:
        clauses.append(f'time < "{_iso(until)}"')
    return " ".join(clauses)


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _raise_for(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    if response.status_code == TOO_MANY_REQUESTS:
        raise _RateRefusal(response.text)
    raise SourceError(
        f"the activity history refused with {response.status_code}: {_reason(response)}; "
        'a failed query is never read as "no activity happened" (ASM-098)'
    )


class _RateRefusal(Exception):
    ...


def _is_rate_limited(error: Exception) -> bool:
    return isinstance(error, _RateRefusal)


def _reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    error = payload.get("error") or {}
    errors = error.get("errors") or []
    first = errors[0] if errors else {}
    return str(first.get("reason") or error.get("status") or error.get("message") or response.text)
