from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import datetime

import httpx

from src.sources.base import (
    ChangePage,
    DocumentUnavailable,
    FetchedDocument,
    SourceDocument,
    SourceError,
)
from src.sources.gdrive_export import (
    KIND_FOLDER,
    KIND_SHORTCUT,
    Retrieval,
    content_digest,
    export_plan,
    is_document,
    reject_oversized,
    reject_oversized_content,
)
from src.sources.gdrive_throttle import (
    UNITS_DOWNLOAD,
    UNITS_GET,
    UNITS_LIST,
    QuotaBudget,
    with_retries,
)

API_ROOT = "https://www.googleapis.com/drive/v3"
SCOPE_READONLY = "https://www.googleapis.com/auth/drive.readonly"

FILE_FIELDS = (
    "id,name,mimeType,version,md5Checksum,modifiedTime,size,parents,trashed,"
    "webViewLink,shortcutDetails(targetId,targetMimeType)"
)
LIST_FIELDS = f"nextPageToken,files({FILE_FIELDS})"
CHANGE_FIELDS = f"nextPageToken,newStartPageToken,changes(fileId,removed,file({FILE_FIELDS}))"

ALL_DRIVES = {"supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}

PAGE_SIZE = 1000

NOT_FOUND = 404
FORBIDDEN = 403
TOO_MANY_REQUESTS = 429


class GoogleDriveSource:

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
            raise SourceError("a Drive source has to name at least one folder to read from")
        self._configured = folders
        self._http = http
        self._token = token
        self._budget = budget or QuotaBudget()
        self._max_attempts = max_attempts
        self._roots: tuple[str, ...] | None = None
        self._ancestry: dict[str, bool] = {folder: True for folder in folders}

    @property
    def configured_folders(self) -> tuple[str, ...]:
        return self._configured

    @property
    def folders(self) -> tuple[str, ...]:
        if self._roots is None:
            self._roots = self._resolve_roots()
        return self._roots


    def start_position(self) -> str:
        payload = self._call("GET", "/changes/startPageToken", params=ALL_DRIVES, units=UNITS_LIST)
        position = payload.get("startPageToken")
        if not position:
            raise SourceError(
                "the source returned no start position for its change feed, so no "
                "incremental run could ever know where to begin"
            )
        return str(position)

    def sweep(self) -> Iterator[SourceDocument]:
        seen: set[str] = set()
        pending = list(self.folders)
        while pending:
            folder = pending.pop()
            for entry in self._list_children(folder):
                kind = entry.get("mimeType", "")
                if kind == KIND_FOLDER:
                    pending.append(str(entry["id"]))
                    continue
                document = self._resolve(entry)
                if document is None or document.external_id in seen:
                    continue
                seen.add(document.external_id)
                yield document

    def changes_since(self, position: str | None) -> ChangePage:
        if not position:
            raise SourceError(
                "an incremental run needs a feed position; a first run takes one with "
                "start_position() before sweeping, so that nothing falls in the gap"
            )
        params = {"pageToken": position, "fields": CHANGE_FIELDS, "includeRemoved": "true",
                  "pageSize": str(PAGE_SIZE), **ALL_DRIVES}
        payload = self._call("GET", "/changes", params=params, units=UNITS_LIST)
        documents: list[SourceDocument] = []
        removed: list[str] = []
        seen: set[str] = set()
        for change in payload.get("changes", []):
            file_id = str(change.get("fileId", ""))
            entry = change.get("file") or {}
            if change.get("removed") or entry.get("trashed"):
                if file_id:
                    removed.append(file_id)
                continue
            document = self._resolve(entry)
            if document is None or document.external_id in seen:
                continue
            if not self._within_watched_tree(document.parents):
                continue
            seen.add(document.external_id)
            documents.append(document)
        next_token = payload.get("nextPageToken")
        new_start = payload.get("newStartPageToken")
        if not next_token and not new_start:
            raise SourceError(
                "the change feed returned neither a next page nor a new start position, so "
                "the run has nowhere to resume from"
            )
        return ChangePage(
            documents=tuple(documents),
            removed=tuple(removed),
            next_position=str(next_token or new_start),
            exhausted=next_token is None,
        )

    def probe(self, external_id: str) -> SourceDocument:
        entry = self._file(external_id)
        if entry.get("trashed"):
            raise DocumentUnavailable(
                external_id,
                "the source still answers for it, but marks it trashed — proof of "
                "deletion, not evidence of a permission change",
                denied=False,
                trashed=True,
            )
        document = self._resolve(entry)
        if document is None:
            raise DocumentUnavailable(
                external_id,
                f"the source holds {entry.get('mimeType', 'something')!r} under that "
                "identifier, which is not a document",
            )
        return document

    def fetch(self, document: SourceDocument) -> FetchedDocument:
        plan = export_plan(document.kind)
        reject_oversized(document.external_id, document.size_bytes)
        if plan.is_export:
            content = self._bytes(
                f"/files/{document.external_id}/export",
                params={"mimeType": plan.media_type, **ALL_DRIVES},
                external_id=document.external_id,
            )
        else:
            content = self._bytes(
                f"/files/{document.external_id}",
                params={"alt": "media", **ALL_DRIVES},
                external_id=document.external_id,
            )
        reject_oversized_content(document.external_id, content)
        return FetchedDocument(
            external_id=document.external_id,
            content=content,
            media_type=plan.media_type,
            content_digest=content_digest(content),
        )


    def _resolve(self, entry: dict) -> SourceDocument | None:
        if not entry:
            return None
        kind = entry.get("mimeType", "")
        if kind == KIND_SHORTCUT:
            target = (entry.get("shortcutDetails") or {}).get("targetId")
            if not target:
                raise SourceError(
                    f"shortcut {entry.get('id')!r} carried no target identifier; the "
                    "field projection is the usual cause (ASM-014)"
                )
            return self._resolve(self._file(str(target)))
        if not is_document(kind):
            return None
        return SourceDocument(
            external_id=str(entry["id"]),
            name=str(entry.get("name", "")),
            kind=kind,
            revision_marker=_marker(entry),
            modified_at=_moment(entry.get("modifiedTime")),
            size_bytes=int(entry["size"]) if entry.get("size") is not None else None,
            parents=tuple(str(parent) for parent in entry.get("parents", [])),
            link=entry.get("webViewLink"),
        )


    def _resolve_roots(self) -> tuple[str, ...]:
        entries: dict[str, dict] = {}
        refusals: list[str] = []
        for folder in self._configured:
            try:
                entries[folder] = self._folder_entry(folder)
            except SourceError as refusal:
                refusals.append(str(refusal))
        if refusals:
            raise SourceError("\n".join(refusals))
        listed = set(self._configured)
        return tuple(
            folder
            for folder in self._configured
            if not self._has_listed_ancestor(entries[folder], listed)
        )

    def _folder_entry(self, folder_id: str) -> dict:
        try:
            entry = self._file(folder_id)
        except DocumentUnavailable as unavailable:
            answer = (
                "access is denied" if unavailable.denied else "there is no such item"
            )
            raise SourceError(
                f"the watched folder {folder_id!r} cannot be read: the source answers that "
                f"{answer} ({unavailable.reason}). The run stops here rather than sweeping "
                "without it — a sweep that finished short of a whole folder would report "
                "itself complete, and every document under it would be absent from a "
                "completed sweep, which is what removes a document from the archive."
            ) from unavailable
        kind = str(entry.get("mimeType", ""))
        if kind != KIND_FOLDER:
            raise SourceError(
                f"the watched folder {folder_id!r} is not a folder: the source says it is "
                f"{kind or 'of no stated type'}. A root that is not a folder enumerates "
                "nothing, which is indistinguishable from an empty corpus; pasting a "
                "document's own link instead of the folder's is the usual cause."
            )
        if entry.get("trashed"):
            raise SourceError(
                f"the watched folder {folder_id!r} is in the source's trash. Its contents "
                "are trashed with it, so the sweep would enumerate nothing and read the "
                "whole folder as deleted."
            )
        return entry

    def _has_listed_ancestor(self, entry: dict, listed: set[str]) -> bool:
        seen = {str(entry.get("id", ""))}
        current = _first_parent(entry)
        while current and current not in seen:
            if current in listed:
                return True
            seen.add(current)
            try:
                above = self._file(current)
            except DocumentUnavailable:
                return False
            current = _first_parent(above)
        return False

    def _within_watched_tree(self, parents: Sequence[str]) -> bool:
        return any(self._folder_within(str(parent)) for parent in parents)

    def _folder_within(self, folder_id: str) -> bool:
        known = self._ancestry.get(folder_id)
        if known is not None:
            return known
        chain: list[str] = []
        current: str | None = folder_id
        verdict = False
        while current and current not in chain:
            cached = self._ancestry.get(current)
            if cached is not None:
                verdict = cached
                break
            chain.append(current)
            try:
                entry = self._file(current)
            except DocumentUnavailable:
                break
            current = _first_parent(entry)
        for folder in chain:
            self._ancestry[folder] = verdict
        return verdict

    def _list_children(self, folder_id: str) -> Iterator[dict]:
        token: str | None = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": LIST_FIELDS,
                "pageSize": str(PAGE_SIZE),
                **ALL_DRIVES,
            }
            if token:
                params["pageToken"] = token
            payload = self._call("GET", "/files", params=params, units=UNITS_LIST)
            yield from payload.get("files", [])
            token = payload.get("nextPageToken")
            if not token:
                return

    def _file(self, external_id: str) -> dict:
        return self._call(
            "GET",
            f"/files/{external_id}",
            params={"fields": FILE_FIELDS, **ALL_DRIVES},
            units=UNITS_GET,
            external_id=external_id,
        )


    def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str],
        units: int,
        external_id: str | None = None,
    ) -> dict:
        response = self._request(method, path, params=params, units=units, external_id=external_id)
        return response.json()

    def _bytes(self, path: str, *, params: dict[str, str], external_id: str) -> bytes:
        response = self._request(
            "GET", path, params=params, units=UNITS_DOWNLOAD, external_id=external_id
        )
        return response.content

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str],
        units: int,
        external_id: str | None,
    ) -> httpx.Response:
        def attempt() -> httpx.Response:
            response = self._http.request(
                method,
                f"{API_ROOT}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._token()}"},
            )
            _raise_for(response, external_id)
            return response

        return with_retries(  # type: ignore[return-value]
            attempt,
            is_rate_limited=_is_rate_limited,
            budget=self._budget,
            units=units,
            max_attempts=self._max_attempts,
        )


def _raise_for(response: httpx.Response, external_id: str | None) -> None:
    if response.status_code < 400:
        return
    if response.status_code == TOO_MANY_REQUESTS:
        raise _RateRefusal(response.text)
    reason = _reason(response)
    if response.status_code == FORBIDDEN and _is_rate_reason(reason):
        raise _RateRefusal(reason)
    if external_id is not None and response.status_code in (NOT_FOUND, FORBIDDEN):
        raise DocumentUnavailable(
            external_id, reason, denied=response.status_code == FORBIDDEN
        )
    raise SourceError(f"the source refused with {response.status_code}: {reason}")


class _RateRefusal(Exception):
    ...


def _is_rate_limited(error: Exception) -> bool:
    return isinstance(error, _RateRefusal)


_RATE_REASONS = frozenset(
    {"ratelimitexceeded", "userratelimitexceeded", "sharingratelimitexceeded", "quotaexceeded"}
)


def _is_rate_reason(reason: str) -> bool:
    lowered = reason.lower().replace(" ", "")
    return any(marker in lowered for marker in _RATE_REASONS)


def _reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    error = payload.get("error") or {}
    errors = error.get("errors") or []
    first = errors[0] if errors else {}
    return str(first.get("reason") or error.get("status") or error.get("message") or response.text)


def _first_parent(entry: dict) -> str | None:
    parents = entry.get("parents") or []
    return str(parents[0]) if parents else None


def _marker(entry: dict) -> str | None:
    checksum = entry.get("md5Checksum")
    if checksum:
        return str(checksum)
    version = entry.get("version")
    return str(version) if version is not None else None


def _moment(value: object) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class _TokenResponse:

    __slots__ = ("_response",)

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self._response.headers

    @property
    def data(self) -> bytes:
        return self._response.content


class _TokenTransport:

    __slots__ = ("_http",)

    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: object,
    ) -> _TokenResponse:
        deadline = {} if timeout is None else {"timeout": timeout}
        return _TokenResponse(
            self._http.request(method, url, content=body, headers=headers, **deadline)
        )


def service_account_token_provider(
    credentials_json: str,
    *,
    http: httpx.Client,
    scopes: tuple[str, ...] = (SCOPE_READONLY,),
) -> Callable[[], str]:
    import json

    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(
        json.loads(credentials_json), scopes=list(scopes)
    )
    transport = _TokenTransport(http)

    def token() -> str:
        if not credentials.valid:
            credentials.refresh(transport)
        return str(credentials.token)

    return token
