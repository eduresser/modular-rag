from __future__ import annotations

import json
from dataclasses import dataclass

from src.sources.gdrive import GoogleDriveSource
from src.sync_decision import Decision, sweep_because

REASON_ANOTHER_IDENTITY = "position_belongs_to_another_identity"
REASON_ANOTHER_TREE = "position_describes_another_tree"
REASON_AHEAD_OF_SOURCE = "position_ahead_of_the_source"


def service_account_identity(credentials_json: str) -> str:
    try:
        address = json.loads(credentials_json).get("client_email", "")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "the source credential is not the JSON a service account key is, so the "
            "identity a feed position would belong to cannot be read from it"
        ) from error
    if not address:
        raise ValueError(
            "the service account key carries no client_email, so there is no identity to "
            "stamp a feed position with"
        )
    return str(address)


def _is_ahead(position: str, current: str) -> bool:
    return position.isdigit() and current.isdigit() and int(position) > int(current)


@dataclass(frozen=True, slots=True)
class DriveResume:

    source: GoogleDriveSource
    identity: str
    folders: tuple[str, ...]

    @property
    def scope(self) -> str:
        return ",".join(sorted(set(self.folders)))

    @property
    def origin(self) -> tuple[str, str]:
        return (self.identity, self.scope)

    def objection(self, position: str, issued_by: str, scope: str) -> Decision | None:
        if issued_by != self.identity:
            return sweep_because(
                REASON_ANOTHER_IDENTITY,
                f"the position was taken by {issued_by!r} and this run authenticates as "
                f"{self.identity!r}; Drive counts changes per credential, so the mark means "
                "something else to this one",
            )
        if scope != self.scope:
            return sweep_because(
                REASON_ANOTHER_TREE,
                f"the position describes folders {scope!r} and this deployment watches "
                f"{self.scope!r}; the enumeration behind the mark is not this corpus",
            )
        current = self.source.start_position()
        if _is_ahead(position, current):
            return sweep_because(
                REASON_AHEAD_OF_SOURCE,
                f"the stored position {position!r} is past the source's own {current!r}; a "
                "position ahead of the source answers every delta with an empty page, and the "
                "archive would freeze without a single error",
            )
        return None
