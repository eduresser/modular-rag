from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.usage import (
    REASON_CONFIGURATION_CHANGED,
    REASON_CONTENT_CHANGED,
    REASON_NEW_DOCUMENT,
    REASON_OPERATOR_REQUESTED,
    REASON_WATERMARK_MOVED,
)


class Action(Enum):

    INDEX_FROM_SOURCE = "index_from_source"
    RECUT_FROM_STORED = "recut_from_stored"
    TOUCH_ONLY = "touch_only"


@dataclass(frozen=True, slots=True)
class Decision:

    action: Action
    reason: str | None
    explanation: str

    @property
    def costs_a_read(self) -> bool:
        return self.action is Action.INDEX_FROM_SOURCE

    @property
    def costs_anything(self) -> bool:
        return self.action is not Action.TOUCH_ONLY


@dataclass(frozen=True, slots=True)
class StoredIdentity:

    external_id: str
    content_digest: str
    indexing_fingerprint: str
    revision_marker: str | None = None


def before_fetch(
    *,
    stored: StoredIdentity | None,
    revision_marker: str | None,
    fingerprint: str,
    operator_requested: bool = False,
) -> Decision | None:
    if stored is None:
        return Decision(
            action=Action.INDEX_FROM_SOURCE,
            reason=REASON_NEW_DOCUMENT,
            explanation="not seen before",
        )
    if operator_requested:
        return None
    if revision_marker is None:
        return None
    if revision_marker != stored.revision_marker:
        return None
    if fingerprint != stored.indexing_fingerprint:
        return Decision(
            action=Action.RECUT_FROM_STORED,
            reason=REASON_CONFIGURATION_CHANGED,
            explanation=(
                "the source claims no change but the indexing configuration moved, so the "
                "pieces are re-cut from the reader's stored output rather than read again"
            ),
        )
    return Decision(
        action=Action.TOUCH_ONLY,
        reason=None,
        explanation=(
            "the marker did not move and the configuration did not change, so nothing "
            "happened that could be attributed to anything"
        ),
    )


def after_fetch(
    *,
    stored: StoredIdentity,
    content_digest: str,
    fingerprint: str,
    operator_requested: bool = False,
) -> Decision:
    if operator_requested:
        return Decision(
            action=Action.INDEX_FROM_SOURCE,
            reason=REASON_OPERATOR_REQUESTED,
            explanation="an operator asked for this document to be reprocessed",
        )
    if content_digest != stored.content_digest:
        return Decision(
            action=Action.INDEX_FROM_SOURCE,
            reason=REASON_CONTENT_CHANGED,
            explanation="the exported bytes differ from the ones indexed",
        )
    if fingerprint != stored.indexing_fingerprint:
        return Decision(
            action=Action.RECUT_FROM_STORED,
            reason=REASON_CONFIGURATION_CHANGED,
            explanation=(
                "the bytes are identical but the indexing configuration moved, so the pieces "
                "are re-cut from the reader's stored output"
            ),
        )
    return Decision(
        action=Action.TOUCH_ONLY,
        reason=REASON_WATERMARK_MOVED,
        explanation=(
            "the source's revision marker moved but the exported bytes are identical, so "
            "nothing is read, cut or embedded (AC-018)"
        ),
    )
