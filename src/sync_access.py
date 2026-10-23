from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from src.sources.base import (
    ActivityEvent,
    DeletionEvent,
    DocumentUnavailable,
    MoveEvent,
    PermissionChangeEvent,
    Source,
    SourceDocument,
)

_NO_EXPLANATIONS: tuple[ActivityEvent, ...] = ()


class Verdict(Enum):
    GONE = "gone"
    ACCESS_LOST = "access_lost"
    UNPROVEN = "unproven"
    STILL_THERE = "still_there"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    external_id: str
    verdict: Verdict
    detail: str
    document: SourceDocument | None = None

    @property
    def removable(self) -> bool:
        return self.verdict is Verdict.GONE


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    results: tuple[ProbeResult, ...]

    @property
    def gone(self) -> tuple[str, ...]:
        return tuple(r.external_id for r in self.results if r.verdict is Verdict.GONE)

    @property
    def access_lost(self) -> tuple[str, ...]:
        return tuple(r.external_id for r in self.results if r.verdict is Verdict.ACCESS_LOST)

    @property
    def unproven(self) -> tuple[str, ...]:
        return tuple(r.external_id for r in self.results if r.verdict is Verdict.UNPROVEN)

    @property
    def still_there(self) -> tuple[ProbeResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.STILL_THERE)

    @property
    def lost_any_access(self) -> bool:
        return bool(self.access_lost)

    def report(self) -> str:
        parts = [f"{len(self.gone)} confirmed gone"]
        if self.access_lost:
            parts.append(
                f"{len(self.access_lost)} left indexed because access was refused or explained "
                f"otherwise, not because they were deleted"
            )
        if self.unproven:
            parts.append(
                f"{len(self.unproven)} left indexed because the source offered no proof either way"
            )
        if self.still_there:
            parts.append(f"{len(self.still_there)} still present despite being reported removed")
        return "; ".join(parts)


def _is_deletion_proof(refusal: DocumentUnavailable) -> bool:
    return refusal.trashed


def probe_removal(
    source: Source,
    external_id: str,
    *,
    explained_by: Sequence[ActivityEvent] = _NO_EXPLANATIONS,
    carried_by: Sequence[ActivityEvent] = _NO_EXPLANATIONS,
) -> ProbeResult:
    try:
        document = source.probe(external_id)
    except DocumentUnavailable as exc:
        if exc.denied:
            return ProbeResult(
                external_id=external_id,
                verdict=Verdict.ACCESS_LOST,
                detail=(
                    f"the source refused access rather than reporting absence ({exc.reason}); "
                    "the document stays indexed, because a permission change is not a deletion"
                ),
            )
        if _is_deletion_proof(exc):
            return ProbeResult(
                external_id=external_id,
                verdict=Verdict.GONE,
                detail=f"the source answered that it does not exist ({exc.reason})",
            )
        for event in explained_by:
            if isinstance(event, DeletionEvent):
                return ProbeResult(
                    external_id=external_id,
                    verdict=Verdict.GONE,
                    detail=f"the activity history recorded a deletion ({event.kind.value})",
                )
            if isinstance(event, MoveEvent):
                return ProbeResult(
                    external_id=external_id,
                    verdict=Verdict.ACCESS_LOST,
                    detail=(
                        "the activity history recorded a move out of the watched tree; "
                        "the document stays indexed, because a move is not a deletion"
                    ),
                )
            if isinstance(event, PermissionChangeEvent):
                return ProbeResult(
                    external_id=external_id,
                    verdict=Verdict.ACCESS_LOST,
                    detail=(
                        "the activity history recorded a permission change; the document "
                        "stays indexed, because a permission change is not a deletion"
                    ),
                )
        for event in carried_by:
            if isinstance(event, MoveEvent):
                return ProbeResult(
                    external_id=external_id,
                    verdict=Verdict.ACCESS_LOST,
                    detail=(
                        f"a folder above it ({event.external_id}) was moved out of the watched "
                        "tree and took it along; the document stays indexed, because a move is "
                        "not a deletion"
                    ),
                )
            if isinstance(event, PermissionChangeEvent):
                return ProbeResult(
                    external_id=external_id,
                    verdict=Verdict.ACCESS_LOST,
                    detail=(
                        f"a folder above it ({event.external_id}) had a permission removed and "
                        "the document went out of sight with it; it stays indexed, because a "
                        "permission change is not a deletion"
                    ),
                )
        return ProbeResult(
            external_id=external_id,
            verdict=Verdict.UNPROVEN,
            detail=(
                f"the source answered not found ({exc.reason}), with no deletion event in its "
                "activity history either, and nothing above it explains the disappearance; "
                "absence is never proof (principle P-005), so the document stays indexed and "
                "is counted as unproven"
            ),
        )
    return ProbeResult(
        external_id=external_id,
        verdict=Verdict.STILL_THERE,
        detail="the source handed it over, so the feed's report was premature",
        document=document,
    )


def probe_removals(
    source: Source,
    external_ids: Iterable[str],
    *,
    explanations: Mapping[str, Sequence[ActivityEvent]] | None = None,
    ancestors: Mapping[str, Sequence[str]] | None = None,
) -> ProbeSummary:
    identifiers: Sequence[str] = sorted({identifier for identifier in external_ids if identifier})
    explanations = explanations or {}
    ancestors = ancestors or {}
    return ProbeSummary(
        results=tuple(
            probe_removal(
                source,
                identifier,
                explained_by=explanations.get(identifier, _NO_EXPLANATIONS),
                carried_by=tuple(
                    event
                    for folder in ancestors.get(identifier, ())
                    for event in explanations.get(folder, ())
                ),
            )
            for identifier in identifiers
        )
    )
