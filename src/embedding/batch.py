from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.embedding.client import Embedder, EmbeddingError


class BatchError(Exception):
    ...


@dataclass(frozen=True, slots=True)
class Group:

    key: str
    texts: tuple[str, ...]
    partial: bool = False

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise BatchError("a group has to name the document its pieces came from")
        if not self.texts:
            raise BatchError(f"group {self.key!r} carries no pieces")

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True, slots=True)
class Batch:

    groups: tuple[Group, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise BatchError("an empty batch is a request nobody has to make")

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(text for group in self.groups for text in group.texts)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(group.key for group in self.groups)

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups)


def split_oversized(group: Group, *, limit: int) -> tuple[Group, ...]:
    if len(group) <= limit:
        return (group,)
    return tuple(
        Group(key=group.key, texts=group.texts[start : start + limit], partial=True)
        for start in range(0, len(group), limit)
    )


def plan_batches(
    groups: Iterable[Group], *, batch_size: int, max_inputs: int
) -> tuple[Batch, ...]:
    if batch_size < 1:
        raise BatchError(f"a batch has to hold at least one piece, got {batch_size}")
    limit = min(batch_size, max_inputs)
    batches: list[Batch] = []
    current: list[Group] = []
    held = 0
    for group in groups:
        for part in split_oversized(group, limit=limit):
            if current and held + len(part) > limit:
                batches.append(Batch(groups=tuple(current)))
                current, held = [], 0
            current.append(part)
            held += len(part)
    if current:
        batches.append(Batch(groups=tuple(current)))
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class BatchOutcome:

    batch: Batch
    succeeded: bool
    vectors: tuple[tuple[float, ...], ...] = ()
    tokens: int = 0
    duration_ms: Decimal = Decimal(0)
    detail: str = ""
    refusal_status: int | None = None

    def by_group(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        if not self.succeeded:
            return {}
        cut: dict[str, tuple[tuple[float, ...], ...]] = {}
        position = 0
        for group in self.batch.groups:
            taken = self.vectors[position : position + len(group)]
            cut[group.key] = cut.get(group.key, ()) + taken
            position += len(group)
        return cut


@dataclass(frozen=True, slots=True)
class EmbeddingRun:

    outcomes: tuple[BatchOutcome, ...]

    @property
    def succeeded(self) -> bool:
        return all(outcome.succeeded for outcome in self.outcomes)

    @property
    def tokens(self) -> int:
        return sum(outcome.tokens for outcome in self.outcomes)

    @property
    def failed_keys(self) -> tuple[str, ...]:
        failed = {key for outcome in self.outcomes if not outcome.succeeded for key in outcome.batch.keys}
        seen: list[str] = []
        for outcome in self.outcomes:
            for key in outcome.batch.keys:
                if key in failed and key not in seen:
                    seen.append(key)
        return tuple(seen)

    def refusal_status_for(self, key: str) -> int | None:
        for outcome in self.outcomes:
            if not outcome.succeeded and key in outcome.batch.keys:
                return outcome.refusal_status
        return None

    def refusal_for(self, key: str) -> str:
        for outcome in self.outcomes:
            if not outcome.succeeded and key in outcome.batch.keys:
                return outcome.detail
        return ""

    @property
    def total(self) -> int:
        return sum(len(outcome.batch) for outcome in self.outcomes)

    @property
    def embedded(self) -> int:
        failed = set(self.failed_keys)
        return sum(
            len(group)
            for outcome in self.outcomes
            if outcome.succeeded
            for group in outcome.batch.groups
            if group.key not in failed
        )

    @property
    def unembedded(self) -> int:
        return self.total - self.embedded

    def vectors(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        failed = set(self.failed_keys)
        collected: dict[str, tuple[tuple[float, ...], ...]] = {}
        for outcome in self.outcomes:
            for key, vectors in outcome.by_group().items():
                if key in failed:
                    continue
                collected[key] = collected.get(key, ()) + vectors
        return collected


def embed_batches(
    batches: Sequence[Batch],
    embedder: Embedder,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> EmbeddingRun:
    outcomes: list[BatchOutcome] = []
    for batch in batches:
        started = clock()
        try:
            result = embedder.embed_documents(batch.texts)
        except EmbeddingError as exc:
            outcomes.append(
                BatchOutcome(
                    batch=batch,
                    succeeded=False,
                    duration_ms=_elapsed_ms(started, clock()),
                    detail=str(exc),
                    refusal_status=getattr(exc, "status", None),
                )
            )
            continue
        outcomes.append(
            BatchOutcome(
                batch=batch,
                succeeded=True,
                vectors=result.vectors,
                tokens=result.tokens,
                duration_ms=_elapsed_ms(started, clock()),
            )
        )
    return EmbeddingRun(outcomes=tuple(outcomes))


def _elapsed_ms(started: float, finished: float) -> Decimal:
    return Decimal(str(round(max(finished - started, 0.0) * 1000, 3)))
