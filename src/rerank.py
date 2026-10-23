from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.config import RerankSettings
from src.search_result import SearchHit


class RerankError(Exception):
    pass


class Scorer(Protocol):
    model: str

    def score(self, question: str, passages: Sequence[str]) -> Sequence[float]:
        ...


@dataclass(frozen=True, slots=True)
class Reranking:
    hits: tuple[SearchHit, ...]
    model: str = ""

    @property
    def happened(self) -> bool:
        return bool(self.model)


def wanted(settings: RerankSettings, requested: bool | None) -> bool:
    return settings.enabled if requested is None else requested


def scorer_for(settings: RerankSettings, scorer: Scorer | None) -> Scorer:
    if not settings.model.strip():
        raise RerankError(
            "reranking was asked for and no model is configured; set 'rerank.model', "
            "because a reranking recorded against no model is a before-and-after "
            "comparison with nothing on one side of it"
        )
    if scorer is None:
        raise RerankError(
            f"reranking was asked for with model {settings.model!r} and no scorer is "
            "wired into this deployment"
        )
    if scorer.model != settings.model:
        raise RerankError(
            f"the configured reranking model is {settings.model!r} and the wired scorer "
            f"is {scorer.model!r}; the search record would name a model that did not "
            "produce the ordering"
        )
    return scorer


def rerank(
    hits: Sequence[SearchHit],
    question: str,
    *,
    settings: RerankSettings,
    scorer: Scorer | None = None,
    requested: bool | None = None,
) -> Reranking:
    ordered = tuple(hits)
    if not wanted(settings, requested):
        return Reranking(hits=ordered)
    chosen = scorer_for(settings, scorer)
    if len(ordered) < 2:
        return Reranking(hits=ordered, model=chosen.model)
    scores = chosen.score(question, [hit.content for hit in ordered])
    if len(scores) != len(ordered):
        raise RerankError(
            f"the reranking model {chosen.model!r} scored {len(scores)} of "
            f"{len(ordered)} passages; a partial ordering cannot be applied to the rest"
        )
    ranked = sorted(
        zip(ordered, scores, strict=True),
        key=lambda pair: (-float(pair[1]), pair[0].chunk_id),
    )
    return Reranking(hits=tuple(hit for hit, _ in ranked), model=chosen.model)
