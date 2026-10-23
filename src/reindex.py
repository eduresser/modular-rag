from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import psycopg

from src.config import AppConfig
from src.embedding.cost import provider_of
from src.fingerprint import stage_fingerprints
from src.ingest import IngestRun, Pipeline, ingest_documents
from src.prices import (
    DEFAULT_CURRENCY,
    OPERATION_EMBED,
    OPERATION_READ,
    OPERATION_CHUNK,
    UNIT_PAGE,
    UNIT_TOKEN,
    PriceBook,
)
from src.processing.cost import MODEL_LOCAL_RECUT, PROVIDER_LOCAL, PROVIDER_READER
from src.reprocess import (
    ACTION_READ,
    ACTION_RECUT,
    ACTION_REEMBED,
    ACTION_TOUCH,
    IngestState,
    decide,
    stale_states,
)
from src.staleness import StalenessReport, archive_staleness

CHARACTERS_PER_TOKEN = 4


class ReindexError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    action: str
    documents: int
    pages: int
    tokens: int
    cost: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class ReindexEstimate:
    documents: int
    cost: Decimal
    by_action: tuple[ActionEstimate, ...] = ()
    currency: str = DEFAULT_CURRENCY

    @property
    def pages(self) -> int:
        return sum(figure.pages for figure in self.by_action)

    @property
    def tokens(self) -> int:
        return sum(figure.tokens for figure in self.by_action)

    def for_action(self, action: str) -> ActionEstimate | None:
        for figure in self.by_action:
            if figure.action == action:
                return figure
        return None

    def summary(self) -> str:
        if not self.documents:
            return "nothing to reprocess: every indexed document matches the configuration"
        breakdown = ", ".join(
            f"{figure.documents} to {figure.action}" for figure in self.by_action if figure.documents
        )
        return (
            f"{self.documents} document(s) would be reprocessed ({breakdown}), "
            f"estimated {self.cost} {self.currency}"
        )


@dataclass(frozen=True, slots=True)
class ReindexPlan:
    staleness: StalenessReport
    estimate: ReindexEstimate
    documents: tuple[int, ...] = ()
    started: bool = False


@dataclass(frozen=True, slots=True)
class ReindexOutcome:
    plan: ReindexPlan
    run: IngestRun
    started: bool = True


def plan_reindex(
    conn: psycopg.Connection,
    config: AppConfig,
    book: PriceBook,
    *,
    document_ids: Sequence[int] | None = None,
) -> ReindexPlan:
    staleness = archive_staleness(conn, config)
    states = stale_states(conn, stage_fingerprints(config).whole)
    if document_ids is not None:
        wanted = set(document_ids)
        states = tuple(state for state in states if state.document_id in wanted)
    estimate = estimate_reindex(conn, config, book, states=states)
    return ReindexPlan(
        staleness=staleness,
        estimate=estimate,
        documents=tuple(state.document_id for state in states),
    )


def estimate_reindex(
    conn: psycopg.Connection,
    config: AppConfig,
    book: PriceBook,
    *,
    states: Sequence[IngestState],
) -> ReindexEstimate:
    fingerprints = stage_fingerprints(config)
    pages = _pages_by_document(conn)
    characters = _characters_by_document(conn)
    grouped: dict[str, list[IngestState]] = {}
    for state in states:
        decision = decide(
            stored=state, content_digest=state.content_digest, fingerprints=fingerprints
        )
        if decision.action == ACTION_TOUCH:
            continue
        grouped.setdefault(decision.action, []).append(state)
    figures: list[ActionEstimate] = []
    total = Decimal(0)
    provider = provider_of(config.embed.base_url)
    for action in (ACTION_READ, ACTION_RECUT, ACTION_REEMBED):
        members = grouped.get(action, [])
        if not members:
            continue
        page_count = sum(pages.get(state.document_id, 0) for state in members)
        token_count = sum(
            _tokens_of(characters.get(state.document_id, 0)) for state in members
        )
        cost = Decimal(0)
        if action == ACTION_READ:
            cost += _priced(
                book, units=page_count, operation=OPERATION_READ,
                provider=PROVIDER_READER, model=config.read.processor_version, unit=UNIT_PAGE,
            )
        if action == ACTION_RECUT:
            cost += _priced(
                book, units=page_count, operation=OPERATION_CHUNK,
                provider=PROVIDER_LOCAL, model=MODEL_LOCAL_RECUT, unit=UNIT_PAGE,
            )
        cost += _priced(
            book, units=token_count, operation=OPERATION_EMBED,
            provider=provider, model=config.embed.model, unit=UNIT_TOKEN,
        )
        figures.append(
            ActionEstimate(
                action=action,
                documents=len(members),
                pages=page_count if action != ACTION_REEMBED else 0,
                tokens=token_count,
                cost=cost,
            )
        )
        total += cost
    return ReindexEstimate(
        documents=sum(figure.documents for figure in figures),
        cost=total,
        by_action=tuple(figures),
    )


def reindex(
    conn: psycopg.Connection,
    pipeline: Pipeline,
    book: PriceBook,
    *,
    confirm: bool = False,
    document_ids: Sequence[int] | None = None,
) -> ReindexPlan | ReindexOutcome:
    plan = plan_reindex(conn, pipeline.config, book, document_ids=document_ids)
    if not confirm:
        return plan
    if not plan.documents:
        raise ReindexError(
            "there is nothing to reprocess: every indexed document already matches the "
            "configuration. Confirming a reprocessing of nothing would write a run record "
            "saying work happened when none did"
        )
    documents = _documents_of(conn, pipeline, plan.documents)
    run = ingest_documents(conn, pipeline, documents, operator_requested=True)
    return ReindexOutcome(
        plan=ReindexPlan(
            staleness=plan.staleness,
            estimate=plan.estimate,
            documents=plan.documents,
            started=True,
        ),
        run=run,
    )


def _priced(
    book: PriceBook,
    *,
    units: int,
    operation: str,
    provider: str,
    model: str,
    unit: str,
) -> Decimal:
    price = book.price_or_none(operation=operation, provider=provider, model=model, unit=unit)
    return price.cost_of(units) if price is not None else Decimal(0)


def _tokens_of(characters: int) -> int:
    return -(-characters // CHARACTERS_PER_TOKEN) if characters else 0


def _pages_by_document(conn: psycopg.Connection) -> Mapping[int, int]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT document_id, COALESCE(page_count, 0) FROM document_reads WHERE state = 'read'"
        )
        return {int(row[0]): int(row[1]) for row in cursor.fetchall()}


def _characters_by_document(conn: psycopg.Connection) -> Mapping[int, int]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, sum(length(content))
              FROM chunks
             GROUP BY document_id
            """
        )
        return {int(row[0]): int(row[1] or 0) for row in cursor.fetchall()}


def _documents_of(
    conn: psycopg.Connection, pipeline: Pipeline, document_ids: Sequence[int]
):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT external_id FROM documents WHERE id = ANY(%s) ORDER BY id",
            (list(document_ids),),
        )
        identifiers = [row[0] for row in cursor.fetchall()]
    found = []
    for external_id in identifiers:
        try:
            found.append(pipeline.source.probe(external_id))
        except Exception:  # noqa: BLE001 — a document that is gone is not reprocessed
            continue
    return tuple(found)
