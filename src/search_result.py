from __future__ import annotations

from dataclasses import dataclass, field

DENSE_LEG = "dense"
LEXICAL_LEG = "lexical"


@dataclass(frozen=True, slots=True)
class Provenance:
    document_id: int
    external_id: str
    title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    ordinal: int | None = None
    link: str | None = None
    location: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.external_id:
            raise ValueError("a citable piece must name the document it came from")
        if self.page_start is None and self.page_end is None and not self.heading_path:
            raise ValueError(
                f"piece from document {self.external_id!r} carries no locator — "
                "a page range or a heading path is what makes it citable (principle P-002)"
            )
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError(f"page range {self.page_start}..{self.page_end} runs backwards")

    def citation(self) -> str:
        parts = [self.title or self.external_id]
        if self.heading_path:
            parts.append(" › ".join(self.heading_path))
        pages = self.page_label()
        if pages:
            parts.append(pages)
        return ", ".join(parts)

    def page_label(self) -> str:
        if self.page_start is None and self.page_end is None:
            return ""
        start = self.page_start if self.page_start is not None else self.page_end
        end = self.page_end if self.page_end is not None else self.page_start
        return f"p. {start}" if start == end else f"pp. {start}-{end}"


@dataclass(frozen=True, slots=True)
class Candidate:
    chunk_id: int
    content: str
    provenance: Provenance
    rank: int
    leg_score: float

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"a candidate's rank is 1-based, got {self.rank}")


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: int
    content: str
    provenance: Provenance
    score: float
    found_by: tuple[str, ...]
    truncated: bool = False
    full_length: int | None = None

    def __post_init__(self) -> None:
        if not self.found_by:
            raise ValueError("a hit must name the leg or legs that found it")


@dataclass(frozen=True, slots=True)
class FusionWeights:
    dense: float = 0.5
    lexical: float = 0.5

    def __post_init__(self) -> None:
        if self.dense < 0 or self.lexical < 0:
            raise ValueError(f"fusion weights cannot be negative, got {self.dense} and {self.lexical}")
        if self.dense == 0 and self.lexical == 0:
            raise ValueError("at least one retrieval leg has to count for something")


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    index_search_effort: int
    weights: FusionWeights
    candidate_depth: int
    result_count: int
    dense_candidates: int = 0
    lexical_candidates: int = 0
    duration_ms: float = 0.0
    offset: int = 0
    documents_considered: int | None = None
    pieces_considered: int | None = None


@dataclass(frozen=True, slots=True)
class Found:
    hits: tuple[SearchHit, ...]
    diagnostics: SearchDiagnostics
    succeeded: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.hits:
            raise ValueError("a Found outcome with no hits is a NothingMatched outcome")


@dataclass(frozen=True, slots=True)
class NothingMatched:
    diagnostics: SearchDiagnostics
    hits: tuple[SearchHit, ...] = ()
    succeeded: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class FilterMatchedNothing:
    diagnostics: SearchDiagnostics
    clause: str
    hits: tuple[SearchHit, ...] = ()
    succeeded: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.clause.strip():
            raise ValueError("a filter-matched-nothing outcome has to name the clause responsible")


@dataclass(frozen=True, slots=True)
class Failed:
    reason: str
    diagnostics: SearchDiagnostics | None = None
    succeeded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a failure has to say why — that is the whole point of AC-077")


SearchOutcome = Found | NothingMatched | FilterMatchedNothing | Failed
