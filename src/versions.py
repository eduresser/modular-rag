from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import psycopg

from src.config import PROCESS_DOCAI, PROCESS_PDFTEXT, AppConfig
from src.fingerprint import STAGE_CUT, STAGE_EMBED, STAGE_READ, STAGES
from src.processing.pdftext import extractor_version
from src.processing.recut import cut_version

DECLARED_VERSIONS: dict[str, tuple[str, ...]] = {
    STAGE_READ: ("pretrained-layout-parser-v1.0-2024-06-03",),
    STAGE_CUT: ("1", "2", "3"),
    STAGE_EMBED: (),
}


class UnknownStageVersion(Exception):
    pass


@dataclass(frozen=True, slots=True)
class StageVersions:
    read: str
    cut: str
    embed: str

    def __getitem__(self, stage: str) -> str:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
        return {STAGE_READ: self.read, STAGE_CUT: self.cut, STAGE_EMBED: self.embed}[stage]


READ_VERSIONS: dict[str, Callable[[AppConfig], str]] = {
    PROCESS_DOCAI: lambda config: config.read.processor_version,
    PROCESS_PDFTEXT: lambda _config: extractor_version(),
}


def read_version(config: AppConfig) -> str:
    try:
        named = READ_VERSIONS[config.process.type]
    except KeyError:  # pragma: no cover — the configuration is a closed Literal
        raise KeyError(
            f"process module {config.process.type!r} names no read version; add it to "
            "READ_VERSIONS, because a piece that cannot say what read it cannot be "
            "re-read when the reader moves (AC-036)"
        ) from None
    return named(config)


def configured_versions(config: AppConfig) -> StageVersions:
    return StageVersions(
        read=read_version(config),
        cut=cut_version(config.process.type),
        embed=config.embed.model,
    )


def available(stage: str, config: AppConfig) -> tuple[str, ...]:
    if stage not in STAGES:
        raise UnknownStageVersion(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
    declared = (*DECLARED_VERSIONS[stage], *(config.embed.understood_models if stage == STAGE_EMBED else ()))
    return tuple(sorted({*declared, configured_versions(config)[stage]}))


def resolve(stage: str, recorded: str, config: AppConfig) -> str:
    known = available(stage, config)
    if recorded in known:
        return recorded
    raise UnknownStageVersion(
        f"the {stage} stage of this document was version {recorded!r}, which this build no "
        f"longer declares; the available versions are {', '.join(known)}. Nothing is "
        "reprocessed under the newest one on its own: that would leave a corpus made by "
        "two different things with no column saying which piece came from which"
    )


def recorded_versions(conn: psycopg.Connection, document_id: int) -> dict[str, str | None]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT reader_version, cut_version FROM document_ingest WHERE document_id = %s",
            (document_id,),
        )
        row = cursor.fetchone()
        cursor.execute(
            """
            SELECT embed_model
              FROM chunks
             WHERE document_id = %s
             ORDER BY ordinal
             LIMIT 1
            """,
            (document_id,),
        )
        stamped = cursor.fetchone()
    return {
        STAGE_READ: row[0] if row else None,
        STAGE_CUT: row[1] if row else None,
        STAGE_EMBED: stamped[0] if stamped else None,
    }


def unresolvable(conn: psycopg.Connection, config: AppConfig) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT reader_version, cut_version FROM document_ingest")
        for reader_version, cut_version in cursor.fetchall():
            found.append((STAGE_READ, reader_version))
            found.append((STAGE_CUT, cut_version))
        cursor.execute("SELECT DISTINCT embed_model FROM chunks")
        found.extend((STAGE_EMBED, model) for (model,) in cursor.fetchall())
    return tuple(
        sorted({(stage, version) for stage, version in found if version not in available(stage, config)})
    )
