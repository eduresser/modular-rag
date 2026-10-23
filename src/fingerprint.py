from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.config import AppConfig
from src.processing.pdftext import extractor_version
from src.processing.recut import cut_version
from src.processing.render import renderer_version
from src.sources.gdrive_export import KIND_DOCUMENT, export_plan

STAGE_READ = "read"
STAGE_CUT = "cut"
STAGE_EMBED = "embed"

STAGES: tuple[str, ...] = (STAGE_READ, STAGE_CUT, STAGE_EMBED)


@dataclass(frozen=True, slots=True)
class StageFingerprints:
    read: str
    cut: str
    embed: str
    whole: str

    def __getitem__(self, stage: str) -> str:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
        return {STAGE_READ: self.read, STAGE_CUT: self.cut, STAGE_EMBED: self.embed}[stage]


READER_SETTINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "docai": (
        ("docai.processor_version", "processor_version"),
        ("docai.enable_table_annotation", "enable_table_annotation"),
        ("docai.enable_image_annotation", "enable_image_annotation"),
    ),
    "pdftext": (),
}

CUTTER_SETTINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "docai": (
        ("chunk.size", "size"),
        ("chunk.include_headings", "include_headings"),
    ),
    # A page is what the file supplies, so there is nothing to set and nothing
    # to hash. The empty tuple is the decision; an absent key raises.
    "pdftext": (),
}


def _module_settings(
    groups: dict[str, tuple[tuple[str, str], ...]],
    section: Any,
    module: str,
    *,
    what: str,
    group: str,
) -> dict[str, Any]:
    try:
        named = groups[module]
    except KeyError:  # pragma: no cover — the configuration is a closed Literal
        raise KeyError(
            f"{what} {module!r} declares no write-time settings; add it to {group}, with "
            "an empty tuple if it genuinely has none"
        ) from None
    return {key: getattr(section, field) for key, field in named}


MEASURED_READ_VERSIONS: dict[str, tuple[str, Callable[[], str]]] = {
    "pdftext": ("pdftext.extractor_version", extractor_version),
}


def _measured_read_versions(module: str) -> dict[str, Any]:
    named = MEASURED_READ_VERSIONS.get(module)
    return {named[0]: named[1]()} if named else {}


def stage_settings(config: AppConfig) -> dict[str, dict[str, Any]]:
    return {
        STAGE_READ: {
            **_module_settings(
                READER_SETTINGS,
                config.read,
                config.process.type,
                what="reader",
                group="READER_SETTINGS",
            ),
            "gdrive_export.document_export_target": export_plan(KIND_DOCUMENT).media_type,
            "render.renderer_version": renderer_version(),
            **_measured_read_versions(config.process.type),
        },
        STAGE_CUT: {
            "chunk.type": config.process.type,
            "parse.version": cut_version(config.process.type),
            **_module_settings(
                CUTTER_SETTINGS,
                config.chunk,
                config.process.type,
                what="cutter",
                group="CUTTER_SETTINGS",
            ),
        },
        STAGE_EMBED: {
            "embed.model": config.embed.model,
            "embed.dimension": config.embed.dimension,
            "embed.extra_params": config.embed.request.body,
            "embed.prompts.document": config.embed.prompts.document,
        },
    }


def write_time_settings(config: AppConfig) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for stage in STAGES:
        flat.update(stage_settings(config)[stage])
    return flat


def compute_fingerprint(config: AppConfig) -> str:
    return digest_of(write_time_settings(config))


def stage_fingerprints(config: AppConfig) -> StageFingerprints:
    grouped = stage_settings(config)
    return StageFingerprints(
        read=digest_of(grouped[STAGE_READ]),
        cut=digest_of(grouped[STAGE_CUT]),
        embed=digest_of(grouped[STAGE_EMBED]),
        whole=compute_fingerprint(config),
    )


def digest_of(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
