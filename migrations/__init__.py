from __future__ import annotations

from importlib import import_module
from types import ModuleType

# Oldest first; append new migrations, never insert
ORDER: tuple[str, ...] = (
    "001_baseline",
    "002_sync_runs",
    "003_usage",
    "004_search_events",
    "005_processing",
    "006_ingest",
    "007_sync_window",
    "008_sync_history",
    "009_position_identity",
    "010_unproven_counter",
    "011_document_record",
    "012_sync_requests",
    "013_artifacts",
    "014_process_module",
    "015_retired_counter",
    "016_filters",
)


def load(name: str) -> ModuleType:
    if name not in ORDER:
        raise ValueError(f"unknown migration {name!r}; known migrations are {', '.join(ORDER)}")
    return import_module(f"{__name__}.{name}")
