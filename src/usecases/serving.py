from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig
from src.db import connect
from src.embedding.stamp import stamp_from_config
from src.ingest_idempotency import APP_VERSION
from src.store_versions import NoActiveVersion, active_version


@dataclass(frozen=True, slots=True)
class ServingUnavailable:
    error: str = "What is serving could not be read; try again later."


@dataclass(frozen=True, slots=True)
class Serving:

    configured_model: str
    configured_dimension: int
    active_model: str | None
    active_dimension: int | None
    app_version: str

    @property
    def agrees(self) -> bool:
        if self.active_model is None:
            return True
        return (
            self.active_model == self.configured_model
            and self.active_dimension == self.configured_dimension
        )

    def payload(self) -> dict[str, object]:
        return {
            "configured": {
                "embed_model": self.configured_model,
                "embed_dimension": self.configured_dimension,
            },
            "active": None
            if self.active_model is None
            else {
                "embed_model": self.active_model,
                "embed_dimension": self.active_dimension,
            },
            "agrees": self.agrees,
            "app_version": self.app_version,
        }


def serving(config: AppConfig) -> Serving | ServingUnavailable:
    configured = stamp_from_config(config)
    try:
        with connect(config.db.url) as conn:
            try:
                active = active_version(conn)
            except NoActiveVersion:
                active = None
    except Exception:
        return ServingUnavailable()
    return Serving(
        configured_model=configured.model,
        configured_dimension=configured.dimension,
        active_model=None if active is None else active.model,
        active_dimension=None if active is None else active.dimension,
        app_version=APP_VERSION,
    )
