from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from src.config import AppConfig
from src.fingerprint import STAGES, stage_fingerprints, stage_settings, write_time_settings
from src.reprocess import documents_not_at, recorded_fingerprints


@dataclass(frozen=True, slots=True)
class SettingChange:
    setting: str
    recorded: Any
    configured: Any
    stage: str

    def __str__(self) -> str:
        return f"{self.setting}: {self.recorded!r} → {self.configured!r}"


@dataclass(frozen=True, slots=True)
class StalenessReport:
    current: bool
    fingerprint: str
    affected: int
    changes: tuple[SettingChange, ...] = ()
    stale_fingerprints: tuple[str, ...] = ()
    unrecorded_fingerprints: tuple[str, ...] = ()

    @property
    def stages(self) -> tuple[str, ...]:
        stale = {change.stage for change in self.changes}
        return tuple(stage for stage in STAGES if stage in stale)

    @property
    def settings(self) -> tuple[str, ...]:
        return tuple(change.setting for change in self.changes)

    def summary(self) -> str:
        if self.current:
            return (
                f"the archive is current: every indexed document was produced under "
                f"{self.fingerprint[:12]}"
            )
        changed = "; ".join(str(change) for change in self.changes) or "settings not recorded"
        return (
            f"the archive is stale: {self.affected} document(s) were produced under an "
            f"indexing configuration that is no longer the one configured ({changed})"
        )


def record_generation(conn: psycopg.Connection, config: AppConfig) -> str:
    fingerprint = stage_fingerprints(config).whole
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO indexing_generations (fingerprint, settings)
                 VALUES (%s, %s)
            ON CONFLICT (fingerprint) DO NOTHING
            """,
            (fingerprint, Jsonb(write_time_settings(config))),
        )
    return fingerprint


def _oldest_first(conn: psycopg.Connection, fingerprints: tuple[str, ...]) -> tuple[str, ...]:
    if not fingerprints:
        return ()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT fingerprint
              FROM indexing_generations
             WHERE fingerprint = ANY(%s)
             ORDER BY first_seen_at, fingerprint
            """,
            (list(fingerprints),),
        )
        known = [row[0] for row in cursor.fetchall()]
    return tuple(known) + tuple(sorted(set(fingerprints) - set(known)))


def recorded_settings(conn: psycopg.Connection, fingerprint: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT settings FROM indexing_generations WHERE fingerprint = %s", (fingerprint,)
        )
        row = cursor.fetchone()
    return dict(row[0]) if row is not None else None


def archive_staleness(conn: psycopg.Connection, config: AppConfig) -> StalenessReport:
    fingerprints = stage_fingerprints(config)
    stale = _oldest_first(
        conn,
        tuple(
            recorded for recorded in recorded_fingerprints(conn) if recorded != fingerprints.whole
        ),
    )
    affected = documents_not_at(conn, fingerprints.whole)
    if not stale:
        return StalenessReport(current=True, fingerprint=fingerprints.whole, affected=affected)
    changes: dict[str, SettingChange] = {}
    unrecorded: list[str] = []
    configured = stage_settings(config)
    stage_of = {setting: stage for stage in STAGES for setting in configured[stage]}
    for recorded_fingerprint in stale:
        settings = recorded_settings(conn, recorded_fingerprint)
        if settings is None:
            unrecorded.append(recorded_fingerprint)
            continue
        for setting, value in write_time_settings(config).items():
            was = settings.get(setting)
            if was != value and setting not in changes:
                changes[setting] = SettingChange(
                    setting=setting,
                    recorded=was,
                    configured=value,
                    stage=stage_of[setting],
                )
    return StalenessReport(
        current=False,
        fingerprint=fingerprints.whole,
        affected=affected,
        changes=tuple(changes[setting] for setting in write_time_settings(config) if setting in changes),
        stale_fingerprints=stale,
        unrecorded_fingerprints=tuple(unrecorded),
    )
