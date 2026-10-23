from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

DEFAULT_UNITS_PER_MINUTE = 325_000

UNITS_LIST = 1
UNITS_GET = 1
UNITS_DOWNLOAD = 200

WINDOW_SECONDS = 60.0

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 64.0
DEFAULT_MAX_ATTEMPTS = 6


class RateLimited(Exception):
    ...


@dataclass(frozen=True, slots=True)
class Spend:

    units: int
    waited_seconds: float


@dataclass
class QuotaBudget:

    units_per_minute: int = DEFAULT_UNITS_PER_MINUTE
    now: Callable[[], float] = field(default=lambda: __import__("time").monotonic())
    sleep: Callable[[float], None] = field(default=lambda seconds: __import__("time").sleep(seconds))
    _window_started: float | None = field(default=None, init=False, repr=False)
    _spent: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.units_per_minute < UNITS_DOWNLOAD:
            raise ValueError(
                f"a budget of {self.units_per_minute} units a minute cannot afford a single "
                f"download at {UNITS_DOWNLOAD} units, so no sweep could ever make progress"
            )

    @property
    def spent(self) -> int:
        return self._spent

    def spend(self, units: int) -> Spend:
        if units < 1:
            raise ValueError(f"a request costs at least one unit, got {units}")
        if units > self.units_per_minute:
            raise ValueError(
                f"a request costing {units} units can never fit a budget of "
                f"{self.units_per_minute} a minute"
            )
        moment = self.now()
        if self._window_started is None:
            self._window_started = moment
        elapsed = moment - self._window_started
        if elapsed >= WINDOW_SECONDS:
            self._window_started = moment
            self._spent = 0
            elapsed = 0.0
        waited = 0.0
        if self._spent + units > self.units_per_minute:
            waited = max(0.0, WINDOW_SECONDS - elapsed)
            if waited:
                self.sleep(waited)
            self._window_started = self.now()
            self._spent = 0
        self._spent += units
        return Spend(units=units, waited_seconds=waited)

    def downloads_per_second(self) -> float:
        return self.units_per_minute / UNITS_DOWNLOAD / WINDOW_SECONDS


def backoff_delays(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    *,
    base: float = BACKOFF_BASE_SECONDS,
    cap: float = BACKOFF_CAP_SECONDS,
    jitter: Callable[[float], float] | None = None,
) -> Iterator[float]:
    if max_attempts < 1:
        raise ValueError(f"there has to be at least one attempt, got {max_attempts}")
    spread = jitter or (lambda ceiling: random.uniform(0.0, ceiling))
    for attempt in range(max_attempts - 1):
        yield spread(min(cap, base * (2**attempt)))


def with_retries(
    call: Callable[[], object],
    *,
    is_rate_limited: Callable[[Exception], bool],
    budget: QuotaBudget,
    units: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    jitter: Callable[[float], float] | None = None,
) -> object:
    delays = list(backoff_delays(max_attempts, jitter=jitter))
    last: Exception | None = None
    for attempt in range(max_attempts):
        budget.spend(units)
        try:
            return call()
        except Exception as exc:
            if not is_rate_limited(exc):
                raise
            last = exc
            if attempt < len(delays):
                budget.sleep(delays[attempt])
    raise RateLimited(
        f"the source kept refusing for rate reasons after {max_attempts} attempts; "
        f"the budget allows about {budget.downloads_per_second():.0f} downloads a second "
        f"for one identity (ASM-013), so a sweep is paced by the source, not by us"
    ) from last
