"""Unit-level coverage for src/sources/gdrive_throttle.py.

AC-011 is proven in tests/test_spec_source_gdrive.py — see CLAUDE.md for why
the proof lives there. This file covers the budget arithmetic and the backoff,
with the clock and the sleeper injected: a test that waits a real minute to
prove the limiter waits is a test nobody runs.
"""

import pytest

from src.sources.gdrive_throttle import (
    BACKOFF_CAP_SECONDS,
    DEFAULT_UNITS_PER_MINUTE,
    UNITS_DOWNLOAD,
    UNITS_LIST,
    WINDOW_SECONDS,
    QuotaBudget,
    RateLimited,
    backoff_delays,
    with_retries,
)


class _Clock:
    """A clock a test moves by hand, and a sleeper that records instead of waiting."""

    def __init__(self) -> None:
        self.now_seconds = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.now_seconds

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now_seconds += seconds

    def budget(self, units_per_minute: int = DEFAULT_UNITS_PER_MINUTE) -> QuotaBudget:
        return QuotaBudget(units_per_minute=units_per_minute, now=self.now, sleep=self.sleep)


def test_the_measured_grant_puts_the_ceiling_where_the_research_found_it():
    # 325,000 units a minute over 200 per download is about twenty-seven a
    # second, and that number is what paces a first full sweep (ASM-013).
    assert QuotaBudget().downloads_per_second() == pytest.approx(27.08, abs=0.01)


def test_requests_are_priced_not_counted():
    # A listing and a download differ by two orders of magnitude; a limiter
    # that counted both as one would either throttle listings pointlessly or
    # let downloads blow through the grant.
    assert UNITS_DOWNLOAD == 200
    assert UNITS_LIST == 1
    assert UNITS_DOWNLOAD > UNITS_LIST * 100


def test_spending_within_the_grant_never_waits():
    clock = _Clock()
    budget = clock.budget(units_per_minute=1_000)

    for _ in range(5):
        assert budget.spend(UNITS_DOWNLOAD).waited_seconds == 0.0

    assert clock.slept == []
    assert budget.spent == 1_000


def test_a_spent_window_waits_for_what_is_left_of_it():
    clock = _Clock()
    budget = clock.budget(units_per_minute=400)
    budget.spend(UNITS_DOWNLOAD)
    budget.spend(UNITS_DOWNLOAD)
    clock.now_seconds = 15.0

    spend = budget.spend(UNITS_DOWNLOAD)

    # Forty-five seconds is what is left of the minute, not a whole one.
    assert spend.waited_seconds == pytest.approx(WINDOW_SECONDS - 15.0)
    assert clock.slept == [pytest.approx(45.0)]


def test_a_window_that_has_already_elapsed_refills_without_waiting():
    clock = _Clock()
    budget = clock.budget(units_per_minute=400)
    budget.spend(UNITS_DOWNLOAD)
    budget.spend(UNITS_DOWNLOAD)
    clock.now_seconds = WINDOW_SECONDS + 1

    assert budget.spend(UNITS_DOWNLOAD).waited_seconds == 0.0
    assert clock.slept == []
    assert budget.spent == UNITS_DOWNLOAD


def test_a_grant_too_small_for_one_download_is_refused_at_construction():
    # It would let a sweep start and never make progress, which reads as a
    # hung job rather than a misconfiguration.
    with pytest.raises(ValueError) as excinfo:
        QuotaBudget(units_per_minute=10)

    assert "no sweep could ever make progress" in str(excinfo.value)


def test_a_request_larger_than_the_whole_grant_is_refused():
    clock = _Clock()
    budget = clock.budget(units_per_minute=300)

    with pytest.raises(ValueError) as excinfo:
        budget.spend(500)

    assert "can never fit" in str(excinfo.value)


def test_a_request_has_to_cost_something():
    with pytest.raises(ValueError):
        _Clock().budget().spend(0)


def test_backoff_grows_and_is_capped():
    ceilings: list[float] = []
    list(backoff_delays(10, jitter=lambda ceiling: ceilings.append(ceiling) or 0.0))

    assert ceilings[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert max(ceilings) == BACKOFF_CAP_SECONDS
    # The cap matters more than the base: a sweep that pauses a minute is fine,
    # one that pauses an hour is a stalled job nobody notices.
    assert all(ceiling <= BACKOFF_CAP_SECONDS for ceiling in ceilings)


def test_backoff_is_full_jitter_so_synchronised_clients_do_not_retry_together():
    # Each delay is random in [0, ceiling), not the ceiling itself. Without it
    # every worker that hit the limit together retries together, and the retry
    # fails for the reason the first attempt did.
    delays = list(backoff_delays(6))

    assert len(delays) == 5
    assert all(0.0 <= delay <= BACKOFF_CAP_SECONDS for delay in delays)
    assert len(set(delays)) > 1, "identical delays are not jitter"


def test_one_attempt_means_no_delays():
    assert list(backoff_delays(1)) == []
    with pytest.raises(ValueError):
        list(backoff_delays(0))


def test_a_rate_refusal_is_retried_and_then_succeeds():
    clock = _Clock()
    budget = clock.budget()
    attempts = {"n": 0}

    class _Refused(Exception):
        pass

    def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Refused()
        return "done"

    result = with_retries(
        call,
        is_rate_limited=lambda error: isinstance(error, _Refused),
        budget=budget,
        units=UNITS_LIST,
        jitter=lambda ceiling: 0.0,
    )

    assert result == "done"
    assert attempts["n"] == 3
    # Each attempt spends its units, because the source charges for a refusal
    # too — a retry loop that spent nothing would understate the sweep's cost.
    assert budget.spent == 3 * UNITS_LIST


def test_exhausted_retries_raise_something_that_names_the_cause():
    clock = _Clock()

    class _Refused(Exception):
        pass

    with pytest.raises(RateLimited) as excinfo:
        with_retries(
            lambda: (_ for _ in ()).throw(_Refused()),
            is_rate_limited=lambda error: isinstance(error, _Refused),
            budget=clock.budget(),
            units=UNITS_LIST,
            max_attempts=3,
            jitter=lambda ceiling: 0.0,
        )

    message = str(excinfo.value)
    assert "3 attempts" in message
    # An operator's action here is to wait or spread the run, not to
    # investigate a document, so the message says which.
    assert "ASM-013" in message


def test_anything_that_is_not_a_rate_refusal_propagates_untouched():
    # Retrying a document that does not exist wastes the budget the documents
    # that do exist need, and hides the real error behind a delay.
    clock = _Clock()

    class _Missing(Exception):
        pass

    with pytest.raises(_Missing):
        with_retries(
            lambda: (_ for _ in ()).throw(_Missing()),
            is_rate_limited=lambda error: False,
            budget=clock.budget(),
            units=UNITS_LIST,
        )

    assert clock.slept == []
