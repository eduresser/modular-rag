from __future__ import annotations

from datetime import datetime

from croniter import (
    CroniterBadCronError,
    CroniterBadDateError,
    CroniterBadTypeRangeError,
    CroniterNotAlphaError,
    croniter,
)

CRON_FIELD_COUNT = 5

_CRONITER_ERRORS = (
    CroniterBadCronError,
    CroniterBadDateError,
    CroniterBadTypeRangeError,
    CroniterNotAlphaError,
)


class InvalidScheduleError(Exception):
    pass


def next_fire(expression: str, *, now: datetime) -> datetime:
    fields = expression.split()
    if len(fields) != CRON_FIELD_COUNT:
        raise InvalidScheduleError(
            f"{expression!r} is not a five-field cron expression (minute hour "
            f"day-of-month month day-of-week); it has {len(fields)} field(s)"
        )
    try:
        return croniter(expression, now).get_next(datetime)
    except _CRONITER_ERRORS as exc:
        raise InvalidScheduleError(f"{expression!r} is not a valid cron expression: {exc}") from exc
