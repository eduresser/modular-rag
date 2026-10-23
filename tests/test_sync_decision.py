
import psycopg

from src.sync_decision import (
    REASON_NO_BASELINE,
    REASON_NO_POSITION,
    REASON_POSITION_HOLDS,
    Decision,
    decide_next_run,
    sweep_because,
)
from src.sync_ledger import (
    RUN_KIND_SWEEP,
    RUN_STATUS_COMPLETED,
    advance_position,
    finish_run,
    start_run,
)
from tests.conftest import TEST_CLIENT

IDENTITY = "rag@example.iam.gserviceaccount.com"
SCOPE = "watched-folder"


class RecordingResume:

    def __init__(self, objection: Decision | None = None) -> None:
        self._objection = objection
        self.asked: list[tuple[str, str, str]] = []

    @property
    def origin(self) -> tuple[str, str]:
        return (IDENTITY, SCOPE)

    def objection(self, position: str, issued_by: str, scope: str) -> Decision | None:
        self.asked.append((position, issued_by, scope))
        return self._objection


def _baseline(conn: psycopg.Connection) -> None:
    run_id = start_run(conn, client=TEST_CLIENT, kind=RUN_KIND_SWEEP)
    finish_run(conn, run_id, status=RUN_STATUS_COMPLETED, sweep_completed=True)


def test_the_source_is_not_consulted_until_the_generic_questions_pass(conn: psycopg.Connection):
    resume = RecordingResume()

    assert decide_next_run(conn, resume).reason == REASON_NO_BASELINE
    assert resume.asked == []

    _baseline(conn)
    assert decide_next_run(conn, resume).reason == REASON_NO_POSITION
    assert resume.asked == []

    advance_position(conn, position="3")
    assert decide_next_run(conn, resume).reason == REASON_NO_POSITION
    assert resume.asked == []

    advance_position(conn, position="3", issued_by=IDENTITY, scope=SCOPE)
    assert decide_next_run(conn, resume).reason == REASON_POSITION_HOLDS
    assert resume.asked == [("3", IDENTITY, SCOPE)]


def test_the_module_can_refuse_for_a_reason_the_shared_layer_never_names(
    conn: psycopg.Connection,
):
    invented = sweep_because("cursor_expired", "this provider's cursor lasts thirty days")
    decision = decide_next_run(conn, RecordingResume(objection=invented))
    assert decision.reason == REASON_NO_BASELINE

    _baseline(conn)
    advance_position(conn, position="3", issued_by=IDENTITY, scope=SCOPE)
    decision = decide_next_run(conn, RecordingResume(objection=invented))
    assert decision.kind == RUN_KIND_SWEEP
    assert decision.reason == "cursor_expired"
    assert decision.detail == "this provider's cursor lasts thirty days"
    assert decision.is_delta is False


def test_recording_a_position_without_an_origin_never_erases_one(conn: psycopg.Connection):
    _baseline(conn)
    advance_position(conn, position="3", issued_by=IDENTITY, scope=SCOPE)
    advance_position(conn, position="4")
    resume = RecordingResume()
    assert decide_next_run(conn, resume).reason == REASON_POSITION_HOLDS
    assert resume.asked == [("4", IDENTITY, SCOPE)]
