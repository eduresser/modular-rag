"""The health route: answers without asking the archive anything.

**It touches nothing** (AC-138). No connection to the database, no call to the
embedding provider, no read of the active model version. The reason is
circular and real: an orchestrator restarts a container when this route stops
answering, and a health check that queries the archive puts load on the archive
exactly when the archive is struggling — so the check itself becomes a cause of
the restarts it is supposed to detect.

**It reports whether the startup checks passed.** The gate runs once, at
startup, and its verdict is kept in the application's state; this route reads
that verdict. A deployment whose configuration is wrong never starts at all
(that is T-130's job), so what this can report as failed is the other kind: the
archive was unreachable when the service came up, which is a condition that
resolves itself without a redeploy and which an operator needs to see.

**It is not behind the credential.** An orchestrator has no token, and there is
nothing here to protect: the payload names no document, no count and no title —
only whether this process considers itself ready.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


@dataclass(frozen=True, slots=True)
class StartupChecks:
    """What the startup gate concluded, kept for this route to report.

    ``detail`` is empty when everything passed. When it is not, it is the
    reason, in the same words the gate produced — an operator reading it should
    not have to correlate it with a log line to know what happened.
    """

    passed: bool
    detail: str = ""

    @property
    def status(self) -> str:
        return "ok" if self.passed else "degraded"


def checks_of(request: Request) -> StartupChecks:
    """The verdict this process recorded at startup.

    A process that somehow has none is reported as not passing rather than as
    passing: "nobody checked" and "the checks passed" are different, and only
    one of them is safe to serve traffic on.
    """
    recorded = getattr(request.app.state, "startup_checks", None)
    if isinstance(recorded, StartupChecks):
        return recorded
    return StartupChecks(passed=False, detail="the startup checks left no verdict")


@router.get("/healthz")
def health(request: Request, response: Response) -> dict[str, object]:
    """Whether this process is serving, and whether its startup checks passed.

    A degraded process answers 503, so an orchestrator does not need to read the
    body to route around it — and the body still says why, so a person does not
    need to read the orchestrator.
    """
    checks = checks_of(request)
    if not checks.passed:
        response.status_code = 503
    return {
        "status": checks.status,
        "checks_passed": checks.passed,
        "detail": checks.detail,
    }
