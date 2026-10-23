"""Edge cases for the process entrypoint (T-183), not themselves an acceptance
criterion — AC-176 is proven in tests/test_spec_worker_entrypoint.py.
"""

from __future__ import annotations

import signal

import pytest

from src.__main__ import ROLE_SERVE, ROLE_WORKER, ROLES, _listen, _signal_stop_flag, main
from src.boot import UnknownRole, check_role


def test_the_roles_are_exactly_serve_and_worker():
    assert set(ROLES) == {"serve", "worker"}
    assert ROLE_SERVE == "serve"
    assert ROLE_WORKER == "worker"


def test_check_role_accepts_every_declared_role():
    for role in ROLES:
        assert check_role(role) == role


def test_check_role_refuses_and_names_what_exists():
    with pytest.raises(UnknownRole) as excinfo:
        check_role("frobnicate")
    message = str(excinfo.value)
    assert "frobnicate" in message
    assert "serve" in message and "worker" in message


def test_main_with_no_argument_refuses_rather_than_guessing(capsys):
    exit_code = main([])
    assert exit_code != 0
    assert "serve" in capsys.readouterr().err


def test_listen_reads_the_platform_port_convention(monkeypatch):
    """`$PORT` is what a container platform sets; a deployment names no port."""
    calls: list[dict] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs})
    )
    monkeypatch.setenv("PORT", "9001")

    _listen(app="the-app", config=None)

    assert calls == [{"app": "the-app", "host": "0.0.0.0", "port": 9001}]


def test_listen_defaults_to_8080_with_no_port_set(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda app, **kwargs: calls.append({"app": app, **kwargs})
    )
    monkeypatch.delenv("PORT", raising=False)

    _listen(app="the-app", config=None)

    assert calls[0]["port"] == 8080


def test_the_signal_flag_is_set_by_a_termination_signal_and_nothing_else():
    """ASM-094: the handler only sets the flag; finishing cleanly is run_schedule's job."""
    import os

    previous = signal.getsignal(signal.SIGTERM)
    try:
        stopped = _signal_stop_flag()
        assert stopped() is False

        os.kill(os.getpid(), signal.SIGTERM)

        assert stopped() is True
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_modular_rag_is_declared_as_a_named_command(pyproject):
    assert pyproject["project"]["scripts"]["modular-rag"] == "src.__main__:main"


@pytest.fixture
def pyproject():
    import pathlib

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11, not this project's floor, but cheap to allow
        import tomli as tomllib  # type: ignore[no-redef]

    location = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    return tomllib.loads(location.read_text())
