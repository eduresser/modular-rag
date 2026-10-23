
def test_ac_176(monkeypatch, schema_dsn, caplog, capsys):
    """AC-176: Each role is one command, and a role nobody defined refuses to start @spec:AC-176"""
    import json
    import logging

    import httpx
    from conftest import EMBED_TOKEN, api_config_path, fake_resolver

    from src.__main__ import main, main_serve, main_worker

    def _key_json() -> str:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        pem = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            .private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            .decode()
        )
        return json.dumps(
            {
                "type": "service_account",
                "project_id": "modular-rag",
                "private_key_id": "kid-1",
                "private_key": pem,
                "client_email": "rag-ingest@modular-rag.iam.gserviceaccount.com",
                "client_id": "1",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    def _refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call before anything should touch it: {request.url}")

    def _refusing_http() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(_refuse))

    location = api_config_path(monkeypatch, schema_dsn)

    captured: dict = {}

    with caplog.at_level(logging.INFO):
        main_serve(
            resolver=fake_resolver,
            http=_refusing_http(),
            path=str(location),
            serve=lambda app, config: captured.update(app=app, config=config),
        )

    assert captured["config"].telemetry.client == "example"
    reported = " | ".join(record.getMessage() for record in caplog.records)
    assert "'serve'" in reported and "'example'" in reported

    caplog.clear()

    def _worker_resolver(reference: str) -> str:
        return _key_json() if "GOOGLE" in reference else EMBED_TOKEN

    with caplog.at_level(logging.INFO):
        main_worker(
            resolver=_worker_resolver,
            http=_refusing_http(),
            path=str(location),
            should_stop=lambda: True,
        )

    reported = " | ".join(record.getMessage() for record in caplog.records)
    assert "'worker'" in reported and "'example'" in reported
    assert "runs nothing on its own" in reported

    exit_code = main(["frobnicate"])
    assert exit_code != 0
    refusal = capsys.readouterr().err
    assert "frobnicate" in refusal
    assert "serve" in refusal and "worker" in refusal

def test_ac_177(monkeypatch, schema_dsn):
    """AC-177: A credential the deployment cannot resolve stops the start, naming it @spec:AC-177"""
    import httpx
    import psycopg
    import pytest
    from conftest import API_TOKEN, api_config

    from src.main import build_service
    from src.secrets import SecretResolutionError, resolver_over

    config = api_config(
        monkeypatch, schema_dsn, RAG_API_CREDENTIALS_SECRET="env:A_TOKEN_NOBODY_SET"
    )
    monkeypatch.delenv("A_TOKEN_NOBODY_SET", raising=False)
    monkeypatch.setenv("OPENROUTER_KEY", "sk-or-not-the-one-under-test")

    def _refuse_database(*args, **kwargs):
        raise AssertionError("the database was contacted before the credential was resolved")

    monkeypatch.setattr(psycopg, "connect", _refuse_database)

    provider_calls: list[httpx.Request] = []

    def _track_provider(request: httpx.Request) -> httpx.Response:
        provider_calls.append(request)
        return httpx.Response(500)

    with pytest.raises(SecretResolutionError) as excinfo:
        build_service(
            config,
            resolver=resolver_over(),
            http=httpx.Client(transport=httpx.MockTransport(_track_provider)),
        )

    message = str(excinfo.value)
    assert "api.credentials_secret" in message
    assert "env:A_TOKEN_NOBODY_SET" in message

    assert API_TOKEN not in message

    assert provider_calls == []

def test_ac_178(monkeypatch, conn, caplog):
    """AC-178: The schedule is in the deployment's file, and an unreadable one stops the boot @spec:AC-178"""
    import logging
    from datetime import datetime, timedelta, timezone

    import pytest

    from src.config import ConfigError
    from src.worker import run_schedule
    from tests.conftest import contract_payload, ingest_config, ingest_world, pdf_file

    def _utc(*args: int) -> datetime:
        return datetime(*args, tzinfo=timezone.utc)

    class _Clock:
        def __init__(self, start: datetime) -> None:
            self.current = start

        def now(self) -> datetime:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += timedelta(seconds=seconds)

    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1")],
        outputs={"doc-1": contract_payload()},
        RAG_SCHEDULE_FULL_SWEEP="0 3 * * *",
        RAG_SCHEDULE_INCREMENTAL="*/15 * * * *",
    )
    clock = _Clock(_utc(2026, 9, 10, 12, 0, 0))

    with caplog.at_level(logging.INFO, logger="src.worker"):
        run_schedule(
            conn,
            world.pipeline,
            should_stop=lambda: True,
            clock=clock.now,
            sleeper=clock.sleep,
        )

    reported = " | ".join(record.getMessage() for record in caplog.records)
    assert "'0 3 * * *'" in reported and "'*/15 * * * *'" in reported
    assert "2026-09-11T03:00:00+00:00" in reported
    assert "2026-09-10T12:15:00+00:00" in reported

    with pytest.raises(ConfigError) as excinfo:
        ingest_config(monkeypatch, RAG_SCHEDULE_FULL_SWEEP="every tuesday at three")

    message = str(excinfo.value)
    assert "[schedule]" in message
    assert "full_sweep" in message
    assert "every tuesday at three" in message

    quiet = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1")],
        outputs={"doc-1": contract_payload()},
    )
    assert quiet.config.schedule.full_sweep == ""
    assert quiet.config.schedule.incremental == ""

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.worker"):
        run_schedule(
            conn,
            quiet.pipeline,
            should_stop=lambda: True,
            clock=clock.now,
            sleeper=clock.sleep,
        )

    assert quiet.reads == 0
    assert any("runs nothing on its own" in record.getMessage() for record in caplog.records)

def test_ac_179(monkeypatch, conn, caplog):
    """AC-179: A scheduled run that arrives while one is in flight is skipped, not queued @spec:AC-179"""
    import logging
    from datetime import datetime, timedelta, timezone

    from src.ingest_idempotency import APP_VERSION, claim
    from src.sync_ledger import RUN_KIND_SWEEP, start_run
    from src.worker import run_schedule
    from tests.conftest import contract_payload, ingest_world, pdf_file

    def _utc(*args: int) -> datetime:
        return datetime(*args, tzinfo=timezone.utc)

    class _Clock:

        def __init__(self, start: datetime) -> None:
            self.current = start

        def now(self) -> datetime:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += timedelta(seconds=seconds)

    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1")],
        outputs={"doc-1": contract_payload()},
        RAG_SCHEDULE_FULL_SWEEP="* * * * *",
    )

    run_id = start_run(conn, client=world.pipeline.client, kind=RUN_KIND_SWEEP)
    claim(
        conn,
        external_id="doc-vanished",
        content_digest="d",
        fingerprint="f",
        run_id=run_id,
        app_version=APP_VERSION,
    )

    clock = _Clock(_utc(2026, 9, 10, 3, 0, 0))

    with caplog.at_level(logging.INFO, logger="src.worker"):
        run_schedule(
            conn,
            world.pipeline,
            should_stop=lambda: any(
                "in flight" in record.getMessage() for record in caplog.records
            ),
            clock=clock.now,
            sleeper=clock.sleep,
            poll_interval=90.0,
        )

    assert any("in flight" in record.getMessage() for record in caplog.records)
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM sync_runs")
        assert cursor.fetchone()[0] == 1
    assert world.reads == 0

def test_ac_180(monkeypatch, conn):
    """AC-180: A run missed while the process was down is not made up afterwards @spec:AC-180"""
    from datetime import datetime, timedelta, timezone

    from src.ingest_idempotency import APP_VERSION, claim
    from src.worker import run_schedule
    from tests.conftest import contract_payload, ingest_world, pdf_file

    def _utc(*args: int) -> datetime:
        return datetime(*args, tzinfo=timezone.utc)

    class _Clock:

        def __init__(self, start: datetime) -> None:
            self.current = start

        def now(self) -> datetime:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += timedelta(seconds=seconds)

    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1"), pdf_file("doc-2")],
        outputs={"doc-1": contract_payload(), "doc-2": contract_payload()},
        RAG_SCHEDULE_FULL_SWEEP="0 3 * * *",
    )

    claim(conn, external_id="doc-1", content_digest="stale", fingerprint="stale", app_version=APP_VERSION)

    clock = _Clock(_utc(2026, 9, 13, 10, 0, 0))

    run_schedule(
        conn,
        world.pipeline,
        should_stop=lambda: world.reads >= 2,
        clock=clock.now,
        sleeper=clock.sleep,
        poll_interval=24 * 3600.0,
    )

    assert world.reads == 2
    with conn.cursor() as cursor:
        cursor.execute("SELECT kind, status, sweep_completed FROM sync_runs ORDER BY id")
        rows = cursor.fetchall()
    assert rows == [("sweep", "completed", True)]

def test_ac_181(monkeypatch, conn):
    """AC-181: A worker asked to stop finishes cleanly and leaves its work claimable @spec:AC-181"""
    from datetime import datetime, timedelta, timezone

    from src.worker import run_schedule, sweep
    from tests.conftest import contract_payload, ingest_world, pdf_file

    def _utc(*args: int) -> datetime:
        return datetime(*args, tzinfo=timezone.utc)

    class _Clock:

        def __init__(self, start: datetime) -> None:
            self.current = start

        def now(self) -> datetime:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += timedelta(seconds=seconds)

    names = ("doc-1", "doc-2", "doc-3")
    world = ingest_world(
        monkeypatch,
        files=[pdf_file(name) for name in names],
        outputs={name: contract_payload() for name in names},
        RAG_SCHEDULE_FULL_SWEEP="* * * * *",
    )
    clock = _Clock(_utc(2026, 9, 10, 3, 0, 0))

    run_schedule(
        conn,
        world.pipeline,
        should_stop=lambda: world.reads >= 1,
        clock=clock.now,
        sleeper=clock.sleep,
        poll_interval=90.0,
        page_size=1,
    )

    assert world.reads == 1
    with conn.cursor() as cursor:
        cursor.execute("SELECT status, sweep_completed FROM sync_runs")
        assert cursor.fetchone() == ("halted", False)
        cursor.execute("SELECT external_id FROM documents ORDER BY external_id")
        assert [row[0] for row in cursor.fetchall()] == ["doc-1"]

    again = ingest_world(
        monkeypatch,
        files=[pdf_file(name) for name in names],
        outputs={name: contract_payload() for name in names},
        storage=world.storage,
    )
    sweep(conn, again.pipeline)

    assert again.reads == 2
