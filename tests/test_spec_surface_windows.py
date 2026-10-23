
def test_ac_294():
    """AC-294: Each surface's window is read from its own block @spec:AC-294"""
    from types import SimpleNamespace

    import anyio

    import src.mcp as mcp_module
    from src.config import (
        ApiSettings,
        AppConfig,
        DbSettings,
        DocaiChunkSettings,
        DocaiProcessSettings,
        DocaiReadSettings,
        EmbedSettings,
        FetchWindow,
        GDriveSourceSettings,
        McpSettings,
        OperationLimit,
        RerankSettings,
        ScheduleSettings,
        SearchSettings,
        SearchWindow,
        SurfaceDefaults,
    ToolSurfaceDefaults,
    ToolSurfaceLimits,
        SurfaceLimits,
        TelemetrySettings,
        UsageSettings,
    )

    config = AppConfig(
        source=GDriveSourceSettings(
            type="gdrive", folder_ids=["folder-123"], credentials_secret="env:GOOGLE_KEY"
        ),
        process=DocaiProcessSettings(
            type="docai",
            credentials_secret="env:GOOGLE_KEY",
            read=DocaiReadSettings(
                processor_id="projects/x/processors/y",
                processor_version="pretrained-layout-parser-v1.0-2024-06-03",
                artifact_bucket="modular-rag-artifacts",
            ),
            chunk=DocaiChunkSettings(size=800),
        ),
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
        ),
        db=DbSettings(url="postgresql://localhost/rag"),
        search=SearchSettings(),
        rerank=RerankSettings(),
        api=ApiSettings(
            credentials_secret="env:API_TOKEN",
            allowed_hosts=["rag.example.internal"],
            defaults=SurfaceDefaults(
                search=SearchWindow(limit=17),
                fetch=FetchWindow(limit=23),
            ),
            limits=SurfaceLimits(
                search=OperationLimit(limit=50),
                fetch=OperationLimit(limit=50),
            ),
        ),
        mcp=McpSettings(
            credentials_secret="env:MCP_TOKEN",
            defaults=ToolSurfaceDefaults(
                search=SearchWindow(limit=5),
                fetch=FetchWindow(limit=7),
            ),
            limits=ToolSurfaceLimits(
                search=OperationLimit(limit=50),
                fetch=OperationLimit(limit=50),
            ),
        ),
        usage=UsageSettings(),
        telemetry=TelemetrySettings(client="example"),
        schedule=ScheduleSettings(),
    )
    assert config.api.defaults.search.limit != config.mcp.defaults.search.limit
    assert config.api.defaults.fetch.limit != config.mcp.defaults.fetch.limit

    service = SimpleNamespace(config=config, embedder=object(), scorer=None)
    server = mcp_module.build_server(service)

    captured_search: dict = {}

    def fake_run_search(query, **kwargs):
        captured_search["result_count"] = query.result_count
        return SimpleNamespace()

    captured_fetch: dict = {}

    def fake_fetch(identifier, **kwargs):
        captured_fetch.update(kwargs)
        return SimpleNamespace(external_id="123")

    original_run_search = mcp_module.run_search
    original_fetch = mcp_module.fetch
    original_grouped = mcp_module.grouped_search_payload
    original_facts = mcp_module._document_facts
    original_document = mcp_module._document
    mcp_module.run_search = fake_run_search
    mcp_module.fetch = fake_fetch
    mcp_module.grouped_search_payload = lambda outcome, audience: {"status": "found"}
    mcp_module._document_facts = lambda config, external_id: (None, ())
    mcp_module._document = lambda found, *, link, location: {"status": "found"}
    try:
        anyio.run(lambda: server.call_tool("search", {"query": "rescisão"}))
        anyio.run(lambda: server.call_tool("fetch", {"id": "123"}))
    finally:
        mcp_module.run_search = original_run_search
        mcp_module.fetch = original_fetch
        mcp_module.grouped_search_payload = original_grouped
        mcp_module._document_facts = original_facts
        mcp_module._document = original_document

    assert captured_search["result_count"] == config.mcp.defaults.search.limit
    assert captured_search["result_count"] != config.api.defaults.search.limit
    from src.responses import ASSISTANT, SERVICE
    from src.usecases.fetch import span_for

    assert captured_fetch["audience"] == ASSISTANT
    assert captured_fetch["limit"] is None
    assert span_for(config, audience=ASSISTANT, named_piece=None) == config.mcp.defaults.fetch.limit
    assert span_for(config, audience=SERVICE, named_piece=None) == config.api.defaults.fetch.limit
    assert config.mcp.defaults.fetch.limit != config.api.defaults.fetch.limit

def test_ac_295():
    """AC-295: A ranking setting named under a surface stops the boot @spec:AC-295"""
    import pytest

    from src.config import ConfigError, load_config
    from tests.conftest import INGEST_DEPLOYMENT, _deep_copy, write_config

    ranking_knobs = ("dense_weight", "lexical_weight", "candidate_depth", "index_search_effort")
    for surface in ("api", "mcp"):
        for knob in ranking_knobs:
            document = _deep_copy(INGEST_DEPLOYMENT)
            document.setdefault(surface, {}).setdefault("defaults", {}).setdefault("search", {})[knob] = 0.9
            with pytest.raises(ConfigError) as refusal:
                load_config(write_config(document))
            message = str(refusal.value)
            assert f"defaults.search.{knob}" in message, message
            assert f"[{surface}]" in message, message
            assert "unrecognised setting" in message, message

    accepted = _deep_copy(INGEST_DEPLOYMENT)
    accepted.setdefault("search", {})["dense_weight"] = 0.9
    assert load_config(write_config(accepted)).search.dense_weight == 0.9

def test_ac_296():
    """AC-296: A window above its own ceiling, or above the conventional one, stops the boot @spec:AC-296"""
    import pytest

    from src.boot import BootCompatibilityError, check_boot_compatibility
    from src.config import (
        FetchWindow,
        OperationLimit,
        SearchWindow,
        SurfaceDefaults,
        SurfaceLimits,
    )
    from tests.test_boot_checks import _valid_config as _deployment

    for surface in ("api", "mcp"):
        for operation, window in (("search", SearchWindow(limit=40)), ("fetch", FetchWindow(limit=40))):
            config = _deployment()
            block = getattr(config, surface)
            setattr(block.defaults, operation, window)
            setattr(block.limits, operation, OperationLimit(limit=7))
            with pytest.raises(BootCompatibilityError) as refusal:
                check_boot_compatibility(config)
            message = str(refusal.value)
            assert f"{surface}.defaults.{operation}.limit" in message, message
            assert f"{surface}.limits.{operation}.limit" in message, message
            assert "40" in message and "7" in message, message

    config = _deployment()
    config.api.limits.fetch = OperationLimit(limit=20)
    config.api.defaults.fetch = FetchWindow(limit=20)
    config.mcp.limits.fetch = OperationLimit(limit=30)
    config.mcp.defaults.fetch = FetchWindow(limit=30)
    with pytest.raises(BootCompatibilityError) as refusal:
        check_boot_compatibility(config)
    message = str(refusal.value)
    assert "mcp.defaults.fetch.limit" in message, message
    assert "api.limits.fetch.limit" in message, message

    healthy = _deployment()
    healthy.api.defaults.fetch = FetchWindow(limit=20)
    healthy.api.limits.fetch = OperationLimit(limit=50)
    healthy.mcp.defaults.fetch = FetchWindow(limit=10)
    healthy.mcp.limits.fetch = OperationLimit(limit=10)
    check_boot_compatibility(healthy)

def test_ac_297():
    """AC-297: Each surface names the credential that opens it @spec:AC-297"""
    import httpx
    from fastapi.testclient import TestClient

    from src.config import (
        ApiSettings,
        AppConfig,
        DbSettings,
        DocaiChunkSettings,
        DocaiProcessSettings,
        DocaiReadSettings,
        EmbedSettings,
        GDriveSourceSettings,
        McpSettings,
        RerankSettings,
        ScheduleSettings,
        SearchSettings,
        TelemetrySettings,
        UsageSettings,
    )
    from src.main import MCP_PATH, build_service, create_app

    allowed_host = "rag.example.internal"
    api_token = "the-conventional-routes-token"
    mcp_token = "the-tool-surface-token"

    def resolver(reference: str) -> str:
        known = {
            "env:API_TOKEN": api_token,
            "env:MCP_TOKEN": mcp_token,
            "env:GOOGLE_KEY": "unused",
            "env:OPENROUTER_KEY": "unused",
        }
        return known[reference]

    config = AppConfig(
        source=GDriveSourceSettings(
            type="gdrive", folder_ids=["folder-123"], credentials_secret="env:GOOGLE_KEY"
        ),
        process=DocaiProcessSettings(
            type="docai",
            credentials_secret="env:GOOGLE_KEY",
            read=DocaiReadSettings(
                processor_id="projects/x/processors/y",
                processor_version="pretrained-layout-parser-v1.0-2024-06-03",
                artifact_bucket="modular-rag-artifacts",
            ),
            chunk=DocaiChunkSettings(size=800),
        ),
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
        ),
        db=DbSettings(url="postgresql://127.0.0.1:1/nowhere"),
        search=SearchSettings(),
        rerank=RerankSettings(),
        api=ApiSettings(credentials_secret="env:API_TOKEN", allowed_hosts=[allowed_host]),
        mcp=McpSettings(credentials_secret="env:MCP_TOKEN"),
        usage=UsageSettings(),
        telemetry=TelemetrySettings(client="example"),
        schedule=ScheduleSettings(),
    )

    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    service = build_service(config, resolver=resolver, http=http)
    app = create_app(service)

    with TestClient(app, base_url=f"http://{allowed_host}") as client:
        conventional_with_own_token = client.get(
            "/v1/filters", headers={"Authorization": f"Bearer {api_token}"}
        )
        conventional_with_tool_token = client.get(
            "/v1/filters", headers={"Authorization": f"Bearer {mcp_token}"}
        )
        assert conventional_with_own_token.status_code != 401
        assert conventional_with_tool_token.status_code == 401

        tool_with_own_token = client.post(
            MCP_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Authorization": f"Bearer {mcp_token}",
                "Accept": "application/json, text/event-stream",
            },
        )
        tool_with_conventional_token = client.post(
            MCP_PATH,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert tool_with_own_token.status_code != 401
        assert tool_with_conventional_token.status_code == 401
