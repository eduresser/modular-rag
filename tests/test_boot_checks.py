"""Unit-level coverage for src/boot.py that is not itself an acceptance criterion.

The acceptance criteria (AC-004, AC-005, AC-006, AC-007) are proven in
tests/test_spec_config_and_boot.py — see CLAUDE.md for why the proof lives
there. This file covers the happy path and edge cases underneath it: checks
that pass, checks that are skipped for lack of information, and several
failures aggregating into one error.
"""

import pathlib

import pytest
from pydantic import ValidationError

from src.boot import (
    BootCompatibilityError,
    check_boot_compatibility,
)
from src.config import (
    ApiSettings,
    AppConfig,
    DocaiChunkSettings,
    DocaiProcessSettings,
    DbSettings,
    DocaiReadSettings,
    RequestSettings,
    EmbedSettings,
    FetchWindow,
    GDriveSourceSettings,
    McpSettings,
    OperationLimit,
    RerankSettings,
    ScheduleSettings,
    SearchSettings,
    SearchWindow,
    SourceSettings,
    SurfaceDefaults,
    ToolSurfaceDefaults,
    ToolSurfaceLimits,
    SurfaceLimits,
    TelemetrySettings,
    UsageSettings,
)
from tests.conftest import GOOGLE_CREDENTIAL_REFERENCE


def _valid_config(**overrides):
    # `read=` and `chunk=` are still how a test says "this deployment differs
    # in its read half", because that is what the test is about. They are the
    # halves of one process now, so they are folded into it here rather than at
    # every call site — the nesting is the service's shape, not a thing each
    # test has to restate.
    read = overrides.pop("read", None)
    chunk = overrides.pop("chunk", None)
    process = overrides.pop("process", None)
    base = dict(
        source=GDriveSourceSettings(
            type="gdrive",
            folder_ids=["folder-123"],
            credentials_secret=GOOGLE_CREDENTIAL_REFERENCE,
        ),
        process=DocaiProcessSettings(
            type="docai",
            credentials_secret=GOOGLE_CREDENTIAL_REFERENCE,
            read=DocaiReadSettings(processor_id="projects/x/processors/y",
                processor_version="pretrained-layout-parser-v1.0-2024-06-03",
                artifact_bucket="modular-rag-artifacts"),
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
        ),
        mcp=McpSettings(credentials_secret="env:MCP_TOKEN"),
        usage=UsageSettings(),
        telemetry=TelemetrySettings(client="example"),
        schedule=ScheduleSettings(),
    )
    if process is not None:
        base["process"] = process
    elif read is not None or chunk is not None:
        base["process"] = base["process"].model_copy(
            update={
                **({"read": read} if read is not None else {}),
                **({"chunk": chunk} if chunk is not None else {}),
            }
        )
    base.update(overrides)
    return AppConfig(**base)


def test_a_fully_compatible_config_boots_cleanly():
    config = _valid_config()

    check_boot_compatibility(config)  # must not raise


def test_a_chunk_size_within_the_window_is_not_flagged():
    # 32,768 is the chosen model's own documented limit — a chunk exactly at it
    # is the boundary case, and the boundary is inclusive.
    config = _valid_config(chunk=DocaiChunkSettings(size=32_768))

    check_boot_compatibility(config)  # must not raise


def test_a_model_cannot_be_configured_without_the_window_it_has_to_fit_in():
    """What replaced the registry, and it is a stronger guarantee than it was.

    The gate used to refuse a model absent from a table in code — a table this
    service maintained about somebody else's products, which could only ever be
    behind. The window is a required setting now, so a model with no window is
    not a deployment that boots and fails a check: it is a file that does not
    parse. There is no unregistered model left to have a case about.
    """
    with pytest.raises(ValidationError) as excinfo:
        EmbedSettings(
            model="some-future-model",
            dimension=2048,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
        )

    assert "context_window" in str(excinfo.value)


def test_the_dimension_check_is_skipped_on_a_first_boot_with_no_stored_column():
    config = _valid_config()

    check_boot_compatibility(config, stored_dimension=None)  # must not raise


def test_a_matching_stored_dimension_is_not_flagged():
    config = _valid_config()

    check_boot_compatibility(config, stored_dimension=2048)  # must not raise


def test_the_allowlist_accepts_a_path_and_query_on_an_allowed_host():
    config = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1/embeddings?beta=true",
            credentials_secret="env:OPENROUTER_KEY",
        )
    )

    check_boot_compatibility(config)  # must not raise


def test_a_request_that_collides_with_nothing_is_not_flagged():
    config = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
            request=RequestSettings(
                headers={"x-api-key": "the-key"},
                params={"api-version": "2023-05-15"},
                body={"encoding_format": "float", "user": "example-user"},
            ),
        )
    )

    check_boot_compatibility(config)  # must not raise


def test_every_colliding_key_is_named_and_says_what_decides_it():
    """A denylist where there was an allowlist, and it names the guarantee.

    "Not permitted, here is the permitted set" told an operator what they may
    write. It cannot any more — a deployment may write anything a provider
    wants — so the refusal has to say why *this* key is the exception, which is
    the only thing left worth saying.
    """
    config = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
            request=RequestSettings(
                headers={"Authorization": "Basic smuggled"},
                body={"encoding_format": "float", "allow_fallbacks": True},
            ),
        )
    )

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config)

    message = str(excinfo.value)
    assert "allow_fallbacks" in message and "AC-052" in message
    assert "Authorization" in message and "embed.credentials_secret" in message
    # A key that collides with nothing is not mentioned at all.
    assert "encoding_format" not in message


def test_several_incompatibilities_are_all_reported_in_one_error():
    config = _valid_config(
        chunk=DocaiChunkSettings(size=40_000),
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://evil.example.com/v1",
            credentials_secret="env:OPENROUTER_KEY",
            request=RequestSettings(body={"model": "something-else"}),
        ),
    )

    try:
        check_boot_compatibility(config, stored_dimension=768)
        raise AssertionError("expected BootCompatibilityError")
    except BootCompatibilityError as exc:
        message = str(exc)
        assert "40000" in message
        assert "768" in message
        assert "evil.example.com" in message
        assert "request.body.model" in message


def test_no_model_name_is_written_anywhere_in_the_boot_gate():
    """The gate compares what the file says; it does not look a vendor up.

    It used to hold a window per model and a host per gateway, so a deployment
    naming a model neither table knew was refused for a line it had no way to
    write — and on a module that declares no chunk size it was not refused at
    all, because the check returns early and the unregistered model went to
    the provider unexamined.
    """
    import src.boot as boot

    source = pathlib.Path(boot.__file__).read_text().lower()

    for vendor in ("qwen", "gemini", "openai", "openrouter", "google/"):
        assert vendor not in source, f"src/boot.py still names {vendor!r}"


def test_an_empty_allowlist_refuses_every_host_rather_than_permitting_them():
    """Fail closed: a list nobody filled in is not a list that permits anything.

    The allowlist moved into the file, and a setting that defaults to empty is
    the one shape where "moved to configuration" could quietly mean "switched
    off". It names the setting so the operator's next move is one line.
    """
    config = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=[],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
        ),
    )

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config, stored_dimension=2048)

    message = str(excinfo.value)
    assert "openrouter.ai" in message
    assert "embed.allowed_hosts" in message


def test_a_cut_larger_than_the_configured_window_still_refuses_the_boot():
    """The guard survived the move; only where the number lives changed."""
    config = _valid_config(
        chunk=DocaiChunkSettings(size=4000),
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=2048,
            allowed_hosts=["openrouter.ai"],
            base_url="https://openrouter.ai/api/v1",
            credentials_secret="env:OPENROUTER_KEY",
        ),
    )

    with pytest.raises(BootCompatibilityError) as excinfo:
        check_boot_compatibility(config, stored_dimension=2048)

    message = str(excinfo.value)
    assert "4000" in message and "2048" in message


# ── A ranking setting named under a surface (AC-295) ──────────────────────


def test_a_ranking_setting_under_a_surfaces_defaults_is_refused_at_construction():
    # AC-295 is enforced by the type, not by a boot check somebody has to
    # remember to write: SurfaceDefaults forbids any key it does not declare,
    # so a ranking knob written under a surface never reaches
    # check_boot_compatibility at all — the service refuses before boot runs.
    with pytest.raises(ValidationError) as refused:
        SurfaceDefaults(dense_weight=0.9)  # type: ignore[call-arg]
    assert "dense_weight" in str(refused.value)


def test_a_ranking_setting_under_the_tool_surface_is_refused_at_construction():
    with pytest.raises(ValidationError) as refused:
        McpSettings(credentials_secret="env:MCP_TOKEN", candidate_depth=200)  # type: ignore[call-arg]
    assert "candidate_depth" in str(refused.value)


# ── The window ceiling walk (AC-296) ───────────────────────────────────────


def test_a_search_default_above_its_own_surfaces_ceiling_is_refused():
    config = _valid_config(
        api=ApiSettings(
            credentials_secret="env:API_TOKEN",
            allowed_hosts=["rag.example.internal"],
            defaults=SurfaceDefaults(search=SearchWindow(limit=60)),
            limits=SurfaceLimits(search=OperationLimit(limit=50)),
        )
    )
    with pytest.raises(BootCompatibilityError) as refused:
        check_boot_compatibility(config)
    message = str(refused.value)
    assert "[api]" in message
    assert "api.defaults.search.limit (60)" in message
    assert "api.limits.search.limit (50)" in message


def test_a_fetch_default_above_its_own_surfaces_ceiling_is_refused_for_mcp_too():
    # The walk covers every surface it finds, not just `api` — the tool
    # surface's own defaults/limits pair is checked the same way.
    config = _valid_config(
        mcp=McpSettings(
            credentials_secret="env:MCP_TOKEN",
            defaults=ToolSurfaceDefaults(fetch=FetchWindow(limit=40)),
            limits=ToolSurfaceLimits(fetch=OperationLimit(limit=10)),
        )
    )
    with pytest.raises(BootCompatibilityError) as refused:
        check_boot_compatibility(config)
    message = str(refused.value)
    assert "[mcp]" in message
    assert "mcp.defaults.fetch.limit (40)" in message
    assert "mcp.limits.fetch.limit (10)" in message


def test_a_tool_surface_default_above_the_conventional_ceiling_is_refused():
    # AC-245's own rule, kept: the tool surface's window may not exceed the
    # conventional surface's ceiling for the same operation, even when it
    # respects its own.
    config = _valid_config(
        api=ApiSettings(
            credentials_secret="env:API_TOKEN",
            allowed_hosts=["rag.example.internal"],
            limits=SurfaceLimits(fetch=OperationLimit(limit=50)),
        ),
        mcp=McpSettings(
            credentials_secret="env:MCP_TOKEN",
            defaults=ToolSurfaceDefaults(fetch=FetchWindow(limit=60)),
            limits=ToolSurfaceLimits(fetch=OperationLimit(limit=60)),
        ),
    )
    with pytest.raises(BootCompatibilityError) as refused:
        check_boot_compatibility(config)
    message = str(refused.value)
    assert "[mcp]" in message
    assert "mcp.defaults.fetch.limit (60)" in message
    assert "api.limits.fetch.limit (50)" in message


def test_several_window_violations_are_all_reported_together():
    # One boot, every reason — the same aggregation the other checks get, so an
    # operator fixes one file once instead of discovering the second failure
    # after redeploying for the first.
    config = _valid_config(
        api=ApiSettings(
            credentials_secret="env:API_TOKEN",
            allowed_hosts=["rag.example.internal"],
            defaults=SurfaceDefaults(search=SearchWindow(limit=60)),
            limits=SurfaceLimits(search=OperationLimit(limit=50), fetch=OperationLimit(limit=50)),
        ),
        mcp=McpSettings(
            credentials_secret="env:MCP_TOKEN",
            defaults=ToolSurfaceDefaults(fetch=FetchWindow(limit=60)),
            limits=ToolSurfaceLimits(fetch=OperationLimit(limit=60)),
        ),
    )
    with pytest.raises(BootCompatibilityError) as refused:
        check_boot_compatibility(config)
    message = str(refused.value)
    assert message.count("[api]") == 1
    assert message.count("[mcp]") == 1


def test_the_shipped_defaults_satisfy_their_own_ceilings():
    # The shipped defaults have to satisfy their own check, or every
    # deployment that names none of these settings fails to boot.
    config = _valid_config()
    check_boot_compatibility(config)  # must not raise
