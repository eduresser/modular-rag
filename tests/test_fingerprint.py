"""Unit-level coverage for src/fingerprint.py that is not itself an acceptance criterion.

AC-008 is proven in tests/test_spec_config_and_boot.py — see CLAUDE.md for why the proof
lives there. This file covers determinism and settings the acceptance criterion does not
mention explicitly.
"""

from src.config import (
    ApiSettings,
    McpSettings,
    AppConfig,
    DocaiChunkSettings,
    DocaiProcessSettings,
    DbSettings,
    DocaiReadSettings,
    EmbedSettings,
    RequestSettings,
    GDriveSourceSettings,
    RerankSettings,
    ScheduleSettings,
    SearchSettings,
    SourceSettings,
    TelemetrySettings,
    UsageSettings,
)
import pytest

from src.fingerprint import (
    STAGE_CUT,
    STAGE_EMBED,
    STAGE_READ,
    STAGES,
    compute_fingerprint,
    digest_of,
    stage_fingerprints,
    stage_settings,
    write_time_settings,
)
from src.processing.recut import cut_version
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
        # The tool surface names the credential that opens it (AC-297). It
        # reaches no fingerprint group either way — a credential is a reference
        # and never a value, and `src/fingerprint.py`'s allowlist is explicit
        # precisely so no `_secret` setting can join a hash written into every
        # row.
        mcp=McpSettings(credentials_secret="env:API_TOKEN"),
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


def test_the_fingerprint_is_deterministic_across_equal_configs():
    a = _valid_config()
    b = _valid_config()

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_the_fingerprint_is_a_sha256_hex_digest():
    fingerprint = compute_fingerprint(_valid_config())

    assert len(fingerprint) == 64
    assert all(c in "0123456789abcdef" for c in fingerprint)


def test_request_body_key_order_does_not_change_the_fingerprint():
    a = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://api.openai.com/v1",
            credentials_secret="env:OPENROUTER_KEY",
            request=RequestSettings(body={"user": "example-user", "encoding_format": "float"}),
        )
    )
    b = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://api.openai.com/v1",
            credentials_secret="env:OPENROUTER_KEY",
            request=RequestSettings(body={"encoding_format": "float", "user": "example-user"}),
        )
    )

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_the_source_module_and_folder_do_not_move_the_fingerprint():
    # Which folder is synced (or which source module reads it) does not change
    # what is written into a chunk once a document reaches the pipeline.
    reference = "env:GOOGLE_SERVICE_ACCOUNT_KEY"
    a = _valid_config(
        source=GDriveSourceSettings(
            type="gdrive", folder_ids=["folder-123"], credentials_secret=reference
        )
    )
    b = _valid_config(
        source=GDriveSourceSettings(
            type="gdrive", folder_ids=["a-different-folder"], credentials_secret=reference
        )
    )

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_the_embedding_base_url_does_not_move_the_fingerprint():
    # Which endpoint serves the model does not change the vectors it produces
    # for identical input, only the embedding model and its parameters do.
    a = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://api.openai.com/v1",
            credentials_secret="env:OPENROUTER_KEY",
        )
    )
    b = _valid_config(
        embed=EmbedSettings(
            model="qwen/qwen3-embedding-8b",
            dimension=2048,
            context_window=32768,
            allowed_hosts=["openrouter.ai"],
            base_url="https://api.openai.com/v2",
            credentials_secret="env:OPENROUTER_KEY",
        )
    )

    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_the_readers_limits_and_region_do_not_move_the_fingerprint():
    # Which of the provider's two regions ran the read, how many batches ran at
    # once and where a document was refused for being too long all describe how
    # the corpus was produced, never what is in it. Re-indexing because one of
    # them changed would be pure waste — the write-time half of this namespace
    # is the processor version and the two description switches, and those are
    # in the payload.
    reference = compute_fingerprint(_valid_config())

    # `region` sits on the process rather than on its read half — it is the
    # host both the batch call and the re-cut call go to — so it is moved
    # through the process, and it must move the fingerprint no more from there
    # than it did from beside the limits.
    moved = _valid_config(process=_valid_config().process.model_copy(update={"region": "eu"}))
    assert compute_fingerprint(moved) == reference, "region"

    for change in (
        {"page_limit": 50},
        {"max_document_bytes": 10_000},
        {"max_documents_per_batch": 10},
        {"max_concurrent_batches": 1},
    ):
        moved = _valid_config(
            read=DocaiReadSettings(processor_id="projects/x/processors/y",
                processor_version="pretrained-layout-parser-v1.0-2024-06-03",
                artifact_bucket="modular-rag-artifacts",
                **change)
        )
        assert compute_fingerprint(moved) == reference, change


def test_the_reader_version_and_the_description_switches_do_move_it():
    # A different processor version reads the same scan differently, and with
    # description off the table caption never arrives at all (ASM-024) — a
    # corpus indexed that way is not the same corpus.
    reference = compute_fingerprint(_valid_config())

    for change in (
        {"processor_version": "pretrained-layout-parser-v1.5"},
        {"enable_table_annotation": False},
        {"enable_image_annotation": False},
    ):
        settings = dict(
            processor_id="projects/x/processors/y",
            processor_version="pretrained-layout-parser-v1.0-2024-06-03",
            artifact_bucket="modular-rag-artifacts",
        )
        settings.update(change)
        moved = _valid_config(read=DocaiReadSettings(**settings))
        assert compute_fingerprint(moved) != reference, change


def test_the_bucket_and_the_processor_itself_do_not_move_it():
    # Where the artifact is kept, and which processor instance produced it, do
    # not change the text: a deployment moving buckets must not re-index.
    reference = compute_fingerprint(_valid_config())
    moved = _valid_config(
        read=DocaiReadSettings(processor_id="projects/outro/processors/z",
            processor_version="pretrained-layout-parser-v1.0-2024-06-03",
            artifact_bucket="outro-bucket")
    )

    assert compute_fingerprint(moved) == reference


# ── The stage split ───────────────────────────────────────────────────────
#
# The allowlist is grouped by the stage each setting invalidates, because that
# is what lets the pipeline charge three different prices for "the
# configuration changed": a read is ten dollars a thousand pages, a re-cut two
# cents, and a re-embedding one cent a million tokens. The whole fingerprint
# stays exactly what it was — the split adds a finer question, it does not
# change the coarse answer.


def test_the_three_stages_partition_the_write_time_settings():
    grouped = stage_settings(_valid_config())
    flat = write_time_settings(_valid_config())

    assert set(grouped) == set(STAGES)
    # No setting in two groups, and none in none of them: a setting counted
    # twice would move two stage digests, and one counted nowhere would move
    # the whole fingerprint with no stage to explain it.
    assert sum(len(grouped[stage]) for stage in STAGES) == len(flat)
    assert {key for stage in STAGES for key in grouped[stage]} == set(flat)


def test_the_whole_fingerprint_is_the_digest_of_the_groups_merged():
    config = _valid_config()

    assert stage_fingerprints(config).whole == compute_fingerprint(config)
    assert compute_fingerprint(config) == digest_of(write_time_settings(config))


def test_each_setting_moves_exactly_one_stage():
    reference = stage_fingerprints(_valid_config())
    moved = {
        STAGE_READ: _valid_config(
            read=DocaiReadSettings(processor_id="projects/x/processors/y",
                processor_version="pretrained-layout-parser-v1.5",
                artifact_bucket="modular-rag-artifacts")
        ),
        STAGE_CUT: _valid_config(chunk=DocaiChunkSettings(size=400)),
        STAGE_EMBED: _valid_config(
            embed=EmbedSettings(
                model="google/gemini-embedding-001",
                dimension=2048,
                context_window=32768,
                allowed_hosts=["openrouter.ai"],
                base_url="https://openrouter.ai/api/v1",
                credentials_secret="env:OPENROUTER_KEY",
            )
        ),
    }

    for stage, config in moved.items():
        after = stage_fingerprints(config)
        assert after[stage] != reference[stage], stage
        assert after.whole != reference.whole, stage
        for other in STAGES:
            if other != stage:
                assert after[other] == reference[other], (stage, other)


def test_the_parse_version_belongs_to_the_cut_and_not_to_the_read(monkeypatch):
    # It is this service's own version for turning the reader's kept output
    # into pieces (ASM-053, amended): no longer a setting, but what the cut
    # makes of a fixed sample. Putting it under the read stage would make
    # changing our own mapping code cost ten dollars a thousand pages instead
    # of nothing, so it is the cut digest and only the cut digest that moves.
    reference = stage_fingerprints(_valid_config())
    monkeypatch.setattr("src.fingerprint.cut_version", lambda _module: "a-different-cut")
    moved = stage_fingerprints(_valid_config())

    assert moved[STAGE_CUT] != reference[STAGE_CUT]
    assert moved[STAGE_READ] == reference[STAGE_READ]
    assert moved[STAGE_EMBED] == reference[STAGE_EMBED]


def test_the_cut_version_is_measured_rather_than_configured():
    # There is no `parse` section any more and no setting anywhere that names
    # a cut version. The value in the hash comes from the cut itself, which is
    # what makes forgetting to bump it impossible rather than merely unlikely.
    assert not hasattr(_valid_config(), "parse")
    assert stage_settings(_valid_config())[STAGE_CUT]["parse.version"] == cut_version("docai")


def test_what_the_reader_is_given_belongs_to_the_read_group(monkeypatch):
    # AC-211: a different export target for a native document, or a different
    # pinned renderer version for an uploaded one, means the reader receives
    # different bytes — a document that must be read again, not merely
    # re-cut or re-embedded.
    reference = stage_fingerprints(_valid_config())

    monkeypatch.setattr(
        "src.fingerprint.export_plan",
        lambda kind: type("Plan", (), {"media_type": "text/html"})(),
    )
    moved_target = stage_fingerprints(_valid_config())
    assert moved_target[STAGE_READ] != reference[STAGE_READ]
    assert moved_target[STAGE_CUT] == reference[STAGE_CUT]
    assert moved_target[STAGE_EMBED] == reference[STAGE_EMBED]
    monkeypatch.undo()

    monkeypatch.setattr("src.fingerprint.renderer_version", lambda: "a-different-suite")
    moved_renderer = stage_fingerprints(_valid_config())
    assert moved_renderer[STAGE_READ] != reference[STAGE_READ]
    assert moved_renderer[STAGE_CUT] == reference[STAGE_CUT]
    assert moved_renderer[STAGE_EMBED] == reference[STAGE_EMBED]


def test_a_stage_nobody_recognises_is_refused_by_name():
    with pytest.raises(KeyError) as excinfo:
        stage_fingerprints(_valid_config())["rerank"]

    assert "rerank" in str(excinfo.value)


def test_no_secret_reaches_the_payload():
    # The allowlist is explicit precisely so a field added later cannot join
    # the hash, and a _SECRET setting is the one that must never.
    flat = write_time_settings(_valid_config())

    assert not any("secret" in key.lower() for key in flat)
    assert not any("secret" in str(value).lower() for value in flat.values())


def test_a_cut_setting_moves_the_cut_stage_and_only_that_one():
    # How the document is cut is a property of every piece, so changing it has
    # to cost a re-cut rather than passing unnoticed — and a re-cut, never a
    # re-read: `read` must not move, or a change worth two cents a thousand
    # pages is billed at ten dollars.
    #
    # This used to be proven by swapping cutters, when there were two. There is
    # one now, so it is proven by the knob that still exists; the claim is the
    # same claim.
    reference = stage_fingerprints(_valid_config())
    moved = stage_fingerprints(
        _valid_config(chunk=DocaiChunkSettings(size=400))
    )

    assert moved[STAGE_CUT] != reference[STAGE_CUT]
    assert moved[STAGE_READ] == reference[STAGE_READ]
    assert moved[STAGE_EMBED] == reference[STAGE_EMBED]


def test_a_module_declares_whether_it_has_write_time_settings_of_its_own():
    # An empty tuple and an absent key are different answers, and only one of
    # them is a decision. A module added without an entry raises rather than
    # quietly contributing nothing to the hash its pieces were read and cut
    # under — and BOTH groups are keyed by the process, because one module runs
    # both halves.
    from src.config import REGISTERED_PROCESS_MODULES

    from src.fingerprint import CUTTER_SETTINGS, READER_SETTINGS

    assert set(CUTTER_SETTINGS) == set(REGISTERED_PROCESS_MODULES)
    assert set(READER_SETTINGS) == set(REGISTERED_PROCESS_MODULES)

    # And a module that is not registered is refused rather than hashed as
    # nothing: an absent key is what a silent invalidation would look like.
    from src.fingerprint import _module_settings

    with pytest.raises(KeyError, match="declares no write-time settings"):
        _module_settings(CUTTER_SETTINGS, object(), "nobody", what="cutter", group="X")


def test_the_readers_own_cutting_carries_no_knob_it_does_not_have():
    # The cut group holds what the provider's request actually takes, and
    # nothing else (AC-234). A knob this service would have to honour by hand
    # would re-cut a corpus and change not a single piece — which is what the
    # three the local cutter contributed used to do the moment it was not the
    # cutter in force.
    cut = stage_settings(_valid_config())[STAGE_CUT]

    assert cut["chunk.type"] == "docai"
    assert not [key for key in cut if "split_on_heading" in key or "separator" in key]


def test_the_schedule_is_in_no_stage_and_never_moves_the_fingerprint():
    # A schedule changes when the pipeline runs, never what it writes — it is
    # in no group in src/fingerprint.py at all, so a deployment moving its
    # sweep from a nightly run to a weekly one must not re-cut or re-embed
    # anything, let alone read the corpus again.
    reference = stage_fingerprints(_valid_config())
    moved = stage_fingerprints(
        _valid_config(schedule=ScheduleSettings(full_sweep="0 3 * * *", incremental="*/15 * * * *"))
    )

    assert moved.whole == reference.whole
    for stage in STAGES:
        assert moved[stage] == reference[stage], stage
    assert not any(key.startswith("schedule.") for key in write_time_settings(_valid_config()))


def test_the_stages_credential_does_not_move_the_fingerprint():
    """A reference names where a value is kept, and changes no text and no vector.

    It is also the one class of setting the allowlist must never admit by
    accident: `src.fingerprint` is explicit precisely so that no `_secret`
    setting can reach a hash that is written into every row.
    """
    reference = compute_fingerprint(_valid_config())
    elsewhere = "env:A_DIFFERENT_VARIABLE"

    moved_read = _valid_config(
        read=DocaiReadSettings(processor_id="projects/x/processors/y",
            processor_version="pretrained-layout-parser-v1.0-2024-06-03",
            artifact_bucket="modular-rag-artifacts"),
        chunk=DocaiChunkSettings(size=800),
    )

    assert compute_fingerprint(moved_read) == reference
    assert "credentials_secret" not in str(write_time_settings(moved_read))
