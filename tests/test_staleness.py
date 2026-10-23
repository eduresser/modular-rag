
from src.fingerprint import stage_fingerprints
from src.processing.recut import cut_version
from src.reprocess import record_state
from src.staleness import archive_staleness, record_generation, recorded_settings
from src.store import upsert_document
from tests.conftest import ingest_config


def _indexed(conn, external_id, config, *, fingerprints=None):
    fingerprints = fingerprints or stage_fingerprints(config)
    document_id = upsert_document(conn, external_id=external_id)
    record_state(
        conn,
        document_id=document_id,
        content_digest=f"digest-of-{external_id}",
        fingerprints=fingerprints,
        reader_version=config.read.processor_version,
        cut_version=cut_version("docai"),
        piece_count=2,
    )
    return document_id


def test_a_generation_records_the_settings_its_fingerprint_stood_for(monkeypatch, conn):
    config = ingest_config(monkeypatch)

    fingerprint = record_generation(conn, config)

    settings = recorded_settings(conn, fingerprint)
    assert settings["chunk.size"] == 800
    assert settings["embed.model"] == "qwen/qwen3-embedding-8b"
    assert settings["docai.processor_version"] == config.read.processor_version
    assert not any("secret" in key or "base_url" in key for key in settings)


def test_recording_the_same_generation_twice_keeps_the_first(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    fingerprint = record_generation(conn, config)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT first_seen_at FROM indexing_generations WHERE fingerprint = %s", (fingerprint,)
        )
        (first_seen,) = cursor.fetchone()

    record_generation(conn, config)

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*), min(first_seen_at) FROM indexing_generations")
        count, seen = cursor.fetchone()
    assert (count, seen) == (1, first_seen)


def test_a_fingerprint_nobody_recorded_reads_as_nothing(conn):
    assert recorded_settings(conn, "never-written") is None


def test_an_empty_archive_is_current(monkeypatch, conn):
    config = ingest_config(monkeypatch)

    report = archive_staleness(conn, config)

    assert report.current is True
    assert report.affected == 0
    assert report.changes == ()


def test_the_report_names_the_setting_the_stage_and_both_values(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    record_generation(conn, config)
    _indexed(conn, "doc-1", config)

    moved = ingest_config(monkeypatch, RAG_EMBED_MODEL="google/gemini-embedding-001")
    report = archive_staleness(conn, moved)

    assert report.current is False
    assert report.affected == 1
    assert [(c.setting, c.recorded, c.configured, c.stage) for c in report.changes] == [
        ("embed.model", "qwen/qwen3-embedding-8b", "google/gemini-embedding-001", "embed")
    ]
    assert str(report.changes[0]) == "embed.model: 'qwen/qwen3-embedding-8b' → 'google/gemini-embedding-001'"


def test_a_reprocessing_stopped_halfway_is_reported_against_every_generation(monkeypatch, conn):
    first = ingest_config(monkeypatch)
    record_generation(conn, first)
    _indexed(conn, "doc-1", first)

    second = ingest_config(monkeypatch, RAG_CHUNK_SIZE="400")
    record_generation(conn, second)
    _indexed(conn, "doc-2", second, fingerprints=stage_fingerprints(second))

    third = ingest_config(
        monkeypatch, RAG_CHUNK_SIZE="200", RAG_CHUNK_INCLUDE_HEADINGS="false"
    )
    report = archive_staleness(conn, third)

    assert report.affected == 2
    assert len(report.stale_fingerprints) == 2
    assert {change.setting for change in report.changes} == {
        "chunk.size",
        "chunk.include_headings",
    }
    recorded = {change.setting: change.recorded for change in report.changes}
    assert recorded["chunk.size"] == 800
    assert recorded["chunk.include_headings"] is True


def test_a_generation_whose_settings_were_never_recorded_is_named_rather_than_ignored(
    monkeypatch, conn
):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn, document_id=document_id, content_digest="digest",
        fingerprints=stage_fingerprints(ingest_config(monkeypatch, RAG_CHUNK_SIZE="400")),
        reader_version="v1", cut_version="1", piece_count=1,
    )

    report = archive_staleness(conn, config)

    assert report.current is False
    assert report.affected == 1
    assert len(report.unrecorded_fingerprints) == 1
    assert report.changes == ()
    assert "settings not recorded" in report.summary()


def test_the_stages_are_reported_dearest_first(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    record_generation(conn, config)
    _indexed(conn, "doc-1", config)

    moved = ingest_config(
        monkeypatch,
        RAG_DOCAI_ENABLE_TABLE_ANNOTATION="false",
        RAG_CHUNK_SIZE="400",
        RAG_EMBED_MODEL="google/gemini-embedding-001",
    )
    report = archive_staleness(conn, moved)

    assert report.stages == ("read", "cut", "embed")
    assert set(report.settings) == {
        "docai.enable_table_annotation", "chunk.size", "embed.model"
    }


def test_the_report_writes_nothing_and_starts_nothing(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    record_generation(conn, config)
    _indexed(conn, "doc-1", config)
    moved = ingest_config(monkeypatch, RAG_CHUNK_SIZE="400")
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        (before,) = cursor.fetchone()

    archive_staleness(conn, moved)
    archive_staleness(conn, moved)

    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM usage_events")
        assert cursor.fetchone()[0] == before
        cursor.execute("SELECT count(*) FROM indexing_generations")
        assert cursor.fetchone()[0] == 1
