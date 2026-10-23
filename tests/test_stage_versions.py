
import pytest

from src.fingerprint import STAGE_CUT, STAGE_EMBED, STAGE_READ
from src.processing.recut import cut_version
from src.store import ModelStamp, Piece, upsert_document, upsert_pieces
from src.versions import (
    DECLARED_VERSIONS,
    StageVersions,
    UnknownStageVersion,
    available,
    configured_versions,
    recorded_versions,
    resolve,
    unresolvable,
)
from src.reprocess import record_state
from src.fingerprint import stage_fingerprints
from tests.conftest import TEST_DIMENSION, direction, ingest_config


def test_two_versions_come_from_the_file_and_the_third_from_the_cut(monkeypatch):
    config = ingest_config(monkeypatch)

    versions = configured_versions(config)

    assert versions == StageVersions(
        read="pretrained-layout-parser-v1.0-2024-06-03",
        cut=cut_version("docai"),
        embed="qwen/qwen3-embedding-8b",
    )
    assert not hasattr(config, "parse")
    assert versions[STAGE_CUT] == cut_version("docai")


def test_the_stages_a_vendor_owns_are_declared_and_the_one_we_own_is_measured():
    assert set(DECLARED_VERSIONS) == {STAGE_READ, STAGE_CUT, STAGE_EMBED}
    assert all(DECLARED_VERSIONS[stage] for stage in (STAGE_READ, STAGE_CUT))
    assert DECLARED_VERSIONS[STAGE_EMBED] == ()
    assert DECLARED_VERSIONS[STAGE_CUT] == ("1", "2", "3")
    assert cut_version("docai") not in DECLARED_VERSIONS[STAGE_CUT]


def test_the_measured_cut_version_is_resolvable_without_anybody_declaring_it(monkeypatch):
    config = ingest_config(monkeypatch)

    assert cut_version("docai") in available(STAGE_CUT, config)
    assert resolve(STAGE_CUT, cut_version("docai"), config) == cut_version("docai")


def test_the_embedding_stage_declares_the_same_models_the_boot_gate_knows(monkeypatch):
    assert DECLARED_VERSIONS[STAGE_EMBED] == ()


def test_a_configured_version_is_always_resolvable_even_if_nobody_declared_it(monkeypatch):
    config = ingest_config(monkeypatch, RAG_DOCAI_PROCESSOR_VERSION="pretrained-layout-parser-v9")

    assert "pretrained-layout-parser-v9" in available(STAGE_READ, config)
    assert resolve(STAGE_READ, "pretrained-layout-parser-v9", config) == "pretrained-layout-parser-v9"


def test_a_recorded_version_this_build_retired_is_refused_by_name(monkeypatch):
    config = ingest_config(monkeypatch)

    with pytest.raises(UnknownStageVersion) as excinfo:
        resolve(STAGE_READ, "pretrained-layout-parser-v1.5", config)

    message = str(excinfo.value)
    assert "pretrained-layout-parser-v1.5" in message
    assert "pretrained-layout-parser-v1.0-2024-06-03" in message
    assert "two different things" in message


def test_an_unknown_stage_is_refused_rather_than_defaulted(monkeypatch):
    config = ingest_config(monkeypatch)

    with pytest.raises(UnknownStageVersion) as excinfo:
        available("rerank", config)

    assert "rerank" in str(excinfo.value)
    with pytest.raises(KeyError):
        configured_versions(config)["rerank"]


def test_the_recorded_versions_of_a_document_come_from_where_each_one_lives(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn,
        document_id=document_id,
        content_digest="digest",
        fingerprints=stage_fingerprints(config),
        reader_version="pretrained-layout-parser-v1.0-2024-06-03",
        cut_version="1",
        piece_count=1,
    )
    stamp = ModelStamp(model="qwen/qwen3-embedding-8b", dimension=TEST_DIMENSION)
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=stamp,
        pieces=[Piece(ordinal=0, content="Cláusula", embedding=direction(0.0), page_start=1, page_end=1)],
    )

    assert recorded_versions(conn, document_id) == {
        STAGE_READ: "pretrained-layout-parser-v1.0-2024-06-03",
        STAGE_CUT: "1",
        STAGE_EMBED: "qwen/qwen3-embedding-8b",
    }


def test_a_document_nobody_has_indexed_records_no_version(monkeypatch, conn):
    document_id = upsert_document(conn, external_id="doc-1")

    assert recorded_versions(conn, document_id) == {
        STAGE_READ: None,
        STAGE_CUT: None,
        STAGE_EMBED: None,
    }


def test_the_whole_corpus_is_asked_at_once_before_an_upgrade(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn, document_id=document_id, content_digest="digest",
        fingerprints=stage_fingerprints(config),
        reader_version="pretrained-layout-parser-v1.5", cut_version="0", piece_count=0,
    )

    assert unresolvable(conn, config) == (
        (STAGE_CUT, "0"),
        (STAGE_READ, "pretrained-layout-parser-v1.5"),
    )


def test_an_archive_this_build_understands_reports_nothing_to_resolve(monkeypatch, conn):
    config = ingest_config(monkeypatch)
    document_id = upsert_document(conn, external_id="doc-1")
    record_state(
        conn, document_id=document_id, content_digest="digest",
        fingerprints=stage_fingerprints(config),
        reader_version=config.read.processor_version, cut_version=configured_versions(config).cut,
        piece_count=0,
    )

    assert unresolvable(conn, config) == ()
