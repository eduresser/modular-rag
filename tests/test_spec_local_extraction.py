
import psycopg
import pytest

from src.ingest import ingest_documents
from tests.conftest import (
    ARTIFACT_BUCKET,
    LOCAL_DEPLOYMENT,
    deployment_config,
    ingest_world,
    pdf_bytes,
    pdf_file,
)


def _sweep_into(world, conn):
    return ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))


def test_ac_266(conn: psycopg.Connection, monkeypatch):
    """AC-266: A deployment naming the free module indexes without reaching any reader @spec:AC-266"""
    pages = ["Clausula primeira do objeto.", "Clausula segunda do prazo."]
    world = ingest_world(
        monkeypatch,
        deployment=LOCAL_DEPLOYMENT,
        files=[pdf_file("doc-1", name="Contrato", content=pdf_bytes(pages))],
    )
    assert world.config.process.type == "pdftext"

    run = _sweep_into(world, conn)

    assert [result.action for result in run.results] == ["read"]
    with conn.cursor() as cursor:
        cursor.execute("SELECT content FROM chunks ORDER BY ordinal")
        stored = [row[0] for row in cursor.fetchall()]
    assert stored == pages

    assert world.reads == 0
    assert world.docai.submissions == []
    assert world.docai.requests == []

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT provider, units, unit_price, cost FROM usage_events "
            "WHERE operation = 'read'"
        )
        reads = cursor.fetchall()
    assert reads == [("internal", 2, 0, 0)]


def test_ac_267(conn: psycopg.Connection, monkeypatch):
    """AC-267: The pieces are one to a page, and each names the page it is @spec:AC-267"""
    pages = ["Clausula primeira.", "Clausula segunda.", "", "Clausula quarta."]
    world = ingest_world(
        monkeypatch,
        deployment=LOCAL_DEPLOYMENT,
        files=[pdf_file("doc-1", name="Contrato", content=pdf_bytes(pages))],
    )

    _sweep_into(world, conn)

    with conn.cursor() as cursor:
        cursor.execute("SELECT content, page_start, page_end FROM chunks ORDER BY ordinal")
        stored = cursor.fetchall()
    assert stored == [
        ("Clausula primeira.", 1, 1),
        ("Clausula segunda.", 2, 2),
        ("Clausula quarta.", 4, 4),
    ]

    assert len(stored) == 3
    assert all(content.strip() for content, _start, _end in stored)


def test_ac_268(conn: psycopg.Connection, monkeypatch):
    """AC-268: A PDF with no text layer is refused, naming why @spec:AC-268"""
    from src.processing.artifact import READ_STATE_FAILED
    from src.processing.pdftext import NO_TEXT_REASON

    world = ingest_world(
        monkeypatch,
        deployment=LOCAL_DEPLOYMENT,
        files=[
            pdf_file("scan-1", name="Contrato digitalizado", content=pdf_bytes(["", ""])),
            pdf_file("doc-2", name="Contrato", content=pdf_bytes(["Clausula primeira."])),
        ],
    )

    run = _sweep_into(world, conn)

    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT d.external_id, r.state, r.failure_reason "
            "FROM document_reads r JOIN documents d ON d.id = r.document_id "
            "ORDER BY d.external_id"
        )
        records = cursor.fetchall()
    refused = {external_id: (state, reason) for external_id, state, reason in records}
    assert refused["scan-1"][0] == READ_STATE_FAILED
    assert NO_TEXT_REASON in refused["scan-1"][1]
    assert "optical character recognition" in refused["scan-1"][1]

    outcomes = {result.external_id: result.action for result in run.results}
    assert outcomes["doc-2"] == "read"
    with conn.cursor() as cursor:
        cursor.execute("SELECT content FROM chunks")
        assert [row[0] for row in cursor.fetchall()] == ["Clausula primeira."]


def test_ac_269():
    """AC-269: A format the chosen module does not take is refused before anything is spent @spec:AC-269"""
    from src.processing import pdftext
    from src.processing.preflight import (
        DOCAI_LIMITS,
        Candidate,
        OUTCOME_REFUSED,
        SUPPORTED_MEDIA_TYPES,
        preflight,
    )

    spreadsheet = Candidate(
        document_id=1,
        external_id="planilha-1",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        byte_size=2_048,
    )

    on_the_free_route = preflight(spreadsheet, limits=pdftext.LIMITS)
    on_the_paid_route = preflight(spreadsheet, limits=DOCAI_LIMITS)

    assert on_the_free_route.outcome == OUTCOME_REFUSED
    assert spreadsheet.media_type in on_the_free_route.reason
    for accepted in pdftext.SUPPORTED_MEDIA_TYPES:
        assert accepted in on_the_free_route.reason

    assert on_the_paid_route.accepted
    assert spreadsheet.media_type in SUPPORTED_MEDIA_TYPES
    assert spreadsheet.media_type not in pdftext.SUPPORTED_MEDIA_TYPES

    assert "image/tiff" not in on_the_free_route.reason


def test_ac_270():
    """AC-270: Both modules are reached through one name, and neither leaks into the other's file @spec:AC-270"""
    import copy

    from src.config import (
        REGISTERED_PROCESS_MODULES,
        ConfigError,
        DocaiProcessSettings,
        PdfTextProcessSettings,
    )
    from tests.conftest import INGEST_DEPLOYMENT

    paid = deployment_config(INGEST_DEPLOYMENT)
    free = deployment_config(LOCAL_DEPLOYMENT)
    assert {key: value for key, value in INGEST_DEPLOYMENT.items() if key != "process"} == {
        key: value for key, value in LOCAL_DEPLOYMENT.items() if key != "process"
    }

    assert isinstance(paid.process, DocaiProcessSettings)
    assert isinstance(free.process, PdfTextProcessSettings)
    assert (paid.process.type, free.process.type) == ("docai", "pdftext")

    borrowed = copy.deepcopy(LOCAL_DEPLOYMENT)
    borrowed["process"]["read"] = {
        "processor_version": "pretrained-layout-parser-v1.0-2024-06-03"
    }
    with pytest.raises(ConfigError) as leaked_into_free:
        deployment_config(borrowed)
    assert "processor_version" in str(leaked_into_free.value)

    summarised = copy.deepcopy(INGEST_DEPLOYMENT)
    summarised["process"]["chunk"]["pages_per_piece"] = 1
    with pytest.raises(ConfigError) as leaked_into_paid:
        deployment_config(summarised)
    assert "pages_per_piece" in str(leaked_into_paid.value)

    assert paid.read.artifact_bucket == ARTIFACT_BUCKET
    assert not hasattr(free.read, "processor_id")
    assert not hasattr(free.chunk, "size")
    assert LOCAL_DEPLOYMENT["process"] == {"type": "pdftext"}
    import pathlib

    import yaml

    base = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent / "apps" / "base" / "config.yaml")
        .read_text(encoding="utf-8")
    )
    assert set(base["process"]) == {"type"}, sorted(base["process"])
    assert base["process"]["type"] in REGISTERED_PROCESS_MODULES

    from src.config import merge_documents

    assert paid.process.type != base["process"]["type"]
    chosen = merge_documents(base, {"process": INGEST_DEPLOYMENT["process"]})["process"]
    assert chosen["type"] == "docai"
    assert "pdftext" not in str(chosen)
    assert merge_documents(base, {})["process"] == {"type": base["process"]["type"]}


def test_ac_271():
    """AC-271: What a module is paid for is what its own group hashes @spec:AC-271"""
    import copy

    from src.fingerprint import STAGE_CUT, STAGE_READ, stage_fingerprints, stage_settings
    from src.processing.recut import SAMPLE_READERS, cut_version
    from tests.conftest import INGEST_DEPLOYMENT

    free = deployment_config(LOCAL_DEPLOYMENT)
    paid = deployment_config(INGEST_DEPLOYMENT)
    free_digests = stage_fingerprints(free)

    with_a_bigger_paid_cut = copy.deepcopy(INGEST_DEPLOYMENT)
    with_a_bigger_paid_cut["process"]["chunk"]["size"] = 1_600
    moved = deployment_config(with_a_bigger_paid_cut)

    assert stage_fingerprints(free).cut == free_digests.cut
    assert stage_fingerprints(free).read == free_digests.read
    assert stage_fingerprints(moved).cut != stage_fingerprints(paid).cut

    free_read, free_cut = stage_settings(free)[STAGE_READ], stage_settings(free)[STAGE_CUT]
    paid_read, paid_cut = stage_settings(paid)[STAGE_READ], stage_settings(paid)[STAGE_CUT]
    assert not [key for key in free_read if key.startswith("docai.")]
    assert not [key for key in free_cut if key.startswith("chunk.") and key != "chunk.type"]
    assert not [key for key in paid_read if key.startswith("pdftext.")]
    assert "pdftext.extractor_version" in free_read

    assert cut_version("pdftext") != cut_version("docai")
    assert {name for name, _cut in SAMPLE_READERS} == {"docai", "pdftext"}
    assert free_cut["parse.version"] == cut_version("pdftext")
    assert paid_cut["parse.version"] == cut_version("docai")


def test_ac_272(conn: psycopg.Connection, monkeypatch):
    """AC-272: The free process names no credential and reaches no service @spec:AC-272"""
    import copy

    from src.config import ConfigError, PdfTextProcessSettings, PdfTextReadSettings
    from src.processing.artifact import ARCHIVE_SCHEME, load_parsed_artifact
    from src.processing.artifact import require_parsed_artifact

    free = deployment_config(LOCAL_DEPLOYMENT)

    assert "credentials_secret" not in PdfTextProcessSettings.model_fields
    assert "artifact_bucket" not in PdfTextReadSettings.model_fields
    assert not PdfTextReadSettings.model_fields
    assert isinstance(free.process, PdfTextProcessSettings)

    for offered in ({"credentials_secret": "env:GOOGLE_SERVICE_ACCOUNT_KEY"},
                    {"read": {"artifact_bucket": "algum-bucket"}}):
        asked = copy.deepcopy(LOCAL_DEPLOYMENT)
        asked["process"].update(offered)
        with pytest.raises(ConfigError) as refusal:
            deployment_config(asked)
        assert "unrecognised setting" in str(refusal.value)

    world = ingest_world(
        monkeypatch,
        deployment=LOCAL_DEPLOYMENT,
        files=[pdf_file("doc-1", name="Contrato", content=pdf_bytes(["Cláusula primeira."]))],
    )
    _sweep_into(world, conn)

    artifact = require_parsed_artifact(conn, _document_id(conn, "doc-1"))
    assert artifact.uri.startswith(ARCHIVE_SCHEME)
    assert world.storage.objects == {}

    kept = load_parsed_artifact(conn, world.pipeline.store, artifact)
    assert [entry["text"] for entry in kept["pages"]] == ["Cláusula primeira."]
    assert kept["extractor"] == "pypdf"


def _document_id(conn: psycopg.Connection, external_id: str) -> int:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = %s", (external_id,))
        return cursor.fetchone()[0]


def test_ac_273(conn: psycopg.Connection, monkeypatch):
    """AC-273: A completed sweep retires the pieces another module produced @spec:AC-273"""
    from src.reprocess import retire_foreign_pieces
    from tests.conftest import INGEST_DEPLOYMENT, FakeObjectStorage, reader_chunk, reader_payload, text_block

    storage = FakeObjectStorage(ARTIFACT_BUCKET)
    files = [
        pdf_file("doc-legivel", name="Contrato",
                 content=pdf_bytes(["Cláusula primeira.", "Cláusula segunda."])),
        pdf_file("doc-escaneado", name="Digitalizado", content=pdf_bytes(["", ""])),
    ]
    outputs = {
        external: reader_payload(
            [text_block("b1", f"LIDO PELO DOCAI — {external}", pages=(1, 1), kind="heading-1")],
            [reader_chunk("c1", f"LIDO PELO DOCAI — {external}", ["b1"])],
        )
        for external in ("doc-legivel", "doc-escaneado")
    }
    pago = ingest_world(monkeypatch, deployment=INGEST_DEPLOYMENT, files=files,
                        outputs=outputs, storage=storage)
    _sweep_into(pago, conn)
    assert _indexed(conn) == {"doc-escaneado": 1, "doc-legivel": 1}

    gratis = ingest_world(monkeypatch, deployment=LOCAL_DEPLOYMENT, files=files, storage=storage)
    _sweep_into(gratis, conn)
    assert _indexed(conn) == {"doc-escaneado": 1, "doc-legivel": 2}

    held = retire_foreign_pieces(conn, module="pdftext", may_remove=False)
    assert held.refusal == "sweep_did_not_complete"
    assert held.considered == 1 and held.retired == ()
    assert _indexed(conn) == {"doc-escaneado": 1, "doc-legivel": 2}

    done = retire_foreign_pieces(conn, module="pdftext", may_remove=True)

    assert done.clean
    assert done.retired == ("doc-escaneado",)
    assert _indexed(conn) == {"doc-legivel": 2}
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT r.failure_reason FROM document_reads r JOIN documents d ON d.id = r.document_id "
            " WHERE d.external_id = 'doc-escaneado'"
        )
        assert "refused for want of a text layer" in cursor.fetchone()[0]

    empty = retire_foreign_pieces(conn, module="a-module-that-indexed-nothing", may_remove=True)
    assert empty.refusal == "module_produced_nothing"
    assert empty.retired == ()
    assert _indexed(conn) == {"doc-legivel": 2}

    from src.ingest_errors import FAILURE_TRANSIENT
    from src.ingest_idempotency import claim, fail

    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE external_id = 'doc-legivel'")
        legivel = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE document_ingest SET process_module = 'docai' WHERE document_id = %s",
            (legivel,),
        )
    taken = claim(
        conn, external_id="doc-legivel", content_digest="d", fingerprint="f",
        document_id=legivel,
    )
    fail(conn, taken.claim.id, reason="a blip at the write",
         kind=FAILURE_TRANSIENT, document_id=legivel)
    waiting = retire_foreign_pieces(conn, module="pdftext", may_remove=True)

    assert waiting.clean
    assert waiting.spared == ("doc-legivel",)
    assert waiting.retired == ()
    assert _indexed(conn) == {"doc-legivel": 2}


def _indexed(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT d.external_id, count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
            " GROUP BY d.external_id ORDER BY d.external_id"
        )
        return {external: int(count) for external, count in cursor.fetchall()}
