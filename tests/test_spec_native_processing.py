
def test_ac_227():
    """AC-227: One name runs both stages @spec:AC-227"""
    from src.config import DocaiChunkSettings, DocaiReadSettings, DocaiProcessSettings
    from tests.conftest import FULL_DEPLOYMENT, deployment_config

    document = FULL_DEPLOYMENT
    assert set(document["process"]) >= {"type", "read", "chunk"}

    config = deployment_config(document)

    assert config.process.type == "docai"
    assert config.read is config.process.read
    assert config.chunk is config.process.chunk
    assert isinstance(config.read, DocaiReadSettings)
    assert isinstance(config.chunk, DocaiChunkSettings)

    for half in (DocaiReadSettings, DocaiChunkSettings):
        assert "type" not in half.model_fields, half.__name__
    assert "type" in DocaiProcessSettings.model_fields


def test_ac_228():
    """AC-228: A sub-section shaped for another module is refused, naming where it sits @spec:AC-228"""
    import pytest

    from src.config import ConfigError, DocaiProcessSettings, DocaiReadSettings
    from tests.conftest import FULL_DEPLOYMENT, deployment_config

    document = {
        **FULL_DEPLOYMENT,
        "process": {
            **FULL_DEPLOYMENT["process"],
            "read": {**FULL_DEPLOYMENT["process"]["read"], "shard_limit": 4},
        },
    }

    with pytest.raises(ConfigError) as excinfo:
        deployment_config(document)

    message = str(excinfo.value)
    assert "[process]" in message
    assert "read.shard_limit" in message
    assert "unrecognised setting" in message

    from typing import get_type_hints

    hints = get_type_hints(DocaiProcessSettings)
    assert hints["read"] is DocaiReadSettings


def test_ac_229():
    """AC-229: One call spends one credential, named once @spec:AC-229"""
    from src.config import (
        ChunkSettings,
        DocaiChunkSettings,
        DocaiProcessSettings,
        DocaiReadSettings,
        ReadSettings,
    )
    from tests.conftest import GOOGLE_CREDENTIAL_REFERENCE, FULL_DEPLOYMENT, deployment_config

    config = deployment_config(FULL_DEPLOYMENT)
    assert config.process.credentials_secret == GOOGLE_CREDENTIAL_REFERENCE

    assert DocaiProcessSettings.model_fields["credentials_secret"].is_required()

    for half in (DocaiReadSettings, ReadSettings, DocaiChunkSettings, ChunkSettings):
        assert "credentials_secret" not in half.model_fields, half.__name__


READ_STAGE_DIGEST = "65978405888a450bcb74dc9ac7c510b46a6bf94b42d406a67f18586d43e7b493"

READ_STAGE_KEYS = (
    "docai.processor_version",
    "docai.enable_table_annotation",
    "docai.enable_image_annotation",
    "gdrive_export.document_export_target",
    "render.renderer_version",
)


def test_ac_230(monkeypatch):
    """AC-230: The read stage's digest is the one the corpus was indexed under @spec:AC-230"""
    from src.fingerprint import STAGE_READ, digest_of, stage_settings
    from tests.test_fingerprint import _valid_config

    monkeypatch.setattr(
        "src.fingerprint.renderer_version", lambda: "pinned-renderer-for-the-lock"
    )
    group = stage_settings(_valid_config())[STAGE_READ]

    assert tuple(group) == READ_STAGE_KEYS
    assert digest_of(group) == READ_STAGE_DIGEST, (
        "the read stage's digest moved. Every indexed document will be READ again, at ten "
        "dollars a thousand pages, to arrive at the same vectors. If a key followed the "
        "rename, put its spelling back; if the export target or the renderer moved, that is "
        "AC-211 and the literal moves with it — deliberately."
    )


CUTTING_SETTINGS: dict[str, str] = {
    "size": "chunkSize",
    "include_headings": "includeAncestorHeadings",
}


def test_ac_234():
    """AC-234: A cutting setting the provider does not take is not offered @spec:AC-234"""
    from src.config import DocaiChunkSettings
    from src.processing.docai import ChunkingOptions, LayoutParser

    declared = set(DocaiChunkSettings.model_fields)

    assert declared == set(CUTTING_SETTINGS), (
        "a setting was added to or removed from the cut stage without saying what the "
        "provider calls it. If the provider takes it, name it in CUTTING_SETTINGS; if it "
        "does not, it is not a cutting setting — it is work this service would have to do "
        "by hand, which is the promise nobody keeps (AC-234)."
    )

    import httpx

    reader = LayoutParser(
        processor_id="projects/x/processors/y",
        processor_version="pretrained-layout-parser-v1.0-2024-06-03",
        http=httpx.Client(),
        token=lambda: "not-used-to-build-a-body",
        chunking=ChunkingOptions(size=800, cutter="docai", include_ancestor_headings=True),
    )
    chunking_config = reader.layout_config()["layoutConfig"]["chunkingConfig"]

    assert set(chunking_config) == set(CUTTING_SETTINGS.values()), chunking_config

    assert "include_document_summary" not in declared


def test_ac_231():
    """AC-231: A block of an unrecognised kind keeps its text @spec:AC-231"""
    from src.models import VersionStamps
    from src.processing.docai import chunks_from_document, index_blocks
    from tests.conftest import DOCAI_VERSION, reader_chunk, reader_payload

    strange = {
        "blockId": "x1",
        "pageSpan": {"pageStart": 3, "pageEnd": 3},
        "audioBlock": {"transcript": "gravação da audiência de conciliação"},
    }
    payload = reader_payload(
        [strange], [reader_chunk("c1", "", ["x1"], pages=(3, 3))]
    )

    mapped = chunks_from_document(
        payload,
        external_id="doc-1",
        stamps=VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1"),
    )

    assert index_blocks(payload)["x1"].text == "gravação da audiência de conciliação"
    assert mapped.chunks[0].content == "gravação da audiência de conciliação"
    assert mapped.chunks[0].pages == (3, 3)

    assert mapped.unrecognised_blocks == ("x1",)

    invented = {
        "blockId": "y1",
        "pageSpan": {"pageStart": 4, "pageEnd": 4},
        "formulaBlock": {"latex": "R$ 12.400,00 × IGP-M"},
    }
    index = index_blocks(reader_payload([invented], []))
    assert index["y1"].text == "R$ 12.400,00 × IGP-M"
    assert index.unrecognised == ("y1",)


def test_ac_232():
    """AC-232: A piece names the kinds of block it was cut from, in the reader's own words @spec:AC-232"""
    from src.models import METADATA_BLOCK_KINDS, VersionStamps
    from src.processing.docai import chunks_from_document
    from tests.conftest import (
        DOCAI_VERSION,
        reader_chunk,
        reader_payload,
        table_block,
        text_block,
    )

    table = table_block("t1", ["Parcela", "Valor"], [["1", "R$ 12.400,00"]], pages=(1, 1))
    payload = reader_payload(
        [
            text_block("h1", "CLÁUSULA QUINTA — DO VALOR", pages=(1, 1), kind="heading-1"),
            table,
            {"blockId": "i1", "pageSpan": {"pageStart": 1, "pageEnd": 1},
             "imageBlock": {"annotations": {"description": "Planta baixa do imóvel"}}},
        ],
        [
            reader_chunk("c1", "CLÁUSULA QUINTA — DO VALOR | 1 |", ["h1", "t1"]),
            reader_chunk("c2", "", ["i1"]),
        ],
    )

    mapped = chunks_from_document(
        payload,
        external_id="doc-1",
        stamps=VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1"),
    )
    kinds = [
        tuple(piece.to_piece([0.0]).metadata[METADATA_BLOCK_KINDS]) for piece in mapped.chunks
    ]

    assert kinds[0] == ("textBlock", "heading-1", "tableBlock")
    assert kinds[1] == ("imageBlock",)

    for borrowed in ("section_header", "picture", "table", "text", "paragraph"):
        assert borrowed not in kinds[0] + kinds[1], borrowed


def test_ac_233(schema_dsn):
    """AC-233: The pipeline reads, cuts and stores with the document model absent @spec:AC-233"""
    import json
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    named = [
        path.relative_to(root).as_posix()
        for path in sorted((root / "src").rglob("*.py"))
        if "docling" in path.read_text(encoding="utf-8")
    ]
    assert named == [], named

    program = """
import json
import sys

class Absent:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "docling_core" or name.startswith("docling_core."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, Absent())
try:
    import docling_core  # noqa: F401
except ModuleNotFoundError:
    pass
else:  # pragma: no cover
    raise SystemExit("the blocker did not block")

import psycopg

from src.db import create_schema
from src.models import VersionStamps
from src.processing.docai import chunks_from_document
from src.store import ModelStamp, upsert_document, upsert_pieces

PAYLOAD = {
    "documentLayout": {"blocks": [
        {"blockId": "h1", "pageSpan": {"pageStart": 1, "pageEnd": 1},
         "textBlock": {"text": "CONTRATO", "type": "heading-1"}},
        {"blockId": "p1", "pageSpan": {"pageStart": 1, "pageEnd": 1},
         "textBlock": {"text": "O imovel destina-se a uso comercial.", "type": "paragraph"}},
    ]},
    "chunkedDocument": {"chunks": [
        {"chunkId": "c1", "content": "O imovel destina-se a uso comercial.",
         "sourceBlockIds": ["p1"], "pageSpan": {"pageStart": 1, "pageEnd": 1}},
    ]},
}

mapped = chunks_from_document(
    PAYLOAD,
    external_id="doc-sem-modelo",
    stamps=VersionStamps(reader_version="v1", fingerprint="fp-1"),
    filename="contrato.pdf",
    media_type="application/pdf",
    content_digest="ab" * 32,
)
[piece] = mapped.chunks
assert piece.content == "O imovel destina-se a uso comercial."
assert piece.pages == (1, 1)
assert piece.heading_path == ("CONTRATO",)

dsn, dimension = sys.argv[1], int(sys.argv[2])
with psycopg.connect(dsn) as conn:
    with conn.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS sem_modelo")
        cursor.execute("SET search_path TO sem_modelo, public")
        create_schema(conn, dimension=dimension)
    document_id = upsert_document(conn, external_id="doc-sem-modelo", title="Contrato")
    written = upsert_pieces(
        conn,
        document_id=document_id,
        pieces=[piece.to_piece([0.1] * dimension)],
        stamp=ModelStamp(model="m", dimension=dimension),
    )
    with conn.cursor() as cursor:
        cursor.execute("SELECT content, metadata FROM chunks")
        stored = cursor.fetchall()
        cursor.execute("DROP SCHEMA sem_modelo CASCADE")
    conn.commit()

assert written == 1, written
[(content, metadata)] = stored
assert content == "O imovel destina-se a uso comercial."
assert metadata["source_block_ids"] == ["p1"]
print(json.dumps({"ok": True, "kinds": metadata["block_kinds"]}))
"""
    from tests.conftest import TEST_DIMENSION

    finished = subprocess.run(
        [sys.executable, "-c", program, schema_dsn, str(TEST_DIMENSION)],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout.strip().splitlines()[-1]) == {
        "ok": True,
        "kinds": ["textBlock", "paragraph"],
    }
