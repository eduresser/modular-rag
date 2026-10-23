
def test_ac_210():
    """AC-210: What reaches the reader is a format the reader can structure @spec:AC-210"""
    from src.models import VersionStamps
    from src.processing.docai import chunks_from_document
    from src.processing.render import MEDIA_PDF, docai_rendering_for, render
    from src.sources.gdrive_export import KIND_DOCUMENT, MEDIA_DOCX, export_plan
    from tests.conftest import DOCAI_VERSION, reader_chunk, reader_payload, text_block
    from tests.test_render import real_word_document

    assert export_plan(KIND_DOCUMENT).media_type == MEDIA_PDF
    assert docai_rendering_for(MEDIA_PDF) == ""

    assert docai_rendering_for(MEDIA_DOCX) == MEDIA_PDF

    rendered = render(
        "contrato-1", MEDIA_DOCX, real_word_document(), to=docai_rendering_for(MEDIA_DOCX)
    )

    assert rendered.media_type == MEDIA_PDF
    assert rendered.content.startswith(b"%PDF")
    assert docai_rendering_for(rendered.media_type) == ""
    assert (
        render(
            "contrato-1",
            rendered.media_type,
            rendered.content,
            to=docai_rendering_for(rendered.media_type),
        ).content
        == rendered.content
    )

    stamps = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")
    text = "cláusula redigida pela própria firma"
    mapped = chunks_from_document(
        reader_payload(
            [text_block("p1", text, pages=(1, 1))],
            [reader_chunk("c1", text, ["p1"])],
        ),
        external_id="contrato-1",
        stamps=stamps,
    )
    assert mapped.chunks[0].page_start == 1
    assert mapped.chunks[0].page_end == 1
    assert mapped.unrecognised_blocks == ()

def test_ac_210_through_the_pipeline(monkeypatch, conn):
    """AC-210: What reaches the reader is a format the reader can structure @spec:AC-210"""
    from src.ingest import ingest_documents
    from src.processing.render import MEDIA_PDF
    import hashlib

    from src.sources.gdrive_export import MEDIA_DOCX
    from tests.conftest import DriveFile, contract_payload, ingest_world
    from tests.test_render import real_word_document

    content = real_word_document()
    files = [
        DriveFile(
            id="doc-docx",
            name="Contrato.docx",
            mimeType=MEDIA_DOCX,
            md5Checksum=hashlib.md5(content).hexdigest(),
            size=len(content),
            parents=("root-folder",),
            content=content,
        )
    ]
    outputs = {"doc-docx": contract_payload(clauses=("Cláusula primeira.",))}
    world = ingest_world(monkeypatch, files=files, outputs=outputs)

    ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))

    assert world.docai.submissions, "the reader was asked for nothing at all"
    media_types = {
        document.get("mimeType")
        for body in world.docai.submissions
        for document in body["inputDocuments"]["gcsDocuments"]["documents"]
    }
    assert media_types == {MEDIA_PDF}, (
        f"the reader was handed {media_types}; the word-processor bytes reaching it is "
        "the case AC-210 exists to prevent, and a sweep on the real corpus hit it"
    )


def test_ac_211(monkeypatch):
    """AC-211: Changing what the reader is given re-reads what was given before @spec:AC-211"""
    from src.fingerprint import STAGE_CUT, STAGE_EMBED, STAGE_READ, stage_fingerprints
    from tests.test_fingerprint import _valid_config

    reference = stage_fingerprints(_valid_config())

    monkeypatch.setattr(
        "src.fingerprint.export_plan",
        lambda kind: type("Plan", (), {"media_type": "application/msword"})(),
    )
    changed_target = stage_fingerprints(_valid_config())

    assert changed_target[STAGE_READ] != reference[STAGE_READ]
    assert changed_target[STAGE_CUT] == reference[STAGE_CUT]
    assert changed_target[STAGE_EMBED] == reference[STAGE_EMBED]
    monkeypatch.undo()

    monkeypatch.setattr("src.fingerprint.renderer_version", lambda: "a-different-suite-version")
    changed_renderer = stage_fingerprints(_valid_config())

    assert changed_renderer[STAGE_READ] != reference[STAGE_READ]
    assert changed_renderer[STAGE_CUT] == reference[STAGE_CUT]
    assert changed_renderer[STAGE_EMBED] == reference[STAGE_EMBED]


def test_ac_211_the_corpus_is_actually_read_again(monkeypatch, conn):
    """AC-211: Changing what the reader is given re-reads what was given before @spec:AC-211"""
    from src.ingest import ingest_documents
    from tests.conftest import contract_payload, ingest_world, pdf_file

    files = [pdf_file("doc-1", content=b"%PDF contrato")]
    outputs = {"doc-1": contract_payload(clauses=("Cláusula primeira.", "Cláusula segunda."))}

    first = ingest_world(monkeypatch, files=files, outputs=outputs)
    ingest_documents(conn, first.pipeline, list(first.pipeline.source.sweep()))

    moved = ingest_world(
        monkeypatch, files=files, outputs=outputs, storage=first.storage
    )
    monkeypatch.setattr("src.fingerprint.renderer_version", lambda: "another-suite-version")
    run = ingest_documents(conn, moved.pipeline, list(moved.pipeline.source.sweep()))

    actions = {result.external_id: result.action for result in run.results}
    assert "doc-1" in actions, f"the document was not processed at all: {run.results}"
    assert actions["doc-1"] != "touch", (
        "the document was touched rather than read, so a change to what the reader "
        "is given did not reach it"
    )


def test_ac_212(conn):
    """AC-212: The stored artifact carries no image binaries @spec:AC-212"""
    import base64

    from src.processing.artifact import load_parsed_artifact, store_parsed_artifact
    from src.store import upsert_document
    from tests.conftest import DOCAI_PROCESSOR, DOCAI_VERSION, FakeObjectStorage, reader_chunk, reader_payload

    binary = b"\x89PNG\r\n\x1a\nthis is not a real image but stands in for one"
    encoded = base64.b64encode(binary).decode()
    image_block = {
        "blockId": "img-1",
        "pageSpan": {"pageStart": 2, "pageEnd": 2},
        "imageBlock": {
            "image": {"content": encoded, "mimeType": "image/png"},
            "annotations": {"description": "Planta baixa do imóvel locado."},
            "imageText": "ÁREA TOTAL: 78,4 m²",
        },
    }
    payload = reader_payload([image_block], [reader_chunk("c1", "", ["img-1"])])
    storage = FakeObjectStorage()
    document_id = upsert_document(conn, external_id="doc-imagem", title="Contrato")

    artifact = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id="doc-imagem",
        payload=payload, reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )

    stored_bytes = storage.objects[storage.store().name_of(artifact.uri)]
    assert encoded not in stored_bytes.decode()
    kept = load_parsed_artifact(conn, storage.store(), artifact)
    kept_image_block = kept["document"]["documentLayout"]["blocks"][0]["imageBlock"]
    assert "image" not in kept_image_block

def test_ac_213(conn):
    """AC-213: What the reader said about an image still reaches the piece @spec:AC-213"""
    from src.models import VersionStamps
    from src.processing.artifact import load_parsed_artifact, store_parsed_artifact
    from src.processing.docai import chunks_from_document
    from src.store import upsert_document
    from tests.conftest import DOCAI_PROCESSOR, DOCAI_VERSION, FakeObjectStorage, reader_chunk, reader_payload

    def _image_response(*, with_binary: bool) -> dict:
        inner = {
            "annotations": {"description": "Planta baixa do imóvel locado."},
            "imageText": "ÁREA TOTAL: 78,4 m²",
        }
        if with_binary:
            inner["image"] = {"content": "Zm9v", "mimeType": "image/png"}
        image_block = {
            "blockId": "img-1",
            "pageSpan": {"pageStart": 2, "pageEnd": 2},
            "imageBlock": inner,
        }
        return reader_payload([image_block], [reader_chunk("c1", "", ["img-1"])])

    stamps = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")
    storage = FakeObjectStorage()
    document_id = upsert_document(conn, external_id="doc-imagem", title="Contrato")
    artifact_with_binary = store_parsed_artifact(
        conn, storage.store(), document_id=document_id, external_id="doc-imagem",
        payload=_image_response(with_binary=True), reader=DOCAI_PROCESSOR, reader_version=DOCAI_VERSION,
    )
    kept_with_binary = load_parsed_artifact(conn, storage.store(), artifact_with_binary)
    mapped_with_binary = chunks_from_document(
        kept_with_binary, external_id="doc-imagem", stamps=stamps
    )

    mapped_without_binary = chunks_from_document(
        _image_response(with_binary=False), external_id="doc-imagem", stamps=stamps
    )

    piece = mapped_with_binary.chunks[0]
    assert "Planta baixa do imóvel locado." in piece.content
    assert "ÁREA TOTAL: 78,4 m²" in piece.content
    assert mapped_with_binary.chunks == mapped_without_binary.chunks
    assert len(mapped_with_binary.chunks) == 1


def test_ac_214():
    """AC-214: A word the reader substituted letters in is searchable by what it says @spec:AC-214"""
    from src.models import VersionStamps
    from src.processing.docai import chunks_from_document
    from src.processing.confusables import CONFUSABLES, UNICODE_VERSION
    from tests.conftest import DOCAI_VERSION, reader_chunk, reader_payload, text_block

    text = (
        "Ver ΑΝΕΧΟ I do OВЈЕТО. СЕР.: 20040-020. "
        "Tolerância de μm, variação Δt e desvio σ², conforme ΑΝΑΓΚΗ."
    )
    stamps = VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1")

    mapped = chunks_from_document(
        reader_payload(
            [text_block("p1", text, pages=(1, 1))],
            [reader_chunk("c1", text, ["p1"])],
        ),
        external_id="doc-1",
        stamps=stamps,
    )
    content = " ".join(chunk.content for chunk in mapped.chunks)

    assert "ANEXO" in content
    assert "OBJETO" in content
    assert "CEP." in content
    assert "ΑΝΕΧΟ" not in content
    assert "OВЈЕТО" not in content
    assert "СЕР." not in content

    assert "μm" in content
    assert "Δt" in content
    assert "σ²" in content
    assert "ΑΝΑΓΚΗ" in content

    assert UNICODE_VERSION
    assert CONFUSABLES["Α"] == "A" and CONFUSABLES["С"] == "C"
    assert "α" not in CONFUSABLES and "σ" not in CONFUSABLES


def test_ac_215():
    """AC-215: A piece records the kind of block it was cut from @spec:AC-215"""
    from src.models import METADATA_BLOCK_KINDS, VersionStamps
    from src.processing.docai import chunks_from_document
    from tests.conftest import (
        DOCAI_VERSION,
        reader_chunk,
        reader_payload,
        table_block,
        text_block,
    )

    def image_block(block_id, description, *, pages=(1, 1)):
        return {
            "blockId": str(block_id),
            "pageSpan": {"pageStart": pages[0], "pageEnd": pages[1]},
            "imageBlock": {"annotations": {"description": description}},
        }

    blocks = [
        text_block("h1", "CLÁUSULA QUINTA — DO VALOR", pages=(1, 1), kind="heading-1"),
        text_block("p1", "O valor estimado da contratação é o do quadro abaixo.", pages=(1, 1)),
        table_block("t1", ["Item", "Valor"], [["Supervisão", "R$ 1.000,00"]], pages=(1, 1)),
        image_block("i1", "The image displays a logo.", pages=(1, 1)),
    ]

    mapped = chunks_from_document(
        reader_payload(
            blocks,
            [
                reader_chunk("c1", "CLÁUSULA QUINTA — DO VALOR", ["h1"]),
                reader_chunk("c2", "O valor estimado da contratação é o do quadro abaixo.", ["p1"]),
                reader_chunk("c3", "Item | Valor", ["t1"]),
                reader_chunk("c4", "The image displays a logo.", ["i1"]),
                reader_chunk("c5", "CLÁUSULA QUINTA — DO VALOR Item | Valor", ["h1", "t1"]),
            ],
        ),
        external_id="doc-1",
        stamps=VersionStamps(reader_version=DOCAI_VERSION, fingerprint="fp-1"),
    )
    kinds = [tuple(piece.to_piece([0.0]).metadata[METADATA_BLOCK_KINDS]) for piece in mapped.chunks]

    assert any("heading-1" in k for k in kinds), kinds
    assert any("paragraph" in k for k in kinds), kinds
    assert any("tableBlock" in k for k in kinds), kinds
    assert any("imageBlock" in k for k in kinds), kinds

    spanning = kinds[-1]
    assert "tableBlock" in spanning and len(spanning) > 1, spanning
