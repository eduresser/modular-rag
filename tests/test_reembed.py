import pathlib

import pytest

from src.embedding.reembed import (
    ReembedError,
    cutover,
    group_by_document,
    reembed,
    stored_pieces,
)
from src.store import ModelStamp
from src.store_versions import activate_version, current_version
from tests.conftest import (
    TEST_DIMENSION,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    FakeEmbeddings,
    index_corpus,
    vector_for,
)

TARGET = ModelStamp(model="qwen/qwen3-embedding-4b", dimension=TEST_DIMENSION)

CORPUS = [
    DocumentSpec(
        external_id="doc-1",
        title="Contrato de locação",
        chunks=(
            ChunkSpec(0, "cláusula primeira", pages=(1, 2), headings=("Cláusulas", "Primeira")),
            ChunkSpec(1, "cláusula segunda", pages=(3, 3), headings=("Cláusulas", "Segunda")),
        ),
    ),
    DocumentSpec(
        external_id="doc-2",
        title="Aditivo",
        chunks=(ChunkSpec(0, "prazo de vigência", pages=(1, 1)),),
    ),
]


def _rows(conn, stamp: ModelStamp) -> list[tuple]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, document_id, ordinal, content, page_start, page_end, heading_path
              FROM chunks
             WHERE embed_model = %s AND embed_dimension = %s
             ORDER BY document_id, ordinal
            """,
            (stamp.model, stamp.dimension),
        )
        return cursor.fetchall()


def test_the_previous_generation_is_left_exactly_where_it_was(conn):
    index_corpus(conn, CORPUS)
    before = _rows(conn, TEST_STAMP)
    provider = FakeEmbeddings()

    reembed(conn, source=TEST_STAMP, target=TARGET, embedder=provider.embedder(model=TARGET.model), batch_size=4, max_inputs=2048)

    assert _rows(conn, TEST_STAMP) == before
    assert len(_rows(conn, TARGET)) == 3


def test_the_provenance_travels_into_the_new_rows(conn):
    index_corpus(conn, CORPUS)
    provider = FakeEmbeddings()

    reembed(conn, source=TEST_STAMP, target=TARGET, embedder=provider.embedder(model=TARGET.model), batch_size=4, max_inputs=2048)

    old = [row[2:] for row in _rows(conn, TEST_STAMP)]
    new = [row[2:] for row in _rows(conn, TARGET)]
    assert new == old


def test_the_new_vectors_are_the_ones_the_provider_returned(conn):
    index_corpus(conn, CORPUS)
    provider = FakeEmbeddings()

    reembed(conn, source=TEST_STAMP, target=TARGET, embedder=provider.embedder(model=TARGET.model), batch_size=4, max_inputs=2048)

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT content, embedding::text FROM chunks
             WHERE embed_model = %s ORDER BY document_id, ordinal
            """,
            (TARGET.model,),
        )
        rows = cursor.fetchall()
    for content, stored in rows:
        components = [float(part) for part in stored.strip("[]").split(",")]
        assert components == pytest.approx(list(vector_for(content)), abs=1e-3)


def test_re_embedding_does_not_start_serving_the_new_vectors(conn):
    index_corpus(conn, CORPUS)
    activate_version(conn, TEST_STAMP)

    reembed(conn, source=TEST_STAMP, target=TARGET, embedder=FakeEmbeddings(model=TARGET.model).embedder(model=TARGET.model), batch_size=4, max_inputs=2048)

    assert current_version(conn) == TEST_STAMP


def test_the_cutover_moves_the_pointer_and_hands_back_the_one_to_roll_back_to(conn):
    index_corpus(conn, CORPUS)
    activate_version(conn, TEST_STAMP)
    reembed(conn, source=TEST_STAMP, target=TARGET, embedder=FakeEmbeddings(model=TARGET.model).embedder(model=TARGET.model), batch_size=4, max_inputs=2048)

    previous = cutover(conn, TARGET)

    assert previous == TEST_STAMP
    assert current_version(conn) == TARGET
    assert cutover(conn, previous) == TARGET
    assert current_version(conn) == TEST_STAMP


def test_a_cutover_to_a_generation_nobody_wrote_is_refused(conn):
    index_corpus(conn, CORPUS)

    with pytest.raises(ReembedError) as excinfo:
        cutover(conn, TARGET)

    assert "no pieces" in str(excinfo.value)


def test_a_document_whose_batch_failed_is_not_written_and_is_named(conn):
    index_corpus(conn, CORPUS)
    provider = FakeEmbeddings(fail_batches=(0,))

    outcome = reembed(
        conn, source=TEST_STAMP, target=TARGET, embedder=provider.embedder(model=TARGET.model), batch_size=2, max_inputs=2048
    )

    doc_one = {row[1] for row in _rows(conn, TEST_STAMP)} - {row[1] for row in _rows(conn, TARGET)}
    assert outcome.failed_documents == tuple(doc_one)
    assert outcome.complete is False
    assert outcome.run.unembedded == 2
    assert outcome.pieces_written == 1


def test_re_embedding_a_stamp_into_itself_is_refused(conn):
    index_corpus(conn, CORPUS)

    with pytest.raises(ReembedError) as excinfo:
        reembed(
            conn,
            source=TEST_STAMP,
            target=TEST_STAMP,
            embedder=FakeEmbeddings(model=TEST_STAMP.model).embedder(model=TEST_STAMP.model),
            batch_size=4, max_inputs=2048,
        )

    assert "into itself" in str(excinfo.value)


def test_an_embedder_that_is_not_the_target_stamp_is_refused(conn):
    index_corpus(conn, CORPUS)
    provider = FakeEmbeddings()

    with pytest.raises(ReembedError) as excinfo:
        reembed(
            conn,
            source=TEST_STAMP,
            target=ModelStamp(model="google/gemini-embedding-001", dimension=TEST_DIMENSION),
            embedder=provider.embedder(model=TARGET.model),
            batch_size=4, max_inputs=2048,
        )

    assert "gemini" in str(excinfo.value)

    with pytest.raises(ReembedError):
        reembed(
            conn,
            source=TEST_STAMP,
            target=ModelStamp(model=TARGET.model, dimension=TEST_DIMENSION + 1),
            embedder=provider.embedder(model=TARGET.model),
            batch_size=4, max_inputs=2048,
        )


def test_re_embedding_a_stamp_nothing_was_stored_under_is_refused(conn):
    with pytest.raises(ReembedError) as excinfo:
        reembed(
            conn,
            source=ModelStamp(model="nobody/wrote-this", dimension=TEST_DIMENSION),
            target=TARGET,
            embedder=FakeEmbeddings(model=TARGET.model).embedder(model=TARGET.model),
            batch_size=4, max_inputs=2048,
        )

    assert "nothing to embed again" in str(excinfo.value)


def test_only_the_documents_asked_for_are_re_embedded(conn):
    written = index_corpus(conn, CORPUS)

    outcome = reembed(
        conn,
        source=TEST_STAMP,
        target=TARGET,
        embedder=FakeEmbeddings(model=TARGET.model).embedder(model=TARGET.model),
        batch_size=4, max_inputs=2048,
        document_ids=[written["doc-2"]],
    )

    assert outcome.documents_written == (written["doc-2"],)
    assert len(_rows(conn, TARGET)) == 1


def test_the_pieces_come_back_grouped_by_document_and_in_their_stored_order(conn):
    index_corpus(conn, CORPUS)

    groups = group_by_document(stored_pieces(conn, TEST_STAMP))

    assert [group.texts for group in groups] == [
        ("cláusula primeira", "cláusula segunda"),
        ("prazo de vigência",),
    ]


def test_nothing_here_can_reach_the_source_or_the_reader():
    source = pathlib.Path("src/embedding/reembed.py").read_text()

    assert "src.sources" not in source
    assert "src.processing" not in source
