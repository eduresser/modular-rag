
import psycopg
import pytest

from src.config import SearchSettings
from src.search import SearchRequest, search, search_on
from src.search_result import Failed, Found, FusionWeights, NothingMatched
from src.store import (
    DOCUMENT_STATE_GONE,
    DOCUMENT_STATE_INDEXED,
    ModelStamp,
    Piece,
    mark_document_gone,
    upsert_document,
    upsert_pieces,
)
from src.store_versions import activate_version
from tests.conftest import (
    TEST_DIMENSION,
    TEST_STAMP,
    ChunkSpec,
    DocumentSpec,
    direction,
    index_corpus,
)

CURRENT = ModelStamp(model="text-embedding-3", dimension=TEST_DIMENSION)
PREVIOUS = ModelStamp(model="text-embedding-ada-002", dimension=TEST_DIMENSION)

CASE_NUMBER = "0001234-56.2020.8.26.0100"

EARLY_TERMINATION = DocumentSpec(
    "rescisao",
    "Contrato de locação — Sala 402",
    (
        ChunkSpec(
            0,
            "Cláusula quinta: da rescisão antecipada do contrato pelo locatário",
            degrees=0,
            pages=(4, 5),
            headings=("Cláusula Quinta", "Rescisão antecipada"),
        ),
    ),
)
CASE_FILE = DocumentSpec(
    "processo",
    "Petição inicial",
    (
        ChunkSpec(
            0,
            f"O processo {CASE_NUMBER} tramita perante a 3ª Vara Cível",
            degrees=90,
            pages=(1, 1),
        ),
    ),
)
ADJUSTMENT = DocumentSpec(
    "reajuste",
    "Aditivo contratual",
    (ChunkSpec(0, "Do reajuste anual pelo índice acordado entre as partes", degrees=45, pages=(2, 2)),),
)

DEFAULTS = SearchSettings()


def _ask(conn, question, degrees, *, defaults=DEFAULTS, **overrides):
    request = SearchRequest(question=question, query_vector=direction(degrees), **overrides)
    return search_on(conn, request, defaults=defaults, stamp=TEST_STAMP)


def _documents(outcome):
    return [hit.provenance.external_id for hit in outcome.hits]


def _state_and_pieces(conn: psycopg.Connection, document_id: int) -> tuple[str, int]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT state FROM documents WHERE id = %s", (document_id,))
        (state,) = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        (count,) = cursor.fetchone()
    return state, int(count)


def test_ac_070(conn: psycopg.Connection):
    """AC-070: Both legs run and are fused @spec:AC-070"""
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    by_meaning = _ask(conn, "rescisão antecipada do contrato", degrees=0, result_count=1)

    assert isinstance(by_meaning, Found)
    assert _documents(by_meaning) == ["rescisao"]

    by_characters = _ask(conn, CASE_NUMBER, degrees=0, result_count=3)
    assert isinstance(by_characters, Found)
    assert "processo" in _documents(by_characters)
    case_file = next(hit for hit in by_characters.hits if hit.provenance.external_id == "processo")
    assert case_file.found_by == ("lexical",)

def test_ac_071(conn: psycopg.Connection):
    """AC-071: Fusion weights are settable per request @spec:AC-071"""
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    lexical_led = _ask(
        conn, CASE_NUMBER, degrees=0, result_count=3, weights=FusionWeights(dense=0.1, lexical=0.9)
    )
    dense_led = _ask(
        conn, CASE_NUMBER, degrees=0, result_count=3, weights=FusionWeights(dense=0.9, lexical=0.1)
    )

    assert _documents(lexical_led)[0] == "processo"
    assert _documents(dense_led)[0] == "rescisao"

    deployment = SearchSettings(dense_weight=0.25, lexical_weight=0.75)
    defaulted = _ask(conn, CASE_NUMBER, degrees=0, result_count=3, defaults=deployment)
    assert defaulted.diagnostics.weights == FusionWeights(dense=0.25, lexical=0.75)
    assert _documents(defaulted)[0] == "processo"

def test_ac_072(conn: psycopg.Connection):
    """AC-072: Results are ordered by a single comparable score @spec:AC-072"""
    index_corpus(
        conn,
        [
            DocumentSpec("ambos", "Ambos", (ChunkSpec(0, "rescisão antecipada", degrees=0, pages=(1, 1)),)),
            DocumentSpec("so_denso", "Só denso", (ChunkSpec(0, "foro de eleição", degrees=1, pages=(1, 1)),)),
            DocumentSpec("so_lexico", "Só léxico", (ChunkSpec(0, "rescisão antecipada", degrees=89, pages=(1, 1)),)),
        ],
    )

    outcome = _ask(conn, "rescisão antecipada", degrees=0, result_count=10)

    assert isinstance(outcome, Found)
    scores = [hit.score for hit in outcome.hits]
    assert all(isinstance(score, float) for score in scores)
    assert scores == sorted(scores, reverse=True)

    found_by = {hit.provenance.external_id: hit for hit in outcome.hits}
    assert set(found_by["ambos"].found_by) == {"dense", "lexical"}
    assert found_by["ambos"].score > found_by["so_denso"].score
    assert found_by["ambos"].score > found_by["so_lexico"].score
    best_single_leg = max(found_by["so_denso"].score, found_by["so_lexico"].score)
    assert found_by["ambos"].score > best_single_leg

def test_ac_073(conn: psycopg.Connection):
    """AC-073: Deleting a document and its pieces is one indivisible operation @spec:AC-073"""
    document_id = upsert_document(conn, external_id="ac-073", title="Contrato de locação")
    upsert_pieces(
        conn,
        document_id=document_id,
        stamp=CURRENT,
        pieces=[
            Piece(
                ordinal=0,
                content="Cláusula primeira",
                embedding=[1.0] + [0.0] * (TEST_DIMENSION - 1),
                page_start=1,
                page_end=1,
            ),
            Piece(
                ordinal=1,
                content="Cláusula segunda",
                embedding=[0.0, 1.0] + [0.0] * (TEST_DIMENSION - 2),
                page_start=2,
                page_end=2,
            ),
        ],
    )
    assert _state_and_pieces(conn, document_id) == (DOCUMENT_STATE_INDEXED, 2)

    assert mark_document_gone(conn, external_id="ac-073") is True

    assert _state_and_pieces(conn, document_id) == (DOCUMENT_STATE_GONE, 0)

    other_id = upsert_document(conn, external_id="ac-073-fails")
    upsert_pieces(
        conn,
        document_id=other_id,
        stamp=CURRENT,
        pieces=[
            Piece(
                ordinal=0,
                content="Cláusula única",
                embedding=[1.0] + [0.0] * (TEST_DIMENSION - 1),
                page_start=1,
                page_end=1,
            )
        ],
    )
    with conn.cursor() as cursor:
        cursor.execute(
            "CREATE FUNCTION pg_temp.fail_on_delete() RETURNS trigger AS "
            "$$ BEGIN RAISE EXCEPTION 'injected failure'; END; $$ LANGUAGE plpgsql"
        )
        cursor.execute(
            "CREATE TRIGGER fail_on_delete BEFORE DELETE ON chunks "
            "FOR EACH ROW EXECUTE FUNCTION pg_temp.fail_on_delete()"
        )
    with pytest.raises(psycopg.errors.RaiseException):
        mark_document_gone(conn, external_id="ac-073-fails")
    assert _state_and_pieces(conn, other_id) == (DOCUMENT_STATE_INDEXED, 1)

def test_ac_074(conn: psycopg.Connection):
    """AC-074: A search never returns a piece of a document that is gone @spec:AC-074"""
    index_corpus(
        conn,
        [
            EARLY_TERMINATION,
            DocumentSpec(
                "removido",
                "Contrato rescindido",
                (ChunkSpec(0, "cláusula de confidencialidade perpétua", degrees=0, pages=(1, 1)),),
                gone=True,
            ),
        ],
    )

    by_meaning = _ask(conn, "confidencialidade", degrees=0, result_count=10)
    by_characters = _ask(conn, "confidencialidade perpétua", degrees=90, result_count=10)

    assert "removido" not in _documents(by_meaning)
    for outcome in (by_meaning, by_characters):
        assert all(hit.provenance.external_id != "removido" for hit in outcome.hits)

def test_ac_075(conn: psycopg.Connection):
    """AC-075: A search never returns a piece from an inactive model version @spec:AC-075"""
    piece = (ChunkSpec(0, "cláusula de rescisão antecipada", degrees=0, pages=(1, 1)),)
    index_corpus(conn, [DocumentSpec("geracao_anterior", "Anterior", piece)], stamp=PREVIOUS)
    index_corpus(conn, [DocumentSpec("geracao_atual", "Atual", piece)], stamp=CURRENT)
    activate_version(conn, CURRENT)

    outcome = search_on(
        conn,
        SearchRequest(question="rescisão antecipada", query_vector=direction(0), result_count=10),
        defaults=DEFAULTS,
    )

    assert isinstance(outcome, Found)
    assert _documents(outcome) == ["geracao_atual"]

    activate_version(conn, PREVIOUS)
    rolled_back = search_on(
        conn,
        SearchRequest(question="rescisão antecipada", query_vector=direction(0), result_count=10),
        defaults=DEFAULTS,
    )
    assert _documents(rolled_back) == ["geracao_anterior"]

def test_ac_076(conn: psycopg.Connection):
    """AC-076: An empty match is an explicit empty result @spec:AC-076"""
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    outcome = _ask(conn, "xilografia bizantina", degrees=200, result_count=10)

    assert isinstance(outcome, NothingMatched)
    assert outcome.succeeded is True
    assert outcome.hits == ()

    assert not isinstance(outcome, Failed)
    assert hasattr(outcome, "hits")
    assert outcome.diagnostics.dense_candidates == 0
    assert outcome.diagnostics.lexical_candidates == 0

def test_ac_077(postgres_dsn: str):
    """AC-077: A failure is an explicit failure @spec:AC-077"""
    unreachable = "postgresql://postgres@127.0.0.1:1/postgres?connect_timeout=1"

    outcome = search(
        unreachable,
        SearchRequest(question="rescisão antecipada", query_vector=direction(0), result_count=10),
        defaults=DEFAULTS,
    )

    assert isinstance(outcome, Failed)
    assert outcome.succeeded is False
    assert outcome.reason.strip()
    assert "archive" in outcome.reason

    assert not hasattr(outcome, "hits")

    reachable = search(
        postgres_dsn,
        SearchRequest(question="rescisão antecipada", query_vector=direction(0), result_count=10),
        defaults=DEFAULTS,
    )
    assert not (isinstance(reachable, Failed) and reachable.reason == outcome.reason)

def test_ac_078(conn: psycopg.Connection):
    """AC-078: Index search effort is settable and recorded @spec:AC-078"""
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])
    deployment = SearchSettings(index_search_effort=137)

    outcome = _ask(conn, "rescisão antecipada", degrees=0, result_count=3, defaults=deployment)

    with conn.cursor() as cursor:
        cursor.execute("SELECT current_setting('hnsw.ef_search')")
        assert cursor.fetchone()[0] == "137"

    assert outcome.diagnostics.index_search_effort == 137
    assert outcome.diagnostics.duration_ms > 0

    overridden = _ask(
        conn, "rescisão antecipada", degrees=0, result_count=3, defaults=deployment, index_search_effort=64
    )
    assert overridden.diagnostics.index_search_effort == 64
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_setting('hnsw.ef_search')")
        assert cursor.fetchone()[0] == "64"

def test_ac_079(conn: psycopg.Connection):
    """AC-079: Retrieval depth and returned count are separate @spec:AC-079"""
    index_corpus(
        conn,
        [
            DocumentSpec(
                f"doc-{index}",
                f"Contrato {index}",
                (ChunkSpec(0, f"cláusula de rescisão número {index}", degrees=index, pages=(1, 1)),),
            )
            for index in range(8)
        ],
    )

    outcome = _ask(conn, "cláusula de rescisão", degrees=0, result_count=2, candidate_depth=8)

    assert isinstance(outcome, Found)
    assert outcome.diagnostics.candidate_depth == 8
    assert outcome.diagnostics.dense_candidates > 2
    assert outcome.diagnostics.lexical_candidates > 2

    assert len(outcome.hits) == 2
    assert len(_ask(conn, "cláusula de rescisão", degrees=0, result_count=1).hits) == 1
    assert len(_ask(conn, "cláusula de rescisão", degrees=0, result_count=50, candidate_depth=50).hits) == 8


def test_p_002(conn: psycopg.Connection):
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    outcome = _ask(conn, "rescisão antecipada do contrato", degrees=0, result_count=10)

    assert isinstance(outcome, Found)
    assert outcome.hits
    for hit in outcome.hits:
        assert hit.provenance.external_id
        assert hit.provenance.document_id > 0
        located = hit.provenance.page_label() or hit.provenance.heading_path
        assert located, f"piece {hit.chunk_id} came back with nothing a lawyer could check"
        assert hit.provenance.citation().strip()

    clause = next(hit for hit in outcome.hits if hit.provenance.external_id == "rescisao")
    assert clause.provenance.page_label() == "pp. 4-5"
    assert clause.provenance.heading_path == ("Cláusula Quinta", "Rescisão antecipada")

def test_p_006(conn: psycopg.Connection, postgres_dsn: str):
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    matched_nothing = _ask(conn, "xilografia bizantina", degrees=200, result_count=10)
    could_not_run = search(
        "postgresql://postgres@127.0.0.1:1/postgres?connect_timeout=1",
        SearchRequest(question="rescisão antecipada", query_vector=direction(0), result_count=10),
        defaults=DEFAULTS,
    )

    assert matched_nothing.succeeded is True
    assert could_not_run.succeeded is False
    assert type(matched_nothing) is not type(could_not_run)

    assert matched_nothing.hits == ()
    assert not hasattr(could_not_run, "hits")
    assert could_not_run.reason.strip()

def test_p_013(conn: psycopg.Connection):
    index_corpus(conn, [EARLY_TERMINATION, CASE_FILE, ADJUSTMENT])

    outcome = _ask(conn, CASE_NUMBER, degrees=0, result_count=10)

    assert isinstance(outcome, Found)
    documents = _documents(outcome)
    assert "processo" in documents, "the lexical leg did not run"
    assert "rescisao" in documents, "the dense leg did not run"
    assert outcome.diagnostics.dense_candidates > 0
    assert outcome.diagnostics.lexical_candidates > 0

    legs_used = {leg for hit in outcome.hits for leg in hit.found_by}
    assert legs_used == {"dense", "lexical"}
