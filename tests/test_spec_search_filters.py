
import pytest

def test_ac_275(conn):
    """AC-275: A filtered search answers only from what the filter selected @spec:AC-275"""
    from src.config import SearchSettings
    from src.filters import validate_filter
    from src.search import SearchRequest, search_on
    from src.store_versions import activate_version
    from tests.conftest import TEST_STAMP, ChunkSpec, DocumentSpec, direction, index_corpus

    index_corpus(
        conn,
        [
            DocumentSpec("near", "Perto", (ChunkSpec(0, "rescisão antecipada", degrees=0, pages=(1, 1)),)),
            DocumentSpec("far", "Longe", (ChunkSpec(0, "rescisão eventual", degrees=45, pages=(1, 1)),)),
        ],
    )
    activate_version(conn, TEST_STAMP)
    defaults = SearchSettings()

    unfiltered = search_on(
        conn, SearchRequest(question="rescisão", query_vector=direction(0), result_count=10), defaults=defaults
    )
    assert unfiltered.hits[0].provenance.external_id == "near"

    node = validate_filter({"field": "document_id", "op": "in", "value": ["far"]})
    filtered = search_on(
        conn,
        SearchRequest(question="rescisão", query_vector=direction(0), filter=node, result_count=10),
        defaults=defaults,
    )

    assert {hit.provenance.external_id for hit in filtered.hits} == {"far"}
    assert all(hit.provenance.external_id != "near" for hit in filtered.hits)

def test_ac_276(conn):
    """AC-276: The filter narrows the candidates before they are ranked @spec:AC-276"""
    from src.filters import validate_filter
    from src.search_dense import dense_candidates
    from src.search_lexical import lexical_candidates
    from tests.conftest import ChunkSpec, DocumentSpec, TEST_STAMP, direction, index_corpus

    noise_docs = [
        DocumentSpec(
            f"noise-{i}", f"Noise {i}",
            (ChunkSpec(0, "rescisão rescisão rescisão do contrato", degrees=0, pages=(1, 1)),),
        )
        for i in range(5)
    ]
    target_docs = [
        DocumentSpec(
            f"target-{i}", f"Target {i}",
            (ChunkSpec(0, "rescisão eventual do contrato", degrees=45, pages=(1, 1)),),
        )
        for i in range(2)
    ]
    index_corpus(conn, noise_docs + target_docs)
    target_ids = {d.external_id for d in target_docs}

    node = validate_filter({"field": "document_id", "op": "in", "value": sorted(target_ids)})

    dense = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=2, index_search_effort=40, filter_=node
    )
    lexical = lexical_candidates(conn, question="rescisão", stamp=TEST_STAMP, depth=2, filter_=node)

    assert len(dense) == 2
    assert {c.provenance.external_id for c in dense} == target_ids
    assert len(lexical) == 2
    assert {c.provenance.external_id for c in lexical} == target_ids

def test_ac_277(conn):
    """AC-277: A narrow filter returns as many results as a wide one @spec:AC-277"""
    import random

    from src.config import SearchSettings
    from src.db import halfvec_literal
    from src.filters import validate_filter
    from src.search_dense import dense_candidates
    from tests.conftest import TEST_STAMP, direction

    n = 5000
    target_ordinals = set(range(0, n, 500))
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO documents (external_id, title) VALUES ('recall-corpus', 'Recall') RETURNING id"
        )
        (document_id,) = cursor.fetchone()
        rng = random.Random(277)
        rows = [
            (
                document_id,
                i,
                f"lorem {'aditivo' if i in target_ordinals else 'ipsum'} dolor sit amet",
                TEST_STAMP.model,
                TEST_STAMP.dimension,
                halfvec_literal(direction(rng.uniform(0, 85))),
            )
            for i in range(n)
        ]
        cursor.executemany(
            """
            INSERT INTO chunks (document_id, ordinal, content, embed_model, embed_dimension,
                                 embedding, page_start, page_end, heading_path)
            VALUES (%s, %s, %s, %s, %s, %s::halfvec, 1, 1, ARRAY['H'])
            """,
            rows,
        )
        cursor.execute("ANALYZE chunks")
        cursor.execute("SET enable_seqscan = off")

    defaults = SearchSettings()
    depth = 8

    wide = validate_filter({"field": "content", "op": "contains", "value": "lorem"})
    narrow = validate_filter({"field": "content", "op": "contains", "value": "aditivo"})

    wide_result = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=depth,
        index_search_effort=defaults.index_search_effort, filter_=wide,
        filter_timeout_ms=defaults.filter_timeout_ms,
    )
    narrow_result = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=depth,
        index_search_effort=defaults.index_search_effort, filter_=narrow,
        filter_timeout_ms=defaults.filter_timeout_ms,
    )

    assert len(narrow_result) == depth
    assert len(narrow_result) == len(wide_result)

def test_ac_278():
    """AC-278: A field the archive does not hold is refused by name @spec:AC-278"""
    from src.filters import FilterError, validate_filter

    with pytest.raises(FilterError) as excinfo:
        validate_filter(
            {"all": [{"field": "landlord_name", "op": "is", "value": "Acme"}]}
        )

    assert "landlord_name" in excinfo.value.reason
    assert excinfo.value.path == "<filter>.all[0]"


def test_ac_279():
    """AC-279: An operator a field does not take is refused naming both @spec:AC-279"""
    from src.filters import FIELDS, OPERATORS, FilterError, validate_filter

    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "page", "op": "glob", "value": "*29*"})
    assert "page" in excinfo.value.reason
    assert FIELDS["page"].type in excinfo.value.reason
    for allowed in OPERATORS[FIELDS["page"].type]:
        assert allowed in excinfo.value.reason

    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "document_id", "op": "gt", "value": 5})
    assert "document_id" in excinfo.value.reason
    assert FIELDS["document_id"].type in excinfo.value.reason
    for allowed in OPERATORS[FIELDS["document_id"].type]:
        assert allowed in excinfo.value.reason

    with pytest.raises(FilterError) as excinfo:
        validate_filter({"field": "size_bytes", "op": "is", "value": "twelve"})
    assert "size_bytes" in excinfo.value.reason

def test_ac_280(conn):
    """AC-280: A page filter keeps the piece that straddles the boundary @spec:AC-280"""
    from src.filters import validate_filter
    from src.search_dense import dense_candidates
    from tests.conftest import ChunkSpec, DocumentSpec, TEST_STAMP, direction, index_corpus

    index_corpus(
        conn,
        [
            DocumentSpec("straddle", "Straddle", (ChunkSpec(0, "cláusula", degrees=0, pages=(29, 31)),)),
            DocumentSpec("inside", "Inside", (ChunkSpec(0, "cláusula", degrees=0, pages=(35, 40)),)),
            DocumentSpec("before", "Before", (ChunkSpec(0, "cláusula", degrees=0, pages=(1, 5)),)),
            DocumentSpec("after", "After", (ChunkSpec(0, "cláusula", degrees=0, pages=(60, 70)),)),
        ],
    )
    node = validate_filter({"field": "page", "op": "between", "value": [30, 50]})

    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40, filter_=node
    )
    ids = {c.provenance.external_id for c in candidates}

    assert "straddle" in ids
    assert "inside" in ids
    assert "before" not in ids
    assert "after" not in ids

def test_ac_281(conn):
    """AC-281: A path filter selects a section and everything under it @spec:AC-281"""
    from src.filters import validate_filter
    from src.search_dense import dense_candidates
    from tests.conftest import ChunkSpec, DocumentSpec, TEST_STAMP, direction, index_corpus

    index_corpus(
        conn,
        [
            DocumentSpec("section", "Section", (ChunkSpec(0, "texto", degrees=0, headings=("Cláusula 5",)),)),
            DocumentSpec(
                "nested", "Nested",
                (ChunkSpec(0, "texto", degrees=0, headings=("Cláusula 5", "Parágrafo 1")),),
            ),
            DocumentSpec(
                "string_prefix", "String prefix",
                (ChunkSpec(0, "texto", degrees=0, headings=("Cláusula 50",)),),
            ),
            DocumentSpec("unrelated", "Unrelated", (ChunkSpec(0, "texto", degrees=0, headings=("Cláusula 6",)),)),
        ],
    )
    node = validate_filter({"field": "heading_path", "op": "path_starts_with", "value": ["Cláusula 5"]})

    candidates = dense_candidates(
        conn, query_vector=direction(0), stamp=TEST_STAMP, depth=10, index_search_effort=40, filter_=node
    )
    ids = {c.provenance.external_id for c in candidates}

    assert ids == {"section", "nested"}
    assert "string_prefix" not in ids
    assert "unrelated" not in ids

def test_ac_282(conn, monkeypatch):
    """AC-282: A filter that selected no document is its own outcome @spec:AC-282"""
    from src.filters import validate_filter
    from src.search_result import FilterMatchedNothing, NothingMatched
    from src.store_versions import activate_version
    from src.usecases.search import SearchQuery, run_search
    from tests.conftest import (
        ChunkSpec,
        DocumentSpec,
        FakeQueryEmbedder,
        TEST_STAMP,
        api_config,
        index_corpus,
    )

    index_corpus(
        conn,
        [DocumentSpec("present", "Presente", (ChunkSpec(0, "rescisão antecipada", degrees=0, pages=(1, 1)),))],
    )
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")
    node = validate_filter({"field": "document_id", "op": "in", "value": ["ausente"]})

    outcome = run_search(
        SearchQuery(question="rescisão", filter=node),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )

    assert isinstance(outcome, FilterMatchedNothing)
    assert not isinstance(outcome, NothingMatched)
    assert outcome.succeeded
    assert outcome.diagnostics.documents_considered == 0
    assert outcome.diagnostics.pieces_considered == 0

    assert "document_id" in outcome.clause
    assert "ausente" in outcome.clause

    unrelated = run_search(
        SearchQuery(question="xilografia bizantina"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=200),
        connection=conn,
    )
    assert isinstance(unrelated, NothingMatched)
    assert not isinstance(unrelated, FilterMatchedNothing)

def test_ac_283(conn, monkeypatch):
    """AC-283: Every filtered answer says how much the filter let through @spec:AC-283"""
    from src.filters import validate_filter
    from src.search_result import NothingMatched
    from src.store_versions import activate_version
    from src.usecases.search import SearchQuery, run_search
    from tests.conftest import (
        ChunkSpec,
        DocumentSpec,
        FakeQueryEmbedder,
        TEST_STAMP,
        api_config,
        index_corpus,
    )

    index_corpus(
        conn,
        [
            DocumentSpec("near", "Perto", (ChunkSpec(0, "rescisão antecipada", degrees=0, pages=(1, 1)),)),
            DocumentSpec("far", "Longe", (ChunkSpec(0, "rescisão eventual", degrees=45, pages=(1, 1)),)),
        ],
    )
    activate_version(conn, TEST_STAMP)
    config = api_config(monkeypatch, "postgresql://localhost/unused")

    node = validate_filter({"field": "document_id", "op": "in", "value": ["far"]})
    found = run_search(
        SearchQuery(question="rescisão", filter=node),
        config=config,
        embedder=FakeQueryEmbedder(degrees=45),
        connection=conn,
    )

    assert found.diagnostics.documents_considered == 1
    assert found.diagnostics.pieces_considered == 1

    node_both = validate_filter({"field": "document_id", "op": "in", "value": ["near", "far"]})
    nothing = run_search(
        SearchQuery(question="xilografia bizantina", filter=node_both),
        config=config,
        embedder=FakeQueryEmbedder(degrees=200),
        connection=conn,
    )
    assert isinstance(nothing, NothingMatched)
    assert nothing.diagnostics.documents_considered == 2
    assert nothing.diagnostics.pieces_considered == 2

    unfiltered = run_search(
        SearchQuery(question="rescisão"),
        config=config,
        embedder=FakeQueryEmbedder(degrees=0),
        connection=conn,
    )
    assert unfiltered.diagnostics.documents_considered is None
    assert unfiltered.diagnostics.pieces_considered is None

def test_ac_284():
    """AC-284: A malformed filter is refused naming the clause that is wrong @spec:AC-284"""
    from src.filters import (
        DEFAULT_MAX_FILTER_CLAUSES,
        DEFAULT_MAX_FILTER_DEPTH,
        FilterError,
        validate_filter,
    )

    with pytest.raises(FilterError) as excinfo:
        validate_filter({"xor": [{"field": "title", "op": "is", "value": "x"}]})
    assert excinfo.value.path == "<filter>"
    assert "xor" in excinfo.value.reason

    with pytest.raises(FilterError) as excinfo:
        validate_filter({"all": [{"field": "title", "op": "is"}]})
    assert excinfo.value.path == "<filter>.all[0]"
    assert "value" in excinfo.value.reason

    node = {"field": "title", "op": "is", "value": "x"}
    for _ in range(DEFAULT_MAX_FILTER_DEPTH + 2):
        node = {"all": [node]}
    with pytest.raises(FilterError) as excinfo:
        validate_filter(node, max_depth=DEFAULT_MAX_FILTER_DEPTH)
    assert str(DEFAULT_MAX_FILTER_DEPTH) in excinfo.value.reason
    assert "max_filter_depth" in excinfo.value.reason

    clauses = [
        {"field": "title", "op": "is", "value": f"clause-{i}"}
        for i in range(DEFAULT_MAX_FILTER_CLAUSES + 1)
    ]
    with pytest.raises(FilterError) as excinfo:
        validate_filter({"all": clauses}, max_clauses=DEFAULT_MAX_FILTER_CLAUSES)
    assert str(DEFAULT_MAX_FILTER_CLAUSES) in excinfo.value.reason
    assert "max_filter_clauses" in excinfo.value.reason
    assert excinfo.value.path == f"<filter>.all[{DEFAULT_MAX_FILTER_CLAUSES}]"


def test_ac_285():
    """AC-285: A pattern longer than this deployment accepts is refused by name @spec:AC-285"""
    import src.filters as filters_module
    from src.filters import DEFAULT_MAX_PATTERN_LENGTH, FilterError, validate_filter

    compiled = {"called": False}
    original_translate = filters_module._glob_to_regex

    def _spy(pattern):
        compiled["called"] = True
        return original_translate(pattern)

    filters_module._glob_to_regex = _spy
    try:
        long_pattern = "a" * (DEFAULT_MAX_PATTERN_LENGTH + 1)
        with pytest.raises(FilterError) as excinfo:
            validate_filter({"field": "title", "op": "glob", "value": long_pattern})
    finally:
        filters_module._glob_to_regex = original_translate

    assert compiled["called"] is False
    assert str(DEFAULT_MAX_PATTERN_LENGTH) in excinfo.value.reason
    assert "max_pattern_length" in excinfo.value.reason

    validate_filter({"field": "title", "op": "glob", "value": "*aditivo*"})

def test_ac_286(sync_database):
    """AC-286: A filter that cannot finish in time is refused, never left running @spec:AC-286"""
    import psycopg

    from src.config import SearchSettings
    from src.db import halfvec_literal
    from src.filters import validate_filter
    from src.search import SearchRequest, search
    from src.search_result import Failed
    from src.store_versions import activate_version
    from tests.conftest import TEST_STAMP, direction

    n = 10_000
    with psycopg.connect(sync_database) as setup:
        with setup.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (external_id, title) VALUES ('slow', 'Slow') RETURNING id"
            )
            (document_id,) = cursor.fetchone()
            cursor.execute(
                """
                INSERT INTO chunks (document_id, ordinal, content, page_start, page_end,
                                     heading_path, embed_model, embed_dimension, embedding)
                SELECT %s, g, 'lorem ipsum dolor sit amet consectetur ' || g::text,
                       1, 1, ARRAY['H'], %s, %s, %s::halfvec
                  FROM generate_series(1, %s) AS g
                """,
                (document_id, TEST_STAMP.model, TEST_STAMP.dimension, halfvec_literal(direction(0)), n),
            )
        activate_version(setup, TEST_STAMP)
        setup.commit()

    try:
        node = validate_filter({"field": "content", "op": "regex", "value": "zzzzz{5,}"})
        request = SearchRequest(question="lorem", query_vector=direction(0), filter=node, result_count=10)
        defaults = SearchSettings(filter_timeout_ms=2)

        outcome = search(sync_database, request, defaults=defaults)

        assert isinstance(outcome, Failed)
        assert "2ms" in outcome.reason
        assert "filter_timeout_ms" in outcome.reason
        assert "content" in outcome.reason

        second = search(
            sync_database,
            SearchRequest(question="lorem", query_vector=direction(0), result_count=10),
            defaults=SearchSettings(),
        )
        assert second.succeeded
    finally:
        with psycopg.connect(sync_database) as cleanup:
            with cleanup.cursor() as cursor:
                cursor.execute("DELETE FROM documents WHERE external_id = 'slow'")
            cleanup.commit()
def test_ac_287(conn):
    """AC-287: The archive publishes its filter vocabulary @spec:AC-287"""
    from src.filters import FIELDS, OPERATORS, validate_filter
    from src.usecases.vocabulary import published_vocabulary

    published = published_vocabulary(conn)

    assert {field.name for field in published} == set(FIELDS)
    for field in published:
        assert field.type == FIELDS[field.name].type
        assert field.operators == OPERATORS[field.type]

    for field in published:
        example_op = field.operators[0]
        example_value: object
        if example_op == "exists":
            example_value = True
        elif field.type in ("number", "range"):
            example_value = 1
        elif field.type == "date":
            example_value = "2024-01-01"
        elif field.type == "text_list":
            example_value = ["x"]
        else:
            example_value = "x"
        try:
            validate_filter({"field": field.name, "op": example_op, "value": example_value})
        except Exception as exc:  # noqa: BLE001 - inspecting the refusal's own reason
            assert "is not a field this archive publishes" not in str(exc)
            assert "does not take" not in str(exc)

def test_ac_288(conn):
    """AC-288: It publishes the values the corpus actually holds, counted @spec:AC-288"""
    from src.usecases.vocabulary import published_vocabulary
    from tests.conftest import ChunkSpec, DocumentSpec, index_corpus

    written = index_corpus(
        conn,
        [
            DocumentSpec("pdf-doc", "Um PDF", (ChunkSpec(0, "conteudo", pages=(1, 1)),)),
            DocumentSpec("docx-doc", "Um DOCX", (ChunkSpec(0, "conteudo", pages=(1, 1)),)),
            DocumentSpec("gone-doc", "Sumiu", (ChunkSpec(0, "conteudo", pages=(1, 1)),), gone=True),
        ],
    )
    conn.execute(
        "UPDATE documents SET media_type = 'application/pdf' WHERE external_id IN (%s, %s)",
        ("pdf-doc", "gone-doc"),
    )
    conn.execute(
        "UPDATE documents SET media_type = "
        "'application/vnd.openxmlformats-officedocument.wordprocessingml.document' "
        "WHERE external_id = %s",
        ("docx-doc",),
    )
    for external_id, module in (("pdf-doc", "docai"), ("docx-doc", "pdftext"), ("gone-doc", "docai")):
        conn.execute(
            """
            INSERT INTO document_ingest (
                document_id, content_digest, fingerprint, read_fingerprint,
                cut_fingerprint, embed_fingerprint, reader_version, cut_version, process_module
            ) VALUES (%s, 'd', 'f', 'fr', 'fc', 'fe', 'v1', 'c1', %s)
            """,
            (written[external_id], module),
        )

    published = {field.name: field for field in published_vocabulary(conn)}

    media_types = {v.value: v.document_count for v in published["media_type"].values}
    process_modules = {v.value: v.document_count for v in published["process_module"].values}
    assert media_types["application/vnd.openxmlformats-officedocument.wordprocessingml.document"] == 1

    assert media_types["application/pdf"] == 1
    assert process_modules == {"docai": 1, "pdftext": 1}

def test_ac_289(api_world, served_corpus):
    """AC-289: The tool surface takes the same filter and still never raises @spec:AC-289"""
    import anyio
    from conftest import FakeQueryEmbedder

    from src.mcp import build_server

    client, service = api_world(embedder=FakeQueryEmbedder(degrees=0))
    server = build_server(service)
    shared_filter = {"field": "document_id", "op": "in", "value": ["drive-rescisao"]}

    route_body = client.post(
        "/v1/search", json={"question": "rescisão antecipada", "filter": shared_filter}
    ).json()
    tool_body = anyio.run(
        lambda: server.call_tool(
            "search", {"query": "rescisão antecipada", "filter": shared_filter}
        )
    ).structured_content

    route_ids = [passage["id"] for doc in route_body["results"] for passage in doc["passages"]]
    tool_ids = [passage["id"] for doc in tool_body["results"] for passage in doc["passages"]]
    assert route_ids and route_ids == tool_ids

    malformed = {"field": "not-a-field", "op": "is", "value": "x"}
    route_refusal = client.post("/v1/search", json={"question": "rescisão", "filter": malformed})
    assert route_refusal.status_code == 422

    tool_refusal = anyio.run(
        lambda: server.call_tool("search", {"query": "rescisão", "filter": malformed})
    )
    assert tool_refusal.is_error is False
    assert tool_refusal.structured_content["status"] == "failed"
    assert "not-a-field" in tool_refusal.structured_content["error"]

def test_ac_290(api_world, served_corpus):
    """AC-290: Filtering works the same whichever answer was asked for @spec:AC-290"""
    from conftest import FakeQueryEmbedder

    client, _ = api_world(embedder=FakeQueryEmbedder(degrees=0))
    shared_filter = {
        "field": "document_id", "op": "in", "value": ["drive-rescisao", "drive-reajuste"],
    }

    passages_body = client.post(
        "/v1/search",
        json={"question": "rescisão antecipada", "filter": shared_filter, "view": "passages"},
    ).json()
    documents_body = client.post(
        "/v1/search",
        json={"question": "rescisão antecipada", "filter": shared_filter, "view": "documents"},
    ).json()

    passage_documents = {doc["document"] for doc in passages_body["results"]}
    listed_documents = {doc["document"] for doc in documents_body["results"]}
    assert passage_documents and passage_documents == listed_documents

    assert passages_body["filter"] == documents_body["filter"]

def test_ac_291(conn):
    """AC-291: Every search records the filter it ran under @spec:AC-291"""
    import json

    applied = {"all": [{"field": "media_type", "op": "is", "value": "application/pdf"}]}

    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO search_events "
            "(client, caller, question_digest, dense_weight, lexical_weight, candidate_depth, "
            "result_count, index_search_effort, duration_ms, outcome, filter) "
            "VALUES ('example', 'mcp', 'digest-filtered', 0.5, 0.5, 10, 5, 40, 12.0, 'found', %s) "
            "RETURNING filter",
            (json.dumps(applied),),
        )
        (filtered_row,) = cursor.fetchone()

        cursor.execute(
            "INSERT INTO search_events "
            "(client, caller, question_digest, dense_weight, lexical_weight, candidate_depth, "
            "result_count, index_search_effort, duration_ms, outcome) "
            "VALUES ('example', 'mcp', 'digest-unfiltered', 0.5, 0.5, 10, 5, 40, 12.0, 'found') "
            "RETURNING filter",
        )
        (unfiltered_row,) = cursor.fetchone()

    assert filtered_row == applied

    assert unfiltered_row is None
    assert unfiltered_row != {}


def test_ac_292(conn):
    """AC-292: A field whose values grow with the corpus publishes none of them @spec:AC-292"""
    from src.filters import FIELDS, OPERATORS
    from src.usecases.vocabulary import published_vocabulary
    from tests.conftest import ChunkSpec, DocumentSpec, index_corpus

    written = index_corpus(
        conn,
        [
            DocumentSpec(
                "with-folder-and-heading", "Um contrato",
                (ChunkSpec(0, "conteudo", headings=("Cláusula 5",), pages=(1, 1)),),
            ),
        ],
    )
    conn.execute(
        "UPDATE documents SET location = ARRAY['Contratos'] WHERE id = %s",
        (written["with-folder-and-heading"],),
    )

    published = {field.name: field for field in published_vocabulary(conn)}

    for name in ("folder", "heading_path"):
        field = published[name]
        assert field.name == name
        assert field.type == FIELDS[name].type
        assert field.operators == OPERATORS[FIELDS[name].type]
        assert field.values is None
        assert field.values_open is True

    for name in ("media_type", "process_module", "block_kind"):
        field = published[name]
        assert field.values is not None
        assert field.values_open is False


def test_ac_293(api_world, served_corpus):
    """AC-293: The assistant can ask what can be filtered @spec:AC-293"""
    import anyio

    from src.mcp import build_server

    client, service = api_world()
    server = build_server(service)

    route_body = client.get("/v1/filters").json()
    tool_body = anyio.run(lambda: server.call_tool("filters", {})).structured_content

    assert tool_body["status"] == "found"
    assert tool_body["fields"] == route_body["fields"]

    schema = {tool.name: tool for tool in anyio.run(server.list_tools)}["filters"].input_schema
    assert schema.get("required", []) == []
    assert not schema.get("properties")
