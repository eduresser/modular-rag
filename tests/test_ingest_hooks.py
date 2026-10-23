import pytest

from src.ingest import ingest_documents
from src.ingest_hooks import (
    HOOK_AFTER_READ,
    HOOK_BEFORE_SEND,
    HOOK_POINTS,
    NO_HOOKS,
    REGISTERED_HOOKS,
    HookError,
    StageHooks,
    hooks_for,
)
from src.sources.base import FetchedDocument, SourceDocument
from tests.conftest import contract_payload, ingest_world, pdf_file

DOCUMENT = SourceDocument(external_id="doc-1", name="Contrato", kind="application/pdf")
FETCHED = FetchedDocument(
    external_id="doc-1", content=b"CPF 000.000.000-00", media_type="application/pdf",
    content_digest="digest-1",
)


def _mask(document, fetched):
    return FetchedDocument(
        external_id=fetched.external_id,
        content=fetched.content.replace(b"000.000.000-00", b"[CPF]"),
        media_type=fetched.media_type,
        content_digest=fetched.content_digest,
    )


def _annotate(external_id, payload):
    return {**payload, "annotatedBy": external_id}


def test_a_deployment_with_no_client_module_runs_no_hooks():
    assert NO_HOOKS.empty is True
    assert REGISTERED_HOOKS == {}
    assert set(HOOK_POINTS) == {HOOK_BEFORE_SEND, HOOK_AFTER_READ}


def test_with_no_hooks_both_points_are_the_identity():
    assert NO_HOOKS.apply_before_send(DOCUMENT, FETCHED) is FETCHED
    payload = {"documentLayout": {"blocks": []}}
    assert NO_HOOKS.apply_after_read("doc-1", payload) is payload


def test_the_before_send_point_acts_while_the_bytes_are_still_ours():
    hooks = StageHooks(before_send=(_mask,))

    masked = hooks.apply_before_send(DOCUMENT, FETCHED)

    assert b"000.000.000-00" not in masked.content
    assert b"[CPF]" in masked.content
    assert masked.external_id == FETCHED.external_id
    assert masked.content_digest == FETCHED.content_digest


def test_the_after_read_point_acts_on_what_pieces_are_cut_from():
    hooks = StageHooks(after_read=(_annotate,))

    payload = hooks.apply_after_read("doc-1", {"documentLayout": {"blocks": []}})

    assert payload["annotatedBy"] == "doc-1"


def test_hooks_run_in_the_order_they_were_given():
    order: list[str] = []

    def first(external_id, payload):
        order.append("first")
        return payload

    def second(external_id, payload):
        order.append("second")
        return payload

    StageHooks(after_read=(first, second)).apply_after_read("doc-1", {})

    assert order == ["first", "second"]


def test_a_hook_that_returns_the_wrong_thing_is_caught_at_the_point_it_ran():
    with pytest.raises(HookError) as excinfo:
        StageHooks(before_send=(lambda document, fetched: None,)).apply_before_send(DOCUMENT, FETCHED)
    assert HOOK_BEFORE_SEND in str(excinfo.value)

    with pytest.raises(HookError) as excinfo:
        StageHooks(after_read=(lambda external_id, payload: "text",)).apply_after_read("doc-1", {})
    assert HOOK_AFTER_READ in str(excinfo.value)


def test_a_name_no_build_registers_stops_the_deployment_rather_than_being_skipped():
    with pytest.raises(HookError) as excinfo:
        hooks_for(["mask-identifiers"])

    message = str(excinfo.value)
    assert "mask-identifiers" in message
    assert "this build registers no hooks" in message
    assert "principle P-011" in message


def test_resolving_no_names_gives_a_deployment_with_no_hooks():
    assert hooks_for([]) == NO_HOOKS


def test_the_pipeline_runs_both_points_around_the_reader(monkeypatch, conn):
    seen: dict[str, object] = {}

    def record_bytes(document, fetched):
        seen["before_send"] = fetched.content
        return fetched

    def record_payload(external_id, payload):
        seen["after_read"] = external_id
        return payload

    world = ingest_world(
        monkeypatch,
        files=[pdf_file("doc-1", content=b"%PDF contrato")],
        outputs={"doc-1": contract_payload(clauses=("Cláusula primeira.",))},
        hooks=StageHooks(before_send=(record_bytes,), after_read=(record_payload,)),
    )

    run = ingest_documents(conn, world.pipeline, list(world.pipeline.source.sweep()))

    assert run.errored == 0
    assert seen["before_send"] == b"%PDF contrato"
    assert seen["after_read"] == "doc-1"
