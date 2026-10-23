import pytest

from src.sources.base import (
    SOURCE_SLUG_GDRIVE,
    ChangePage,
    DocumentUnavailable,
    FetchedDocument,
    Source,
    SourceDocument,
    SourceError,
    registered_slugs,
    source_class,
)
from src.sources.gdrive import GoogleDriveSource
from src.sources.gdrive_export import KIND_DOCUMENT


def test_the_registry_is_a_closed_mapping_not_an_import_of_a_setting():
    assert registered_slugs() == (SOURCE_SLUG_GDRIVE,)
    assert source_class(SOURCE_SLUG_GDRIVE) is GoogleDriveSource


@pytest.mark.parametrize("slug", ["dropbox", "src.sources.gdrive", "", "../etc/passwd"])
def test_an_unregistered_slug_refuses_and_lists_what_exists(slug):
    with pytest.raises(SourceError) as excinfo:
        source_class(slug)

    message = str(excinfo.value)
    assert SOURCE_SLUG_GDRIVE in message


def test_the_registered_class_satisfies_the_contract():
    assert issubclass(GoogleDriveSource, Source)
    for method in ("sweep", "changes_since", "start_position", "fetch", "probe"):
        assert callable(getattr(GoogleDriveSource, method)), method


def test_a_document_without_an_identifier_is_refused():
    with pytest.raises(SourceError):
        SourceDocument(external_id="", name="Contrato", kind=KIND_DOCUMENT)


def test_a_document_with_no_type_names_the_likely_cause():
    with pytest.raises(SourceError) as excinfo:
        SourceDocument(external_id="doc-1", name="Contrato", kind="")

    assert "ASM-014" in str(excinfo.value)


def test_a_null_marker_means_no_claim_not_unchanged():
    document = SourceDocument(external_id="doc-1", name="Contrato", kind=KIND_DOCUMENT)

    assert document.revision_marker is None


def test_a_document_with_no_link_carries_none_rather_than_a_guess():
    document = SourceDocument(external_id="doc-1", name="Contrato", kind=KIND_DOCUMENT)

    assert document.link is None


def test_a_document_carries_the_link_the_source_gave_it():
    document = SourceDocument(
        external_id="doc-1", name="Contrato", kind=KIND_DOCUMENT,
        link="https://drive.google.com/file/d/doc-1/view",
    )

    assert document.link == "https://drive.google.com/file/d/doc-1/view"


def test_fetched_bytes_without_a_digest_are_refused():
    with pytest.raises(SourceError) as excinfo:
        FetchedDocument(
            external_id="doc-1", content=b"texto", media_type="text/markdown", content_digest=""
        )

    assert "AC-018" in str(excinfo.value)


def test_document_unavailable_carries_which_kind_of_unavailable():
    absent = DocumentUnavailable("doc-1", "notFound")
    denied = DocumentUnavailable("doc-2", "forbidden", denied=True)

    assert absent.denied is False
    assert denied.denied is True
    assert denied.external_id == "doc-2"
    assert "doc-2" in str(denied)


def test_an_empty_change_page_says_so():
    page = ChangePage(documents=(), removed=(), next_position="t1", exhausted=True)

    assert page.is_empty is True
    assert ChangePage(documents=(), removed=("a",), next_position="t1", exhausted=True).is_empty is False
