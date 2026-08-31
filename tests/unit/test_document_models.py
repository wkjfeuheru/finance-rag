from finance_rag.src.rag.models.document_category import merge_category_into_metadata
from finance_rag.src.rag.models.document_version import extract_document_version


def test_merge_category_preserves_metadata_and_sets_category():
    metadata = {"source": "report.pdf", "category": "policy"}

    result = merge_category_into_metadata(metadata, "research_report")

    assert result == {"source": "report.pdf", "category": "research_report"}
    assert metadata["category"] == "policy"


def test_merge_category_adds_empty_default():
    assert merge_category_into_metadata({"source": "report.pdf"})["category"] == ""


def test_extract_version_prefers_explicit_metadata():
    assert extract_document_version("report_v1.2.pdf", {"version": "2.0"}) == "2.0"


def test_extract_version_from_filename():
    assert extract_document_version("annual-report-v3.1.pdf") == "3.1"
    assert extract_document_version("annual-report.pdf") == ""
