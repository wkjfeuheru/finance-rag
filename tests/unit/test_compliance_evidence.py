from finance_rag.src.services.citation_validator import validate_finding_evidence
from finance_rag.src.services.compliance_service import _build_findings, _build_regulation_evidence


def test_build_findings_binds_verbatim_document_and_regulation_evidence():
    content = "产品承诺保本保息。"
    rules = [{
        "rule_id": "RL-002",
        "behavior": "非法集资或变相吸收公众存款",
        "legal_basis": "《防范和处置非法集资条例》",
        "keywords": ["保本保息"],
    }]
    sources = [{
        "title": "《防范和处置非法集资条例》",
        "source": "regulation.md",
        "chunk": 1,
        "content": "第五条：不得承诺保本保息。",
        "score": 0.9,
    }]
    catalog = _build_regulation_evidence(sources, content)
    findings = _build_findings(
        content,
        [{"segment": content, "rules": rules}],
        catalog,
        {"reason": "命中规则", "suggestions": ["删除承诺" ]},
    )

    finding = findings[0]
    assert finding["document_evidence"][0]["location"]["quote"] in content
    assert finding["regulation_evidence"][0]["evidence_id"] == "ev-0001"
    assert finding["evidence_status"] == "verified"
    assert validate_finding_evidence(finding, content, {"ev-0001": catalog[0]})["valid"]


def test_validate_finding_evidence_rejects_fabricated_quote_and_id():
    finding = {
        "document_evidence": [{"location": {"quote": "不存在的原文"}}],
        "regulation_evidence_ids": ["ev-missing"],
    }
    result = validate_finding_evidence(finding, "真实原文", {})
    assert result["valid"] is False
    assert any("原文证据" in issue for issue in result["issues"])
    assert any("证据 ID 不存在" in issue for issue in result["issues"])
