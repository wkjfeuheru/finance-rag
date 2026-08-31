"""Initialize relational business data for a local or deployed environment."""

from __future__ import annotations

from datetime import datetime, timezone

from finance_rag.src.infrastructure.relational_db.knowledge_base import KnowledgeBaseRepository
from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES


DEFAULT_DESCRIPTIONS = {
    "investment_research": "收录各类市场研究报告、行业分析、公司深度研究、宏观经济研判等专业分析材料",
    "compliance_risk": "包含监管政策、合规要求、审计准则和方法、反欺诈规则等，确保业务符合监管要求",
    "business_operations": "涵盖金融产品说明书、操作流程SOP、市场营销策略等支持日常业务开展的文件",
    "management": "汇集公司组织架构、人力资源管理制度、行政管理办法、财务管理制度、内部控制体系、绩效考核方案等企业内部治理和综合管理类文件",
    "it_technology": "包含系统开发文档、数据库设计、接口规范、运维手册、网络安全策略、数据治理标准、软件版本说明等与信息技术基础设施和系统研发相关的技术资料",
}


def load_seed_data() -> dict:
    """Build default knowledge-base metadata without a file-backed registry."""
    created_at = datetime.now(timezone.utc).isoformat()
    names = [*DOCUMENT_CATEGORIES, "it_technology"]
    return {
        "knowledge_bases": [
            {
                "name": name,
                "display_name": DOCUMENT_CATEGORIES.get(name, "IT技术类"),
                "description": DEFAULT_DESCRIPTIONS[name],
                "created_at": created_at,
            }
            for name in names
        ],
        "builtin_seeded": True,
    }


def main() -> None:
    """Create relational tables and seed default knowledge-base metadata."""
    repository = KnowledgeBaseRepository()
    repository._get_engine()
    data = load_seed_data()
    repository.save(data)
    print(f"knowledge bases initialized: {len(data['knowledge_bases'])}")


if __name__ == "__main__":
    main()
