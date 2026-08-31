from finance_rag.src.infrastructure.relational_db.knowledge_base import KnowledgeBaseRepository


def test_knowledge_base_repository_round_trip(tmp_path):
    repository = KnowledgeBaseRepository(f"sqlite:///{tmp_path / 'business.db'}")
    repository.save(
        {
            "knowledge_bases": [
                {
                    "name": "research",
                    "display_name": "投研",
                    "description": "研究资料",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "builtin_seeded": True,
        }
    )

    data = repository.load()

    assert data["builtin_seeded"] is True
    assert data["knowledge_bases"] == [
        {
            "name": "research",
            "display_name": "投研",
            "description": "研究资料",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
