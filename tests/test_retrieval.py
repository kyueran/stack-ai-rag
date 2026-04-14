from pathlib import Path

from app.db.database import Database
from app.db.repositories import RetrievalRepository
from app.services.keyword_search import KeywordSearchHit
from app.services.retrieval import HybridRetrievalService
from app.services.semantic_search import SemanticSearchHit


class StubKeywordService:
    def search(self, query: str, top_k: int = 20) -> list[KeywordSearchHit]:
        _ = (query, top_k)
        return [
            KeywordSearchHit("chunk-1", "doc-1", "About retrieval", 1, 1, 9.0),
            KeywordSearchHit("chunk-2", "doc-1", "About food", 2, 2, 2.0),
        ]


class StubSemanticService:
    def search(self, query: str, top_k: int = 20) -> list[SemanticSearchHit]:
        _ = (query, top_k)
        return [
            SemanticSearchHit("chunk-2", "doc-1", "About food", 2, 2, 0.9),
            SemanticSearchHit("chunk-1", "doc-1", "About retrieval", 1, 1, 0.6),
        ]


def test_hybrid_retrieval_fuses_scores_and_logs_results(tmp_path: Path) -> None:
    db = Database(tmp_path / "rag.sqlite")
    db.initialize()
    retrieval_repo = RetrievalRepository(db)

    service = HybridRetrievalService(StubKeywordService(), StubSemanticService(), retrieval_repo)
    results = service.retrieve(
        query="retrieval question",
        transformed_query="retrieval question",
        intent="knowledge",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-2"

    with db.connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM retrieval_logs").fetchone()["count"]
    assert int(count) == 1
