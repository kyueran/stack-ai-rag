from dataclasses import dataclass

from app.db.repositories import RetrievalRepository
from app.services.mistral_client import MistralClient


@dataclass(frozen=True)
class SemanticSearchHit:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    score: float


class SemanticSearchService:
    def __init__(self, retrieval_repository: RetrievalRepository, mistral_client: MistralClient) -> None:
        self.retrieval_repository = retrieval_repository
        self.mistral_client = mistral_client

    def search(self, query: str, top_k: int = 20) -> list[SemanticSearchHit]:
        query = query.strip()
        if not query:
            return []

        query_vector = self.mistral_client.embed_texts([query]).vectors[0]
        candidates = self.retrieval_repository.list_semantic_candidates()
        scored = [
            (
                _cosine_similarity(query_vector, candidate.vector),
                candidate,
            )
            for candidate in candidates
        ]

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            SemanticSearchHit(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                text=candidate.text,
                score=score,
            )
            for score, candidate in ranked
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(float(x * y) for x, y in zip(left, right, strict=True))
    left_norm = sum(x * x for x in left) ** 0.5
    right_norm = sum(y * y for y in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(dot_product / (left_norm * right_norm))
