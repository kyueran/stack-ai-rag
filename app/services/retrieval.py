from dataclasses import dataclass
from typing import Any

from app.db.repositories import RetrievalRepository


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    keyword_score: float
    semantic_score: float
    rrf_score: float
    fused_score: float
    source_filename: str | None = None


class HybridRetrievalService:
    def __init__(
        self,
        keyword_service: Any,
        semantic_service: Any,
        retrieval_repository: RetrievalRepository,
    ) -> None:
        self.keyword_service = keyword_service
        self.semantic_service = semantic_service
        self.retrieval_repository = retrieval_repository

    def retrieve(
        self,
        query: str,
        transformed_query: str,
        intent: str,
        top_k: int = 20,
        *,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        keyword_hits = self.keyword_service.search(transformed_query, top_k=top_k * 2)
        semantic_hits = self.semantic_service.search(transformed_query, top_k=top_k * 2)

        keyword_norm = _normalize_scores({hit.chunk_id: hit.score for hit in keyword_hits})
        semantic_norm = _normalize_scores({hit.chunk_id: hit.score for hit in semantic_hits})

        keyword_rank = {hit.chunk_id: rank for rank, hit in enumerate(keyword_hits, start=1)}
        semantic_rank = {hit.chunk_id: rank for rank, hit in enumerate(semantic_hits, start=1)}

        metadata: dict[str, tuple[str, str | None, int, int, str]] = {}
        for hit in keyword_hits:
            metadata[hit.chunk_id] = (hit.document_id, hit.source_filename, hit.page_start, hit.page_end, hit.text)
        for semantic_hit in semantic_hits:
            metadata[semantic_hit.chunk_id] = (
                semantic_hit.document_id,
                semantic_hit.source_filename,
                semantic_hit.page_start,
                semantic_hit.page_end,
                semantic_hit.text,
            )

        candidate_ids = sorted(set(keyword_rank) | set(semantic_rank))
        ranked_candidates: list[RetrievedChunk] = []

        for chunk_id in candidate_ids:
            keyword_score = keyword_norm.get(chunk_id, 0.0)
            semantic_score = semantic_norm.get(chunk_id, 0.0)

            rrf_score = 0.0
            if chunk_id in keyword_rank:
                rrf_score += 1.0 / (rrf_k + keyword_rank[chunk_id])
            if chunk_id in semantic_rank:
                rrf_score += 1.0 / (rrf_k + semantic_rank[chunk_id])

            present_weight = 0.0
            weighted_sum = 0.0
            if chunk_id in keyword_rank:
                weighted_sum += keyword_weight * keyword_score
                present_weight += keyword_weight
            if chunk_id in semantic_rank:
                weighted_sum += semantic_weight * semantic_score
                present_weight += semantic_weight

            blended_score = weighted_sum / present_weight if present_weight else 0.0
            fused_score = blended_score + (0.2 * rrf_score)
            doc_id, source_filename, page_start, page_end, text = metadata[chunk_id]
            ranked_candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    page_start=page_start,
                    page_end=page_end,
                    text=text,
                    keyword_score=keyword_score,
                    semantic_score=semantic_score,
                    rrf_score=rrf_score,
                    fused_score=fused_score,
                    source_filename=source_filename,
                )
            )

        ranked_candidates.sort(
            key=lambda item: (-item.fused_score, -item.semantic_score, -item.keyword_score, item.chunk_id)
        )
        top_candidates = ranked_candidates[:top_k]

        self.retrieval_repository.log_retrieval(
            query_text=query,
            transformed_query=transformed_query,
            intent=intent,
            top_k=top_k,
            results=[
                {
                    "chunk_id": candidate.chunk_id,
                    "document_id": candidate.document_id,
                    "source_filename": candidate.source_filename,
                    "page_start": candidate.page_start,
                    "page_end": candidate.page_end,
                    "keyword_score": round(candidate.keyword_score, 6),
                    "semantic_score": round(candidate.semantic_score, 6),
                    "rrf_score": round(candidate.rrf_score, 6),
                    "fused_score": round(candidate.fused_score, 6),
                }
                for candidate in top_candidates
            ],
        )
        return top_candidates


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return dict.fromkeys(scores.keys(), 0.5)
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}
