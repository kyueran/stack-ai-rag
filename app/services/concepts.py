import math
from dataclasses import dataclass

from app.db.repositories import (
    ConceptRepository,
    DocumentOptionRow,
    TermStatsRow,
)


@dataclass(frozen=True)
class ConceptSupport:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    tf: int
    snippet: str


@dataclass(frozen=True)
class ConceptScore:
    term: str
    tf: int
    df: int
    idf: float
    tfidf: float
    document_coverage: float
    supports: list[ConceptSupport]


class ConceptService:
    def __init__(self, repository: ConceptRepository) -> None:
        self.repository = repository

    def get_document_options(self) -> list[DocumentOptionRow]:
        return self.repository.list_documents()

    def get_concepts(
        self,
        document_id: str | None = None,
        *,
        top_n: int = 30,
        min_term_length: int = 3,
        support_k: int = 3,
    ) -> tuple[int, list[ConceptScore]]:
        stats_rows: list[TermStatsRow] = self.repository.list_term_stats(
            document_id=document_id,
            min_term_length=min_term_length,
        )
        if not stats_rows:
            total_docs = self.repository.count_documents()
            return total_docs, []

        total_docs = stats_rows[0].total_docs
        scored: list[ConceptScore] = []
        for row in stats_rows:
            idf = math.log((total_docs + 1) / (row.df + 1)) + 1.0
            tfidf = float(row.tf) * idf
            supports = [
                ConceptSupport(
                    chunk_id=support.chunk_id,
                    document_id=support.document_id,
                    filename=support.filename,
                    page_start=support.page_start,
                    page_end=support.page_end,
                    tf=support.tf,
                    snippet=support.snippet,
                )
                for support in self.repository.list_term_support(
                    term=row.term,
                    document_id=document_id,
                    limit=support_k,
                )
            ]
            scored.append(
                ConceptScore(
                    term=row.term,
                    tf=row.tf,
                    df=row.df,
                    idf=round(idf, 6),
                    tfidf=round(tfidf, 6),
                    document_coverage=round(row.df / total_docs, 6),
                    supports=supports,
                )
            )

        scored.sort(key=lambda item: (-item.tfidf, -item.tf, item.term))
        return total_docs, scored[:top_n]
