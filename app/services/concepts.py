import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

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


@dataclass(frozen=True)
class ConceptGraphEdge:
    source: str
    target: str
    weight: int


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
                    document_coverage=round(min(row.df / total_docs, 1.0), 6),
                    supports=supports,
                )
            )

        scored.sort(key=lambda item: (-item.tfidf, -item.tf, item.term))
        return total_docs, scored[:top_n]

    def get_concept_graph(
        self,
        document_id: str | None = None,
        *,
        top_n: int = 30,
        min_term_length: int = 3,
        support_k: int = 3,
        edge_limit: int = 120,
    ) -> tuple[int, list[ConceptScore], list[ConceptGraphEdge]]:
        total_docs, concepts = self.get_concepts(
            document_id=document_id,
            top_n=top_n,
            min_term_length=min_term_length,
            support_k=support_k,
        )
        if not concepts:
            return total_docs, [], []

        top_terms = [item.term for item in concepts]
        presence_rows = self.repository.list_term_chunk_presence(
            terms=top_terms,
            document_id=document_id,
        )

        chunk_terms: dict[str, set[str]] = defaultdict(set)
        for row in presence_rows:
            chunk_terms[row.chunk_id].add(row.term)

        edge_weights: dict[tuple[str, str], int] = defaultdict(int)
        for terms in chunk_terms.values():
            if len(terms) < 2:
                continue
            ordered_terms = sorted(terms)
            for left, right in combinations(ordered_terms, 2):
                edge_weights[(left, right)] += 1

        ranked_edges = sorted(edge_weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        edges = [
            ConceptGraphEdge(source=pair[0], target=pair[1], weight=weight)
            for pair, weight in ranked_edges[:edge_limit]
        ]

        return total_docs, concepts, edges
