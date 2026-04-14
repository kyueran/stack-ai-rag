import math
from dataclasses import dataclass

from app.db.database import Database
from app.services.tokenizer import tokenize


@dataclass(frozen=True)
class KeywordSearchHit:
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    score: float


class KeywordSearchService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def search(self, query: str, top_k: int = 20, *, k1: float = 1.5, b: float = 0.75) -> list[KeywordSearchHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        with self.database.connection() as conn:
            total_chunks_row = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            total_chunks = int(total_chunks_row["count"]) if total_chunks_row else 0
            if total_chunks == 0:
                return []

            avg_len_row = conn.execute("SELECT AVG(char_count) AS avg_len FROM chunks").fetchone()
            avg_doc_len = float(avg_len_row["avg_len"] or 1.0) if avg_len_row else 1.0

            placeholders = ",".join("?" for _ in query_terms)
            df_rows = conn.execute(
                f"""
                SELECT term, COUNT(DISTINCT chunk_id) AS df
                FROM terms
                WHERE term IN ({placeholders})
                GROUP BY term
                """,
                query_terms,
            ).fetchall()
            doc_freq = {str(row["term"]): int(row["df"]) for row in df_rows}
            if not doc_freq:
                return []

            postings = conn.execute(
                f"""
                SELECT t.term, t.chunk_id, t.tf, c.document_id, c.page_start, c.page_end, c.char_count, c.text
                FROM terms t
                JOIN chunks c ON c.chunk_id = t.chunk_id
                WHERE t.term IN ({placeholders})
                """,
                query_terms,
            ).fetchall()

        scores: dict[str, float] = {}
        metadata: dict[str, tuple[str, int, int, str]] = {}

        for row in postings:
            term = str(row["term"])
            chunk_id = str(row["chunk_id"])
            tf = int(row["tf"])
            doc_len = max(int(row["char_count"]), 1)
            df = doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
            bm25 = idf * ((tf * (k1 + 1)) / max(denominator, 1e-9))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25
            metadata[chunk_id] = (
                str(row["document_id"]),
                int(row["page_start"]),
                int(row["page_end"]),
                str(row["text"]),
            )

        ranked_chunk_ids = sorted(scores.keys(), key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
        return [
            KeywordSearchHit(
                chunk_id=chunk_id,
                document_id=metadata[chunk_id][0],
                page_start=metadata[chunk_id][1],
                page_end=metadata[chunk_id][2],
                text=metadata[chunk_id][3],
                score=scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids
        ]
