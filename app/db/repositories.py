import json
from dataclasses import dataclass
from datetime import UTC, datetime

from app.db.database import Database
from app.services.tokenizer import term_frequencies, tokenize


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    page_start: int
    page_end: int
    char_count: int
    text: str


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    model: str
    vector: list[float]


class IngestionRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def upsert_document(
        self,
        document_id: str,
        filename: str,
        byte_size: int,
        page_count: int,
        chunk_count: int,
        text_char_count: int,
    ) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, byte_size, page_count, chunk_count, text_char_count, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    byte_size = excluded.byte_size,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    text_char_count = excluded.text_char_count,
                    ingested_at = excluded.ingested_at
                """,
                (
                    document_id,
                    filename,
                    byte_size,
                    page_count,
                    chunk_count,
                    text_char_count,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def replace_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        with self.database.connection() as conn:
            existing_chunk_ids = [
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            if existing_chunk_ids:
                conn.executemany("DELETE FROM terms WHERE chunk_id = ?", [(chunk_id,) for chunk_id in existing_chunk_ids])
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, document_id, page_start, page_end, char_count, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (chunk.chunk_id, document_id, chunk.page_start, chunk.page_end, chunk.char_count, chunk.text)
                    for chunk in chunks
                ],
            )
            term_rows: list[tuple[str, str, int, str]] = []
            for chunk in chunks:
                frequencies = term_frequencies(tokenize(chunk.text))
                for term, frequency in frequencies.items():
                    term_rows.append((term, chunk.chunk_id, frequency, "body"))
            if term_rows:
                conn.executemany(
                    """
                    INSERT INTO terms (term, chunk_id, tf, field)
                    VALUES (?, ?, ?, ?)
                    """,
                    term_rows,
                )

    def replace_embeddings(self, embeddings: list[EmbeddingRecord]) -> None:
        with self.database.connection() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (chunk_id, model, dimension, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    model = excluded.model,
                    dimension = excluded.dimension,
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                [
                    (
                        embedding.chunk_id,
                        embedding.model,
                        len(embedding.vector),
                        json.dumps(embedding.vector),
                        datetime.now(UTC).isoformat(),
                    )
                    for embedding in embeddings
                ],
            )


@dataclass(frozen=True)
class SemanticCandidate:
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    text: str
    vector: list[float]


class RetrievalRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list_semantic_candidates(self) -> list[SemanticCandidate]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.page_start, c.page_end, c.text, e.vector_json
                FROM chunks c
                JOIN embeddings e ON e.chunk_id = c.chunk_id
                """
            ).fetchall()
        candidates: list[SemanticCandidate] = []
        for row in rows:
            candidates.append(
                SemanticCandidate(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    page_start=int(row["page_start"]),
                    page_end=int(row["page_end"]),
                    text=str(row["text"]),
                    vector=[float(value) for value in json.loads(str(row["vector_json"]))],
                )
            )
        return candidates

    def log_retrieval(
        self,
        query_text: str,
        transformed_query: str,
        intent: str,
        top_k: int,
        results: list[dict[str, object]],
    ) -> None:
        with self.database.connection() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_logs (query_text, transformed_query, intent, top_k, results_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_text,
                    transformed_query,
                    intent,
                    top_k,
                    json.dumps(results, ensure_ascii=True),
                    datetime.now(UTC).isoformat(),
                ),
            )
