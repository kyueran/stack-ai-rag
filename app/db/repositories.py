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

    def ingest_document_atomic(
        self,
        document_id: str,
        filename: str,
        byte_size: int,
        page_count: int,
        chunk_count: int,
        text_char_count: int,
        chunks: list[ChunkRecord],
        embeddings: list[EmbeddingRecord],
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
            existing_chunk_ids = [
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            if existing_chunk_ids:
                conn.executemany("DELETE FROM terms WHERE chunk_id = ?", [(chunk_id,) for chunk_id in existing_chunk_ids])
                conn.executemany(
                    "DELETE FROM embeddings WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in existing_chunk_ids],
                )
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

            if embeddings:
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


@dataclass(frozen=True)
class DocumentOptionRow:
    document_id: str
    filename: str
    chunk_count: int


@dataclass(frozen=True)
class TermStatsRow:
    term: str
    tf: int
    df: int
    total_docs: int


@dataclass(frozen=True)
class ConceptSupportRow:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    tf: int
    snippet: str


@dataclass(frozen=True)
class TermChunkPresenceRow:
    term: str
    chunk_id: str


class ConceptRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def count_documents(self) -> int:
        with self.database.connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"]) if row else 0

    def list_documents(self) -> list[DocumentOptionRow]:
        with self.database.connection() as conn:
            rows = conn.execute(
                """
                SELECT document_id, filename, chunk_count
                FROM documents
                ORDER BY ingested_at DESC
                """
            ).fetchall()
        return [
            DocumentOptionRow(
                document_id=str(row["document_id"]),
                filename=str(row["filename"]),
                chunk_count=int(row["chunk_count"]),
            )
            for row in rows
        ]

    def list_term_stats(self, document_id: str | None, min_term_length: int) -> list[TermStatsRow]:
        where_conditions: list[str] = ["LENGTH(t.term) >= ?"]
        scope_params: list[object] = [min_term_length]
        if document_id:
            where_conditions.append("c.document_id = ?")
            scope_params.append(document_id)
        scope_where_clause = "WHERE " + " AND ".join(where_conditions)

        with self.database.connection() as conn:
            total_docs_row = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
            total_docs = int(total_docs_row["count"]) if total_docs_row else 0
            if total_docs == 0:
                return []

            rows = conn.execute(
                f"""
                WITH scope_tf AS (
                    SELECT t.term AS term, SUM(t.tf) AS tf
                    FROM terms t
                    JOIN chunks c ON c.chunk_id = t.chunk_id
                    {scope_where_clause}
                    GROUP BY t.term
                ),
                global_df AS (
                    SELECT t.term AS term, COUNT(DISTINCT c.document_id) AS df
                    FROM terms t
                    JOIN chunks c ON c.chunk_id = t.chunk_id
                    WHERE LENGTH(t.term) >= ?
                    GROUP BY t.term
                )
                SELECT s.term, s.tf, g.df
                FROM scope_tf s
                JOIN global_df g ON g.term = s.term
                ORDER BY s.tf DESC, g.df ASC, s.term ASC
                """,
                [*scope_params, min_term_length],
            ).fetchall()

        return [
            TermStatsRow(
                term=str(row["term"]),
                tf=int(row["tf"]),
                df=int(row["df"]),
                total_docs=total_docs,
            )
            for row in rows
        ]

    def list_term_support(
        self,
        term: str,
        document_id: str | None,
        *,
        limit: int = 3,
    ) -> list[ConceptSupportRow]:
        where_conditions = ["t.term = ?"]
        params: list[object] = [term]
        if document_id:
            where_conditions.append("c.document_id = ?")
            params.append(document_id)
        where_clause = "WHERE " + " AND ".join(where_conditions)

        with self.database.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT t.chunk_id, c.document_id, d.filename, c.page_start, c.page_end, t.tf, c.text
                FROM terms t
                JOIN chunks c ON c.chunk_id = t.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                {where_clause}
                ORDER BY t.tf DESC, c.char_count DESC, c.chunk_id ASC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()

        return [
            ConceptSupportRow(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                filename=str(row["filename"]),
                page_start=int(row["page_start"]),
                page_end=int(row["page_end"]),
                tf=int(row["tf"]),
                snippet=str(row["text"])[:180],
            )
            for row in rows
        ]

    def list_term_chunk_presence(
        self,
        *,
        terms: list[str],
        document_id: str | None,
    ) -> list[TermChunkPresenceRow]:
        if not terms:
            return []

        term_placeholders = ",".join("?" for _ in terms)
        where_conditions = [f"t.term IN ({term_placeholders})"]
        params: list[object] = list(terms)
        if document_id:
            where_conditions.append("c.document_id = ?")
            params.append(document_id)
        where_clause = "WHERE " + " AND ".join(where_conditions)

        with self.database.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT t.term, t.chunk_id
                FROM terms t
                JOIN chunks c ON c.chunk_id = t.chunk_id
                {where_clause}
                """,
                params,
            ).fetchall()

        return [
            TermChunkPresenceRow(
                term=str(row["term"]),
                chunk_id=str(row["chunk_id"]),
            )
            for row in rows
        ]
