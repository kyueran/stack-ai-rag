from dataclasses import dataclass
from datetime import UTC, datetime

from app.db.database import Database


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    page_start: int
    page_end: int
    char_count: int
    text: str


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
