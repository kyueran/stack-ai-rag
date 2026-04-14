from pathlib import Path

from app.db.database import Database
from app.db.repositories import ChunkRecord, IngestionRepository
from app.services.keyword_search import KeywordSearchService


def test_keyword_search_ranks_relevant_chunk_first(tmp_path: Path) -> None:
    database = Database(tmp_path / "rag.sqlite")
    database.initialize()

    repo = IngestionRepository(database)
    repo.upsert_document(
        document_id="doc-a",
        filename="guide.pdf",
        byte_size=123,
        page_count=1,
        chunk_count=2,
        text_char_count=220,
    )
    repo.replace_chunks(
        document_id="doc-a",
        chunks=[
            ChunkRecord(
                chunk_id="c1",
                page_start=1,
                page_end=1,
                char_count=120,
                text="RAG systems combine retrieval and generation to answer grounded questions.",
            ),
            ChunkRecord(
                chunk_id="c2",
                page_start=1,
                page_end=1,
                char_count=110,
                text="A healthy diet includes vegetables, fruits, and protein.",
            ),
        ],
    )

    service = KeywordSearchService(database)
    hits = service.search("How does retrieval augmented generation work?", top_k=2)

    assert hits
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0
