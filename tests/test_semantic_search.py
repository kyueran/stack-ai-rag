from pathlib import Path

from app.core.config import Settings
from app.db.database import Database
from app.db.repositories import (
    ChunkRecord,
    EmbeddingRecord,
    IngestionRepository,
    RetrievalRepository,
)
from app.services.mistral_client import MistralClient
from app.services.semantic_search import SemanticSearchService


def test_semantic_search_scores_candidates(tmp_path: Path) -> None:
    database = Database(tmp_path / "rag.sqlite")
    database.initialize()

    ingestion_repo = IngestionRepository(database)
    ingestion_repo.upsert_document(
        document_id="doc-1",
        filename="a.pdf",
        byte_size=100,
        page_count=1,
        chunk_count=2,
        text_char_count=180,
    )
    chunks = [
        ChunkRecord(
            chunk_id="chunk-rag",
            page_start=1,
            page_end=1,
            char_count=90,
            text="RAG retrieves evidence and then generates answers grounded on sources.",
        ),
        ChunkRecord(
            chunk_id="chunk-food",
            page_start=1,
            page_end=1,
            char_count=90,
            text="A recipe for soup contains carrots, onions, and salt.",
        ),
    ]
    ingestion_repo.replace_chunks("doc-1", chunks)

    client = MistralClient(Settings())
    embedded = client.embed_texts([chunk.text for chunk in chunks])
    ingestion_repo.replace_embeddings(
        [
            EmbeddingRecord(chunk_id=chunk.chunk_id, model=embedded.model, vector=vector)
            for chunk, vector in zip(chunks, embedded.vectors, strict=True)
        ]
    )

    semantic_service = SemanticSearchService(RetrievalRepository(database), client)
    hits = semantic_service.search("What is retrieval augmented generation?", top_k=2)

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
