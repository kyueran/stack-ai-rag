from pathlib import Path

from app.db.database import Database
from app.db.repositories import ChunkRecord, ConceptRepository, IngestionRepository
from app.services.concepts import ConceptService


def test_concept_service_returns_ranked_tfidf_concepts(tmp_path: Path) -> None:
    database = Database(tmp_path / "rag.sqlite")
    database.initialize()

    ingestion_repository = IngestionRepository(database)
    ingestion_repository.upsert_document(
        document_id="doc-a",
        filename="a.pdf",
        byte_size=100,
        page_count=1,
        chunk_count=2,
        text_char_count=220,
    )
    ingestion_repository.replace_chunks(
        document_id="doc-a",
        chunks=[
            ChunkRecord(
                chunk_id="a-1",
                page_start=1,
                page_end=1,
                char_count=120,
                text="rapid equation discovery model retrieves equations from noisy data",
            ),
            ChunkRecord(
                chunk_id="a-2",
                page_start=2,
                page_end=2,
                char_count=100,
                text="equation discovery uses optimizer and embedding search",
            ),
        ],
    )

    ingestion_repository.upsert_document(
        document_id="doc-b",
        filename="b.pdf",
        byte_size=100,
        page_count=1,
        chunk_count=1,
        text_char_count=100,
    )
    ingestion_repository.replace_chunks(
        document_id="doc-b",
        chunks=[
            ChunkRecord(
                chunk_id="b-1",
                page_start=1,
                page_end=1,
                char_count=100,
                text="retrieval pipeline query rewriting and intent routing",
            )
        ],
    )

    service = ConceptService(ConceptRepository(database))
    total_docs, concepts = service.get_concepts(document_id="doc-a", top_n=5)

    assert total_docs == 2
    assert concepts
    assert concepts[0].tfidf >= concepts[-1].tfidf
    assert any(item.term == "equation" for item in concepts)
    equation = next(item for item in concepts if item.term == "equation")
    assert equation.supports
    assert equation.supports[0].document_id == "doc-a"


def test_concept_service_builds_graph_edges_from_chunk_cooccurrence(tmp_path: Path) -> None:
    database = Database(tmp_path / "rag.sqlite")
    database.initialize()

    ingestion_repository = IngestionRepository(database)
    ingestion_repository.upsert_document(
        document_id="doc-a",
        filename="a.pdf",
        byte_size=120,
        page_count=1,
        chunk_count=2,
        text_char_count=180,
    )
    ingestion_repository.replace_chunks(
        document_id="doc-a",
        chunks=[
            ChunkRecord(
                chunk_id="a-1",
                page_start=1,
                page_end=1,
                char_count=90,
                text="equation discovery optimization embedding model",
            ),
            ChunkRecord(
                chunk_id="a-2",
                page_start=1,
                page_end=1,
                char_count=90,
                text="equation model optimization benchmark",
            ),
        ],
    )

    service = ConceptService(ConceptRepository(database))
    total_docs, concepts, edges = service.get_concept_graph(document_id="doc-a", top_n=10)

    assert total_docs == 1
    assert concepts
    assert edges
    assert any({edge.source, edge.target} == {"equation", "model"} for edge in edges)
