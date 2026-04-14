from app.services.chunking import chunk_pages
from app.services.pdf_extract import ExtractedPage


def test_chunking_is_deterministic_and_respects_overlap() -> None:
    pages = [
        ExtractedPage(
            page_number=1,
            text=(
                "FastAPI is a modern Python web framework. "
                "It is designed for speed. "
                "The framework includes automatic docs. "
                "Retrieval augmented generation combines retrieval and generation."
            ),
            char_count=170,
        )
    ]

    first = chunk_pages("doc-123", pages, chunk_size=80, chunk_overlap=20)
    second = chunk_pages("doc-123", pages, chunk_size=80, chunk_overlap=20)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) >= 2
    assert first[0].page_start == 1
    assert first[0].page_end == 1
    assert any("framework" in chunk.text.lower() for chunk in first)
