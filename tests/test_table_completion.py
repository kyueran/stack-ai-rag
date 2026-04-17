from app.services.retrieval import RetrievedChunk
from app.services.table_completion import ensure_exhaustive_table_coverage


def _chunk(
    chunk_id: str,
    text: str,
    *,
    page_start: int = 1,
    page_end: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-nimbus",
        text=text,
        page_start=page_start,
        page_end=page_end,
        keyword_score=0.9,
        semantic_score=0.9,
        rrf_score=0.03,
        fused_score=0.9,
        source_filename="project_nimbus_spec.pdf",
    )


def test_table_completion_adds_missing_category_rows_for_exhaustive_query() -> None:
    query = "Give me a table of Nimbus rate limits per tier."
    answer = (
        "| Claim | Evidence |\n"
        "| --- | --- |\n"
        "| Free-tier tenants are limited to 100 requests per minute. | "
        "Free-tier tenants are limited to 100 requests per minute. [source:chunk-free pages 1-1] |\n"
        "| Standard-tier tenants are limited to 2,000 requests per minute. | "
        "Standard-tier tenants are limited to 2,000 requests per minute. [source:chunk-standard pages 1-1] |"
    )
    evidence = [
        _chunk("chunk-free", "Free-tier tenants are limited to 100 requests per minute."),
        _chunk(
            "chunk-standard",
            "Standard-tier tenants are limited to 2,000 requests per minute. "
            "Enterprise-tier tenants have no fixed limit but are subject to fair-use review above 50,000 requests per minute.",
        ),
    ]

    completed = ensure_exhaustive_table_coverage(query=query, answer=answer, evidence=evidence)

    assert "Enterprise-tier tenants have no fixed limit" in completed
    assert "[source:chunk-standard pages 1-1]" in completed


def test_table_completion_skips_non_exhaustive_queries() -> None:
    query = "Summarize Nimbus rate limits."
    answer = (
        "| Claim | Evidence |\n"
        "| --- | --- |\n"
        "| Free-tier tenants are limited to 100 requests per minute. | "
        "Free-tier tenants are limited to 100 requests per minute. [source:chunk-free pages 1-1] |"
    )
    evidence = [
        _chunk(
            "chunk-standard",
            "Standard-tier tenants are limited to 2,000 requests per minute. "
            "Enterprise-tier tenants have no fixed limit but are subject to fair-use review above 50,000 requests per minute.",
        )
    ]

    completed = ensure_exhaustive_table_coverage(query=query, answer=answer, evidence=evidence)

    assert completed == answer

