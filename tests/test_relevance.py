from app.services.relevance import query_evidence_coverage
from app.services.retrieval import RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        text=text,
        page_start=1,
        page_end=1,
        keyword_score=0.9,
        semantic_score=0.9,
        rrf_score=0.01,
        fused_score=0.9,
    )


def test_query_evidence_coverage_handles_inflections() -> None:
    coverage = query_evidence_coverage(
        "nimbus rate limits per tier",
        [_chunk("Standard-tier tenants are limited to 2,000 requests per minute.")],
    )
    assert coverage >= 0.5


def test_query_evidence_coverage_ignores_instruction_noise() -> None:
    coverage = query_evidence_coverage(
        "Give me a table of Nimbus rate limits per tier",
        [_chunk("Nimbus rate limits vary by tier for each tenant type.")],
    )
    assert coverage >= 0.7


def test_query_evidence_coverage_low_for_off_topic_evidence() -> None:
    coverage = query_evidence_coverage(
        "exact implementation timeline",
        [_chunk("Sourdough uses bread flour, water, starter, and salt.")],
    )
    assert coverage == 0.0

