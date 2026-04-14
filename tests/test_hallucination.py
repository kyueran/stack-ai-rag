from app.services.hallucination import EvidenceChecker
from app.services.retrieval import RetrievedChunk


def test_evidence_checker_removes_unsupported_sentences() -> None:
    checker = EvidenceChecker(sentence_support_threshold=0.4)
    evidence = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="RAG retrieves documents before generation.",
            page_start=1,
            page_end=1,
            keyword_score=1.0,
            semantic_score=1.0,
            rrf_score=1.0,
            fused_score=1.0,
        )
    ]
    answer = "RAG retrieves documents before generation. It was invented in 1812."

    filtered, unsupported = checker.filter_answer(answer, evidence)

    assert "retrieves documents" in filtered
    assert any("1812" in sentence for sentence in unsupported)
