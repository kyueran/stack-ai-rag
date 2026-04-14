from app.services.query_rewrite import QueryRewriter


def test_query_rewrite_expands_abbreviations() -> None:
    rewriter = QueryRewriter()
    result = rewriter.rewrite("Explain RAG and LLM best practices")

    assert "retrieval" in result.rewritten_query
    assert "generation" in result.rewritten_query
    assert "language" in result.rewritten_query


def test_query_rewrite_removes_filler_words() -> None:
    rewriter = QueryRewriter()
    result = rewriter.rewrite("Could you please tell me about API limits")

    assert "please" not in result.rewritten_query
    assert "application" in result.rewritten_query
