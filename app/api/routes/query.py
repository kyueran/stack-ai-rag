from fastapi import APIRouter

from app.core.config import get_settings
from app.core.runtime import (
    get_evidence_checker,
    get_generation_service,
    get_hybrid_retrieval_service,
    get_intent_router,
    get_query_policy_engine,
    get_query_rewriter,
)
from app.models.query import Citation, QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(payload: QueryRequest) -> QueryResponse:
    settings = get_settings()
    intent_router = get_intent_router()
    query_rewriter = get_query_rewriter()
    retrieval_service = get_hybrid_retrieval_service()
    generation_service = get_generation_service()
    evidence_checker = get_evidence_checker()
    policy_engine = get_query_policy_engine()

    policy_decision = policy_engine.evaluate(payload.query)
    if policy_decision.refuse:
        return QueryResponse(
            status="refused",
            intent="refusal",
            rewritten_query=payload.query.strip().lower(),
            answer="I can't help with requests involving sensitive personal data.",
            refusal_reason=policy_decision.refusal_reason,
        )

    intent_result = intent_router.detect(payload.query)
    rewritten = query_rewriter.rewrite(payload.query)

    if intent_result.intent == "chitchat":
        return QueryResponse(
            status="no_search",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="Hi! Ask me a question about your uploaded PDFs and I can help.",
            disclaimer=policy_decision.disclaimer,
        )

    if intent_result.intent == "refusal":
        return QueryResponse(
            status="refused",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="I can't help with that request.",
            refusal_reason=intent_result.reason,
            disclaimer=policy_decision.disclaimer,
        )

    top_k = payload.top_k or settings.retrieval_top_k
    candidates = retrieval_service.retrieve(
        query=payload.query,
        transformed_query=rewritten.rewritten_query,
        intent=intent_result.intent,
        top_k=top_k,
    )

    if not candidates:
        return QueryResponse(
            status="insufficient_evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="insufficient evidence",
            retrieval_count=0,
            disclaimer=policy_decision.disclaimer,
        )

    strongest_score = max(candidate.fused_score for candidate in candidates)
    if strongest_score < settings.evidence_similarity_threshold:
        return QueryResponse(
            status="insufficient_evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="insufficient evidence",
            retrieval_count=len(candidates),
            disclaimer=policy_decision.disclaimer,
        )

    citations = [
        Citation(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            page_start=item.page_start,
            page_end=item.page_end,
            score=item.fused_score,
            snippet=item.text[:280],
        )
        for item in candidates[: settings.citation_top_k]
    ]
    generated_answer = generation_service.generate(
        query=payload.query,
        intent=intent_result.intent,
        output_format=payload.output_format,
        evidence=candidates[: settings.citation_top_k],
    )
    filtered_answer, unsupported_claims = evidence_checker.filter_answer(
        generated_answer,
        candidates[: settings.citation_top_k],
    )
    if not filtered_answer:
        return QueryResponse(
            status="insufficient_evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="insufficient evidence",
            retrieval_count=len(candidates),
            unsupported_claims=unsupported_claims,
            disclaimer=policy_decision.disclaimer,
        )

    return QueryResponse(
        status="ok",
        intent=intent_result.intent,
        rewritten_query=rewritten.rewritten_query,
        answer=filtered_answer,
        citations=citations,
        retrieval_count=len(candidates),
        unsupported_claims=unsupported_claims,
        disclaimer=policy_decision.disclaimer,
    )
