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
from app.services.answer_shape import select_output_format
from app.services.relevance import query_evidence_coverage
from app.services.table_completion import ensure_exhaustive_table_coverage

router = APIRouter(prefix="/api/v1", tags=["query"])


def _policy_refusal_message(refusal_reason: str | None) -> str:
    if refusal_reason == "pii_request":
        return "I can't help with requests involving sensitive personal data."
    if refusal_reason == "legal_advice_request":
        return (
            "I can't provide legal advice. "
            "Please consult a licensed attorney for advice specific to your situation."
        )
    if refusal_reason == "medical_advice_request":
        return (
            "I can't provide diagnosis or medication advice. "
            "Chest pain and shortness of breath can be an emergency. "
            "Call 911 now or seek emergency care immediately."
        )
    return "I can't help with that request."


@router.post("/query", response_model=QueryResponse)
def query_knowledge_base(payload: QueryRequest) -> QueryResponse:
    settings = get_settings()
    policy_engine = get_query_policy_engine()

    policy_decision = policy_engine.evaluate(payload.query)
    if policy_decision.refuse:
        return QueryResponse(
            status="refused",
            intent="refusal",
            rewritten_query=payload.query.strip().lower(),
            answer=_policy_refusal_message(policy_decision.refusal_reason),
            refusal_reason=policy_decision.refusal_reason,
            disclaimer=policy_decision.disclaimer,
        )

    intent_router = get_intent_router()
    intent_result = intent_router.detect(payload.query)

    if intent_result.intent == "chitchat":
        return QueryResponse(
            status="no_search",
            intent=intent_result.intent,
            rewritten_query=payload.query.strip().lower(),
            answer="Hi, ask me any question regarding your documents!",
            disclaimer=policy_decision.disclaimer,
        )

    if intent_result.intent == "refusal":
        return QueryResponse(
            status="refused",
            intent=intent_result.intent,
            rewritten_query=payload.query.strip().lower(),
            answer="I can't help with that request.",
            refusal_reason=intent_result.reason,
            disclaimer=policy_decision.disclaimer,
        )

    query_rewriter = get_query_rewriter()
    retrieval_service = get_hybrid_retrieval_service()
    generation_service = get_generation_service()
    evidence_checker = get_evidence_checker()
    rewritten = query_rewriter.rewrite(payload.query)
    answer_format = select_output_format(payload.query, intent_result.intent)
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

    coverage = query_evidence_coverage(rewritten.rewritten_query, candidates[: settings.citation_top_k])
    if coverage < settings.query_evidence_min_coverage:
        return QueryResponse(
            status="insufficient_evidence",
            intent=intent_result.intent,
            rewritten_query=rewritten.rewritten_query,
            answer="insufficient evidence",
            retrieval_count=len(candidates),
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
            source_filename=item.source_filename,
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
        output_format=answer_format,
        evidence=candidates[: settings.citation_top_k],
    )
    generated_answer = ensure_exhaustive_table_coverage(
        query=payload.query,
        answer=generated_answer,
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
        answer_format=answer_format,
        citations=citations,
        retrieval_count=len(candidates),
        unsupported_claims=unsupported_claims,
        disclaimer=policy_decision.disclaimer,
    )
