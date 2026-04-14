from fastapi import APIRouter, Query

from app.core.runtime import get_concept_service
from app.models.concepts import ConceptDocumentOption, ConceptItem, ConceptsResponse, ConceptSupport

router = APIRouter(prefix="/api/v1", tags=["concepts"])


@router.get("/concepts", response_model=ConceptsResponse)
def list_tfidf_concepts(
    document_id: str | None = Query(default=None),
    top_n: int = Query(default=30, ge=1, le=100),
) -> ConceptsResponse:
    concept_service = get_concept_service()
    total_documents, concepts = concept_service.get_concepts(document_id=document_id, top_n=top_n)
    available_documents = concept_service.get_document_options()

    return ConceptsResponse(
        document_id=document_id,
        total_documents=total_documents,
        top_n=top_n,
        concepts=[
            ConceptItem(
                term=item.term,
                tf=item.tf,
                df=item.df,
                idf=item.idf,
                tfidf=item.tfidf,
                document_coverage=item.document_coverage,
                supports=[
                    ConceptSupport(
                        chunk_id=support.chunk_id,
                        document_id=support.document_id,
                        filename=support.filename,
                        page_start=support.page_start,
                        page_end=support.page_end,
                        tf=support.tf,
                        snippet=support.snippet,
                    )
                    for support in item.supports
                ],
            )
            for item in concepts
        ],
        available_documents=[
            ConceptDocumentOption(
                document_id=document.document_id,
                filename=document.filename,
                chunk_count=document.chunk_count,
            )
            for document in available_documents
        ],
    )
