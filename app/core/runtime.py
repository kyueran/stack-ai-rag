from functools import lru_cache

from app.core.config import get_settings
from app.db.database import Database, build_database
from app.db.repositories import ConceptRepository, IngestionRepository, RetrievalRepository
from app.services.concepts import ConceptService
from app.services.generation import GenerationService
from app.services.hallucination import EvidenceChecker
from app.services.intent import IntentRouter
from app.services.keyword_search import KeywordSearchService
from app.services.mistral_client import MistralClient
from app.services.policy import QueryPolicyEngine
from app.services.query_rewrite import QueryRewriter
from app.services.retrieval import HybridRetrievalService
from app.services.semantic_search import SemanticSearchService


@lru_cache(maxsize=1)
def get_database() -> Database:
    return build_database(get_settings())


@lru_cache(maxsize=1)
def get_ingestion_repository() -> IngestionRepository:
    return IngestionRepository(get_database())


@lru_cache(maxsize=1)
def get_keyword_search_service() -> KeywordSearchService:
    return KeywordSearchService(get_database())


@lru_cache(maxsize=1)
def get_retrieval_repository() -> RetrievalRepository:
    return RetrievalRepository(get_database())


@lru_cache(maxsize=1)
def get_mistral_client() -> MistralClient:
    return MistralClient(get_settings())


@lru_cache(maxsize=1)
def get_semantic_search_service() -> SemanticSearchService:
    return SemanticSearchService(get_retrieval_repository(), get_mistral_client())


@lru_cache(maxsize=1)
def get_hybrid_retrieval_service() -> HybridRetrievalService:
    return HybridRetrievalService(
        keyword_service=get_keyword_search_service(),
        semantic_service=get_semantic_search_service(),
        retrieval_repository=get_retrieval_repository(),
    )


@lru_cache(maxsize=1)
def get_intent_router() -> IntentRouter:
    return IntentRouter()


@lru_cache(maxsize=1)
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter()


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    return GenerationService(get_mistral_client())


@lru_cache(maxsize=1)
def get_evidence_checker() -> EvidenceChecker:
    return EvidenceChecker()


@lru_cache(maxsize=1)
def get_query_policy_engine() -> QueryPolicyEngine:
    return QueryPolicyEngine()


@lru_cache(maxsize=1)
def get_concept_repository() -> ConceptRepository:
    return ConceptRepository(get_database())


@lru_cache(maxsize=1)
def get_concept_service() -> ConceptService:
    return ConceptService(get_concept_repository())
