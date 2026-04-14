from functools import lru_cache

from app.core.config import get_settings
from app.db.database import Database, build_database
from app.db.repositories import IngestionRepository
from app.services.keyword_search import KeywordSearchService


@lru_cache(maxsize=1)
def get_database() -> Database:
    return build_database(get_settings())


@lru_cache(maxsize=1)
def get_ingestion_repository() -> IngestionRepository:
    return IngestionRepository(get_database())


@lru_cache(maxsize=1)
def get_keyword_search_service() -> KeywordSearchService:
    return KeywordSearchService(get_database())
