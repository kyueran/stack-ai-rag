from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="stack-ai-rag", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-small-latest", alias="MISTRAL_MODEL")
    mistral_embedding_model: str = Field(default="mistral-embed", alias="MISTRAL_EMBEDDING_MODEL")
    mistral_api_base: str = Field(default="https://api.mistral.ai/v1", alias="MISTRAL_API_BASE")
    mistral_timeout_seconds: float = Field(default=30.0, alias="MISTRAL_TIMEOUT_SECONDS", ge=1, le=120)

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB", ge=1, le=200)
    max_files_per_upload: int = Field(default=10, alias="MAX_FILES_PER_UPLOAD", ge=1, le=100)
    max_pdf_pages: int = Field(default=500, alias="MAX_PDF_PAGES", ge=1, le=5000)
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE", ge=128, le=4000)
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP", ge=0, le=1200)
    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K", ge=1, le=100)
    citation_top_k: int = Field(default=5, alias="CITATION_TOP_K", ge=1, le=20)
    evidence_similarity_threshold: float = Field(default=0.35, alias="EVIDENCE_SIMILARITY_THRESHOLD", ge=0.0, le=1.0)
    query_evidence_min_coverage: float = Field(default=0.34, alias="QUERY_EVIDENCE_MIN_COVERAGE", ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be lower than CHUNK_SIZE"
            raise ValueError(msg)
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
