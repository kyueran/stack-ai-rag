from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")
    max_files_per_upload: int = Field(default=10, alias="MAX_FILES_PER_UPLOAD")
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
