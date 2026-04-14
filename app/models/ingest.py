from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestFileResult(BaseModel):
    filename: str
    document_id: str | None = None
    status: Literal["accepted", "rejected"]
    bytes_received: int = 0
    page_count: int = 0
    chunk_count: int = 0
    text_char_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    extraction_error: str | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    status: Literal["ok", "partial_success", "error"]
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    accepted_count: int
    rejected_count: int
    files: list[IngestFileResult]
