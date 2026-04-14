from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    output_format: Literal["paragraph", "list", "table"] = "paragraph"
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    status: Literal["ok", "no_search", "refused", "insufficient_evidence"]
    intent: Literal["chitchat", "knowledge_lookup", "refusal"]
    rewritten_query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_count: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None
    disclaimer: str | None = None
