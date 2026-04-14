from pydantic import BaseModel, Field


class ConceptSupport(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    tf: int
    snippet: str


class ConceptItem(BaseModel):
    term: str
    tf: int
    df: int
    idf: float
    tfidf: float
    document_coverage: float
    supports: list[ConceptSupport] = Field(default_factory=list)


class ConceptDocumentOption(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class ConceptsResponse(BaseModel):
    document_id: str | None = None
    total_documents: int
    top_n: int
    concepts: list[ConceptItem] = Field(default_factory=list)
    available_documents: list[ConceptDocumentOption] = Field(default_factory=list)


class ConceptGraphNode(BaseModel):
    term: str
    tfidf: float
    tf: int
    df: int
    document_coverage: float
    supports: list[ConceptSupport] = Field(default_factory=list)


class ConceptGraphEdge(BaseModel):
    source: str
    target: str
    weight: int


class ConceptsGraphResponse(BaseModel):
    document_id: str | None = None
    total_documents: int
    top_n: int
    nodes: list[ConceptGraphNode] = Field(default_factory=list)
    edges: list[ConceptGraphEdge] = Field(default_factory=list)
    available_documents: list[ConceptDocumentOption] = Field(default_factory=list)
