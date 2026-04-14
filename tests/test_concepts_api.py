from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass(frozen=True)
class StubDocument:
    document_id: str
    filename: str
    chunk_count: int


@dataclass(frozen=True)
class StubSupport:
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    tf: int
    snippet: str


@dataclass(frozen=True)
class StubConcept:
    term: str
    tf: int
    df: int
    idf: float
    tfidf: float
    document_coverage: float
    supports: list[StubSupport]


class StubConceptService:
    def get_document_options(self) -> list[StubDocument]:
        return [StubDocument(document_id="doc-a", filename="a.pdf", chunk_count=3)]

    def get_concepts(self, document_id: str | None = None, **kwargs: object) -> tuple[int, list[StubConcept]]:
        _ = kwargs
        return (
            1,
            [
                StubConcept(
                    term="equation",
                    tf=4,
                    df=1,
                    idf=1.1,
                    tfidf=4.4,
                    document_coverage=1.0,
                    supports=[
                        StubSupport(
                            chunk_id="chunk-1",
                            document_id=document_id or "doc-a",
                            filename="a.pdf",
                            page_start=1,
                            page_end=1,
                            tf=2,
                            snippet="equation discovery",
                        )
                    ],
                )
            ],
        )

    def get_concept_graph(
        self, document_id: str | None = None, **kwargs: object
    ) -> tuple[int, list[StubConcept], list[object]]:
        _ = kwargs
        return (1, self.get_concepts(document_id=document_id)[1], [])


def test_concepts_api_returns_tfidf_payload(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes.concepts as concepts_route

    monkeypatch.setattr(concepts_route, "get_concept_service", lambda: StubConceptService())
    response = client.get("/api/v1/concepts?document_id=doc-a&top_n=15")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == "doc-a"
    assert payload["top_n"] == 15
    assert payload["concepts"][0]["term"] == "equation"
    assert payload["available_documents"][0]["document_id"] == "doc-a"


def test_concepts_graph_api_returns_nodes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes.concepts as concepts_route

    monkeypatch.setattr(concepts_route, "get_concept_service", lambda: StubConceptService())
    response = client.get("/api/v1/concepts/graph?document_id=doc-a&top_n=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["term"] == "equation"
    assert payload["edges"] == []
