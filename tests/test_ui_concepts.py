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
        return [StubDocument(document_id="doc-a", filename="paper.pdf", chunk_count=4)]

    def get_concepts(self, document_id: str | None = None, **kwargs: object) -> tuple[int, list[StubConcept]]:
        _ = kwargs
        return (
            1,
            [
                StubConcept(
                    term="equation",
                    tf=5,
                    df=1,
                    idf=1.0,
                    tfidf=5.0,
                    document_coverage=1.0,
                    supports=[
                        StubSupport(
                            chunk_id="c1",
                            document_id=document_id or "doc-a",
                            filename="paper.pdf",
                            page_start=2,
                            page_end=2,
                            tf=2,
                            snippet="equation discovery details",
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


def test_ui_concepts_panel_renders_interactive_graph(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes.ui as ui_route

    monkeypatch.setattr(ui_route, "get_concept_service", lambda: StubConceptService())
    monkeypatch.setattr(ui_route, "_has_backing_pdf", lambda _: True)
    response = client.get("/ui/concepts?top_n=20")

    assert response.status_code == 200
    assert "Interactive Knowledge Graph (TF-IDF)" in response.text
    assert "Ranked Table" not in response.text
    assert "equation" in response.text
    assert "forceSimulation" in response.text
    assert "/ui/document/${support.document_id}?page=${support.page_start}" in response.text


def test_ui_concepts_panel_hides_stale_data_without_backing_pdf(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.routes.ui as ui_route

    monkeypatch.setattr(ui_route, "get_concept_service", lambda: StubConceptService())
    monkeypatch.setattr(ui_route, "_has_backing_pdf", lambda _: False)
    response = client.get("/ui/concepts?top_n=20")

    assert response.status_code == 200
    assert "Upload PDFs to generate a concept graph." in response.text
