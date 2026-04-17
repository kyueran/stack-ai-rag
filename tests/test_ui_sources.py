from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.models.query import Citation, QueryResponse


def test_ui_document_route_renders_viewer(client: TestClient, monkeypatch: Any, tmp_path: Path) -> None:
    import app.api.routes.ui as ui_route

    document_id = "0123456789abcdef0123456789abcdef"
    raw_dir = tmp_path / "pdfs" / "raw"
    raw_dir.mkdir(parents=True)
    sample_pdf = raw_dir / f"{document_id}_sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.7\n%sample")

    class DummySettings:
        data_dir = tmp_path

    monkeypatch.setattr(ui_route, "get_settings", lambda: DummySettings())

    response = client.get(f"/ui/document/{document_id}?page=4")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert f"/ui/document/{document_id}/raw#page=4" in response.text


def test_ui_document_raw_route_serves_inline_pdf(client: TestClient, monkeypatch: Any, tmp_path: Path) -> None:
    import app.api.routes.ui as ui_route

    document_id = "0123456789abcdef0123456789abcdef"
    raw_dir = tmp_path / "pdfs" / "raw"
    raw_dir.mkdir(parents=True)
    sample_pdf = raw_dir / f"{document_id}_sample.pdf"
    sample_pdf.write_bytes(b"%PDF-1.7\n%sample")

    class DummySettings:
        data_dir = tmp_path

    monkeypatch.setattr(ui_route, "get_settings", lambda: DummySettings())

    response = client.get(f"/ui/document/{document_id}/raw")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"].startswith("inline")


def test_ui_query_renders_clickable_inline_source_links(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.ui as ui_route

    def fake_query(payload: Any) -> QueryResponse:
        _ = payload
        return QueryResponse(
            status="ok",
            intent="knowledge_lookup",
            rewritten_query="readme framework",
            answer="README is fast [source:abc123 pages 3-3].",
            citations=[
                Citation(
                    chunk_id="abc123",
                    document_id="0123456789abcdef0123456789abcdef",
                    source_filename="readme.pdf",
                    page_start=3,
                    page_end=3,
                    score=0.95,
                    snippet="README is fast",
                )
            ],
            retrieval_count=1,
        )

    monkeypatch.setattr(ui_route, "query_knowledge_base", fake_query)

    response = client.post("/ui/query", data={"query": "what is README"})
    assert response.status_code == 200
    assert "/ui/document/0123456789abcdef0123456789abcdef?page=3" in response.text
    assert "source:readme.pdf p3-3" in response.text
