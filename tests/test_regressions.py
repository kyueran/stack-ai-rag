from io import BytesIO
from typing import Any

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.services.retrieval import RetrievedChunk


class RaisingIngestionRepository:
    def ingest_document_atomic(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        raise RuntimeError("db failed")


class LowEvidenceRetrieval:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        _ = (args, kwargs)
        return [
            RetrievedChunk(
                chunk_id="weak-citation",
                document_id="doc-1",
                text="Weakly related context.",
                page_start=1,
                page_end=1,
                keyword_score=0.0,
                semantic_score=0.0,
                rrf_score=0.0,
                fused_score=0.01,
            )
        ]


def _make_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_ingest_rejects_invalid_pdf_signature(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest",
        files=[("files", ("fake.pdf", b"not really a pdf", "application/pdf"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["files"][0]["error"] == "File does not match PDF signature"


def test_ingest_rolls_back_when_storage_fails(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.ingest as ingest_route

    monkeypatch.setattr(ingest_route, "get_ingestion_repository", lambda: RaisingIngestionRepository())
    response = client.post(
        "/api/v1/ingest",
        files=[("files", ("ok.pdf", _make_pdf_bytes(), "application/pdf"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["files"][0]["error"].startswith("Ingestion failed and was rolled back safely")


def test_query_regression_for_weak_evidence_and_policy(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: LowEvidenceRetrieval())

    weak_response = client.post("/api/v1/query", json={"query": "Tell me the exact contract timeline"})
    assert weak_response.status_code == 200
    weak_payload = weak_response.json()
    assert weak_payload["status"] == "insufficient_evidence"

    policy_response = client.post("/api/v1/query", json={"query": "Can you share my credit card number?"})
    assert policy_response.status_code == 200
    policy_payload = policy_response.json()
    assert policy_payload["status"] == "refused"
    assert policy_payload["refusal_reason"] == "pii_request"
