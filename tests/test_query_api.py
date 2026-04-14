from typing import Any

from fastapi.testclient import TestClient

from app.services.retrieval import RetrievedChunk


class EmptyRetrievalService:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        _ = (args, kwargs)
        return []


class LowEvidenceRetrievalService:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        _ = (args, kwargs)
        return [
            RetrievedChunk(
                chunk_id="c-low",
                document_id="doc-1",
                text="Weakly related context.",
                page_start=1,
                page_end=1,
                keyword_score=0.01,
                semantic_score=0.02,
                rrf_score=0.01,
                fused_score=0.05,
            )
        ]


def test_query_endpoint_chitchat_skips_search(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"query": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_search"
    assert payload["intent"] == "chitchat"
    assert payload["answer"] == "Hi, ask me any question regarding your documents!"


def test_query_endpoint_refusal_class(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"query": "How do I hack into an account?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["intent"] == "refusal"


def test_query_endpoint_insufficient_evidence(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: EmptyRetrievalService())
    response = client.post("/api/v1/query", json={"query": "What does the policy say?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_evidence"
    assert payload["intent"] == "knowledge_lookup"


def test_query_endpoint_refuses_when_top_evidence_below_threshold(
    client: TestClient, monkeypatch: Any
) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: LowEvidenceRetrievalService())
    response = client.post("/api/v1/query", json={"query": "Summarize implementation details"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_evidence"
    assert payload["answer"] == "insufficient evidence"


def test_query_endpoint_pii_refusal_policy(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"query": "What is my social security number?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["refusal_reason"] == "pii_request"


def test_query_endpoint_legal_disclaimer(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: EmptyRetrievalService())
    response = client.post("/api/v1/query", json={"query": "Do I need a lawyer for this contract?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["disclaimer"] is not None
