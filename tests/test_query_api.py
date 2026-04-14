from typing import Any

from fastapi.testclient import TestClient


class EmptyRetrievalService:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        _ = (args, kwargs)
        return []


def test_query_endpoint_chitchat_skips_search(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"query": "hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_search"
    assert payload["intent"] == "chitchat"


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
