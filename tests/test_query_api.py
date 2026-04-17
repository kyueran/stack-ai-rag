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


class ExplodingRetrievalService:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        _ = (args, kwargs)
        raise AssertionError("retrieval should not run for policy refusal")


class TierRetrievalService:
    def retrieve(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        _ = (args, kwargs)
        return [
            RetrievedChunk(
                chunk_id="nimbus-1",
                document_id="doc-nimbus",
                text="Free-tier tenants are limited to 100 requests per minute.",
                page_start=1,
                page_end=1,
                keyword_score=1.0,
                semantic_score=1.0,
                rrf_score=0.03,
                fused_score=0.98,
                source_filename="project_nimbus_spec.pdf",
            ),
            RetrievedChunk(
                chunk_id="nimbus-2",
                document_id="doc-nimbus",
                text=(
                    "Standard-tier tenants are limited to 2,000 requests per minute. "
                    "Enterprise-tier tenants have no fixed limit but are subject to fair-use review above 50,000 requests per minute."
                ),
                page_start=1,
                page_end=1,
                keyword_score=0.95,
                semantic_score=0.95,
                rrf_score=0.03,
                fused_score=0.95,
                source_filename="project_nimbus_spec.pdf",
            ),
        ]


class PartialTableGenerationService:
    def generate(self, *args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        return (
            "| Claim | Evidence |\n"
            "| --- | --- |\n"
            "| Free-tier tenants are limited to 100 requests per minute. | "
            "Free-tier tenants are limited to 100 requests per minute. [source:nimbus-1 pages 1-1] |\n"
            "| Standard-tier tenants are limited to 2,000 requests per minute. | "
            "Standard-tier tenants are limited to 2,000 requests per minute. [source:nimbus-2 pages 1-1] |"
        )


class PassthroughEvidenceChecker:
    def filter_answer(self, answer: str, evidence: list[RetrievedChunk]) -> tuple[str, list[str]]:
        _ = evidence
        return answer, []


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


def test_query_endpoint_legal_advice_refusal_preempts_retrieval(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: ExplodingRetrievalService())
    response = client.post("/api/v1/query", json={"query": "Do I need a lawyer for this contract?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["intent"] == "refusal"
    assert payload["refusal_reason"] == "legal_advice_request"
    assert "legal advice" in payload["answer"].lower()
    assert "attorney" in payload["answer"].lower()
    assert payload["disclaimer"] is not None


def test_query_endpoint_medical_advice_refusal_preempts_retrieval(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: ExplodingRetrievalService())
    response = client.post(
        "/api/v1/query",
        json={"query": "I have chest pain and shortness of breath. What medication should I take?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["intent"] == "refusal"
    assert payload["refusal_reason"] == "medical_advice_request"
    assert "medication advice" in payload["answer"].lower()
    assert "call 911" in payload["answer"].lower()
    assert payload["disclaimer"] is not None


def test_query_endpoint_completes_missing_table_categories(client: TestClient, monkeypatch: Any) -> None:
    import app.api.routes.query as query_route

    monkeypatch.setattr(query_route, "get_hybrid_retrieval_service", lambda: TierRetrievalService())
    monkeypatch.setattr(query_route, "get_generation_service", lambda: PartialTableGenerationService())
    monkeypatch.setattr(query_route, "get_evidence_checker", lambda: PassthroughEvidenceChecker())
    response = client.post("/api/v1/query", json={"query": "Give me a table of Nimbus rate limits per tier."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["answer_format"] == "table"
    assert "enterprise-tier tenants have no fixed limit" in payload["answer"].lower()
    assert "[source:nimbus-2 pages 1-1]" in payload["answer"]
