from typing import Any

from fastapi.testclient import TestClient


class StubIngestionRepository:
    def clear_all_documents(self) -> int:
        return 3


def test_ingest_clear_endpoint_removes_documents_and_files(
    client: TestClient,
    monkeypatch: Any,
) -> None:
    import app.api.routes.ingest as ingest_route

    monkeypatch.setattr(ingest_route, "get_ingestion_repository", lambda: StubIngestionRepository())
    monkeypatch.setattr(ingest_route, "clear_all_ingestion_artifacts", lambda _settings: 8)

    response = client.post("/api/v1/ingest/clear")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["cleared_documents"] == 3
    assert payload["removed_files"] == 8
