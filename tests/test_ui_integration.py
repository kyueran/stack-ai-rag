from fastapi.testclient import TestClient


def test_ui_ingest_endpoint_returns_status_fragment(client: TestClient) -> None:
    response = client.post(
        "/ui/ingest",
        files=[("files", ("note.txt", b"not a pdf", "text/plain"))],
    )
    assert response.status_code == 200
    assert "ingest-status" in response.text
    assert "rejected" in response.text


def test_ui_query_endpoint_appends_chat_fragment(client: TestClient) -> None:
    response = client.post("/ui/query", data={"query": "hello"})
    assert response.status_code == 200
    assert "bubble user" in response.text
    assert "bubble" in response.text


def test_ui_clear_ingest_endpoint_returns_status_fragment(client: TestClient) -> None:
    response = client.post("/ui/ingest/clear")
    assert response.status_code == 200
    assert "ingest-status" in response.text
    assert "Cleared" in response.text
