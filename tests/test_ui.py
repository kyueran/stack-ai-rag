from fastapi.testclient import TestClient


def test_ui_root_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Grounded PDF Chat" in response.text
    assert "Knowledge Base Upload" in response.text
    assert "Chat" in response.text
    assert "Interactive Knowledge Graph" in response.text
    assert 'id="mode-workspace" class="mode-view is-active"' in response.text
    assert 'id="mode-graph" class="mode-view" aria-hidden="true" hidden' in response.text


def test_ui_static_css_served(client: TestClient) -> None:
    response = client.get("/ui/static/styles.css")
    assert response.status_code == 200
    assert "--accent" in response.text
