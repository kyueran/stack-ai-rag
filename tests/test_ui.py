from fastapi.testclient import TestClient


def test_ui_root_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Grounded PDF Chat" in response.text


def test_ui_static_css_served(client: TestClient) -> None:
    response = client.get("/ui/static/styles.css")
    assert response.status_code == 200
    assert "--accent" in response.text
