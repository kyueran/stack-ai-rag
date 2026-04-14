from fastapi.testclient import TestClient


def test_app_startup(client: TestClient) -> None:
    response = client.get('/openapi.json')
    assert response.status_code == 200
    payload = response.json()
    assert payload['info']['title'] == 'stack-ai-rag'


def test_healthz(client: TestClient) -> None:
    response = client.get('/healthz')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['environment'] in {'development', 'test', 'production'}
