from typing import Any

import httpx

from app.core.config import Settings
from app.services.mistral_client import MistralClient


def test_embed_texts_falls_back_when_embedding_request_fails(monkeypatch: Any) -> None:
    client = MistralClient(Settings(MISTRAL_API_KEY="test-key"))

    def _raise_http_error(texts: list[str]) -> dict[str, Any]:
        _ = texts
        raise httpx.HTTPError("network down")

    monkeypatch.setattr(client, "_post_embeddings", _raise_http_error)
    response = client.embed_texts(["hello world"])

    assert response.model == "local-fallback"
    assert len(response.vectors) == 1


def test_embed_texts_falls_back_when_embedding_count_is_invalid(monkeypatch: Any) -> None:
    client = MistralClient(Settings(MISTRAL_API_KEY="test-key"))

    def _invalid_payload(texts: list[str]) -> dict[str, Any]:
        _ = texts
        return {"model": "mistral-embed", "data": []}

    monkeypatch.setattr(client, "_post_embeddings", _invalid_payload)
    response = client.embed_texts(["hello world"])

    assert response.model == "local-fallback"
    assert len(response.vectors) == 1
