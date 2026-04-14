from dataclasses import dataclass
from hashlib import sha1
from typing import Any

import httpx

from app.core.config import Settings


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: list[list[float]]
    model: str


class MistralClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(vectors=[], model=self._settings.mistral_embedding_model)

        api_key = self._settings.mistral_api_key.strip()
        if not api_key:
            # Deterministic fallback for local development and tests when no API key is set.
            fallback_vectors = [_deterministic_fallback_embedding(text) for text in texts]
            return EmbeddingResponse(vectors=fallback_vectors, model="local-fallback")

        response_payload = self._post_embeddings(texts)
        raw_data = response_payload.get("data")
        if not isinstance(raw_data, list):
            msg = "Unexpected Mistral embeddings payload: 'data' must be a list"
            raise ValueError(msg)

        vectors: list[list[float]] = []
        for item in raw_data:
            if not isinstance(item, dict):
                msg = "Unexpected Mistral embeddings payload item type"
                raise ValueError(msg)
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                msg = "Unexpected Mistral embeddings payload: embedding must be a list"
                raise ValueError(msg)
            vectors.append([float(value) for value in embedding])

        return EmbeddingResponse(vectors=vectors, model=str(response_payload.get("model", "unknown")))

    def _post_embeddings(self, texts: list[str]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._settings.mistral_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.mistral_embedding_model,
            "input": texts,
        }
        with httpx.Client(timeout=self._settings.mistral_timeout_seconds) as client:
            response = client.post(
                f"{self._settings.mistral_api_base}/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                msg = "Unexpected Mistral embeddings response format"
                raise ValueError(msg)
            return parsed


def _deterministic_fallback_embedding(text: str, *, dimension: int = 64) -> list[float]:
    digest = sha1(text.encode()).digest()
    vector = [0.0 for _ in range(dimension)]
    for index, byte in enumerate(digest):
        slot = index % dimension
        vector[slot] += (byte / 255.0) - 0.5
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]
