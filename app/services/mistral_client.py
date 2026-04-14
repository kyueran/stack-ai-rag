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

    def generate_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        api_key = self._settings.mistral_api_key.strip()
        if not api_key:
            return self._fallback_generation(user_prompt)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.mistral_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=self._settings.mistral_timeout_seconds) as client:
            response = client.post(
                f"{self._settings.mistral_api_base}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                msg = "Unexpected Mistral completion response format"
                raise ValueError(msg)
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices:
                msg = "Mistral completion response missing choices"
                raise ValueError(msg)
            first = choices[0]
            if not isinstance(first, dict):
                msg = "Unexpected choice format"
                raise ValueError(msg)
            message = first.get("message")
            if not isinstance(message, dict):
                msg = "Unexpected message format"
                raise ValueError(msg)
            content = message.get("content", "")
            if not isinstance(content, str):
                msg = "Unexpected completion content format"
                raise ValueError(msg)
            return content.strip()

    def _fallback_generation(self, user_prompt: str) -> str:
        lines = [line.strip() for line in user_prompt.splitlines() if line.strip().startswith("[source:")]
        if not lines:
            return "I could not find enough relevant context to answer."
        bullets = [f"- {line.split('] ', maxsplit=1)[-1][:180]}" for line in lines[:3]]
        return "Based on the retrieved context:\n" + "\n".join(bullets)


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
