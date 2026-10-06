"""Ollama-backed embedder. Used automatically when an embedding-capable model exists."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from rag.embed.base import l2_normalize
from rag.embed.hashed import HashedNGramEmbedder


class OllamaUnavailableError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "bge-m3",
        host: str = "http://127.0.0.1:11434",
        dim: int | None = None,
        batch_size: int = 16,
        timeout: float = 120.0,
    ) -> None:
        self.name = f"ollama:{model}"
        self.weight_profile = "semantic"
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.dim = dim or 0
        self._token_fallback = HashedNGramEmbedder(dim=dim or 512, use_char=False)

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        try:
            data = _post(f"{self.host}/api/embed", payload, self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise OllamaUnavailableError(f"ollama embed failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaUnavailableError(f"ollama unreachable at {self.host}: {exc}") from exc
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise OllamaUnavailableError("ollama embed returned no vectors")
        return embeddings

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self._embed_batch(texts[start : start + self.batch_size]))
        matrix = np.asarray(rows, dtype=np.float32)
        self.dim = matrix.shape[1]
        self._token_fallback = HashedNGramEmbedder(dim=self.dim, use_char=False)
        return l2_normalize(matrix)

    def embed_tokens(self, text: str) -> np.ndarray:
        return self._token_fallback.embed_tokens(text)


def list_ollama_models(host: str = "http://127.0.0.1:11434", timeout: float = 6.0) -> list[str]:
    try:
        data = _get(f"{host.rstrip('/')}/api/tags", timeout)
    except Exception:
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def ollama_supports_embeddings(model: str, host: str = "http://127.0.0.1:11434", timeout: float = 20.0) -> bool:
    try:
        data = _post(f"{host.rstrip('/')}/api/show", {"model": model}, timeout)
    except Exception:
        return False
    capabilities = data.get("capabilities") or []
    if "embedding" in [c.lower() for c in capabilities]:
        return True
    return bool(data.get("embedding"))