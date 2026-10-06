"""OpenAI-compatible embedder (works with vLLM, LM Studio, Strata, OpenAI itself)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from rag.embed.base import l2_normalize
from rag.embed.hashed import HashedNGramEmbedder


class OpenAICompatEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        dim: int | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        self.name = f"openai-compat:{model}"
        self.weight_profile = "semantic"
        self.model = model
        self.base_url = (base_url or os.environ.get("RAG_EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("RAG_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.batch_size = batch_size
        self.timeout = timeout
        self.dim = dim or 0
        self._token_fallback = HashedNGramEmbedder(dim=dim or 512, use_char=False)

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/embeddings", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"embeddings call failed ({exc.code}): {exc.read().decode('utf-8', 'ignore')}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"embeddings endpoint unreachable at {self.base_url}: {exc}") from exc

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            data = self._post({"model": self.model, "input": list(texts[start : start + self.batch_size])})
            rows.extend(item["embedding"] for item in data.get("data", []))
        matrix = np.asarray(rows, dtype=np.float32)
        self.dim = matrix.shape[1]
        self._token_fallback = HashedNGramEmbedder(dim=self.dim, use_char=False)
        return l2_normalize(matrix)

    def embed_tokens(self, text: str) -> np.ndarray:
        return self._token_fallback.embed_tokens(text)