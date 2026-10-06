"""Embedder selection: explicit config first, then capability probing, then offline fallback."""

from __future__ import annotations

import os

from rag.embed.base import Embedder
from rag.embed.hashed import HashedNGramEmbedder
from rag.embed.lsa import LSAEmbedder
from rag.embed.ollama import OllamaEmbedder, list_ollama_models, ollama_supports_embeddings
from rag.embed.openai_compat import OpenAICompatEmbedder

EMBEDDING_MODEL_HINTS = (
    "bge-m3",
    "bge-large",
    "nomic-embed-text",
    "snowflake-arctic-embed",
    "mxbai-embed-large",
    "nomic-embed",
    "embedding",
    "embed",
)


def build_embedder(name: str = "auto", dim: int = 512, ollama_host: str = "http://127.0.0.1:11434") -> Embedder:
    name = (name or "auto").lower()
    if name in {"hashed", "hash", "ngram"}:
        return HashedNGramEmbedder(dim=dim)
    if name == "lsa":
        return LSAEmbedder(dim=min(dim, 256))
    if name.startswith("ollama:"):
        return OllamaEmbedder(model=name.split(":", 1)[1], host=ollama_host, dim=dim)
    if name.startswith("openai:"):
        return OpenAICompatEmbedder(model=name.split(":", 1)[1], dim=dim)
    if name == "openai":
        return OpenAICompatEmbedder(dim=dim)
    if name == "auto":
        return probe_embedder(dim=dim, ollama_host=ollama_host)
    raise ValueError(f"unknown embedder: {name}")


def probe_embedder(dim: int = 512, ollama_host: str = "http://127.0.0.1:11434") -> Embedder:
    """Prefer a real embedding model; degrade to a deterministic offline embedder."""

    if os.environ.get("RAG_EMBED_BASE_URL") or os.environ.get("RAG_EMBED_MODEL"):
        try:
            return OpenAICompatEmbedder(
                model=os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small"), dim=dim
            )
        except Exception:
            pass

    for model in list_ollama_models(ollama_host):
        if any(hint in model.lower() for hint in EMBEDDING_MODEL_HINTS) and ollama_supports_embeddings(model, ollama_host):
            return OllamaEmbedder(model=model, host=ollama_host, dim=dim)

    return HashedNGramEmbedder(dim=dim)