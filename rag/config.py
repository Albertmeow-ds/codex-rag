"""Layered configuration with YAML override support."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.agentic import AgenticConfig
from rag.chunking import ChunkingConfig
from rag.generate import GenerationConfig
from rag.retrieve.hybrid import RetrievalConfig


@dataclass(slots=True)
class WebConfig:
    enabled: bool = True
    backend: str = "auto"
    max_results: int = 6
    fetch_pages: bool = True
    prefer_snippet: bool = True
    min_snippet_chars: int = 160
    timeout: float = 12.0


@dataclass(slots=True)
class RAGConfig:
    db_path: str = "data/rag.db"
    embed_cache_path: str = "data/embeddings.db"
    embedder: str = "auto"
    embed_dim: int = 512
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.8:27b"
    llm: str = "auto"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_wire_api: str = "chat"
    reranker: str = "late"
    transform: str = "none"
    agentic: bool = True
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agentic_config: AgenticConfig = field(default_factory=AgenticConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "RAGConfig":
        return cls.from_dict(cls._load_raw(Path(path), set()))

    @staticmethod
    def _load_raw(path: Path, seen: set[Path]) -> dict:
        """Resolve an `extends` chain to any depth.

        Expanding only one level silently dropped every grandparent key: a config that
        extended config.casebook.yaml lost reranker and chunking settings inherited from
        config.yaml and fell back to dataclass defaults.
        """

        import yaml

        path = Path(path)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"circular config extends at {resolved}")
        seen = seen | {resolved}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base_ref = raw.pop("extends", None)
        if base_ref:
            # A child config usually overrides one knob. A shallow merge would drop every
            # sibling key in that section and silently fall back to dataclass defaults.
            base = RAGConfig._load_raw(path.parent / base_ref, seen)
            raw = RAGConfig.deep_merge(base, raw)
        return raw

    @staticmethod
    def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = RAGConfig.deep_merge(current, value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RAGConfig":
        scalar_keys = {
            "db_path", "embed_cache_path", "embedder", "embed_dim", "ollama_host", "ollama_model",
            "llm", "llm_base_url", "llm_model", "llm_wire_api", "reranker", "transform", "agentic",
        }
        config = cls()
        for key, value in raw.items():
            if key in scalar_keys:
                setattr(config, key, value)

        def apply(section: str, target: Any, allowed: set[str]) -> None:
            for key, value in (raw.get(section) or {}).items():
                if key in allowed and hasattr(target, key):
                    current = getattr(target, key)
                    if isinstance(current, tuple) and isinstance(value, list):
                        value = tuple(value)
                    setattr(target, key, value)

        apply("chunking", config.chunking, {"strategy", "max_tokens", "min_tokens", "overlap_tokens", "parent_tokens", "child_tokens", "semantic_break_percentile", "include_headings_in_text"})
        apply("retrieval", config.retrieval, {"candidate_k", "top_k", "channels", "fusion", "rrf_k", "weights", "sentence_window_radius", "expand_parents", "filters", "max_per_document"})
        apply("generation", config.generation, {"max_context_tokens", "max_output_tokens", "max_claims", "order", "dedupe", "require_citations", "system_prompt"})
        apply("agentic", config.agentic_config, {"enabled", "max_rounds", "max_graded_candidates", "relevance_threshold", "weak_coverage_threshold", "groundedness_threshold", "allow_web_fallback", "adaptive", "max_llm_calls", "max_total_tokens", "evidence", "evidence_drives_loop", "evidence_gate", "evidence_threshold",
                      "evidence_stop_on_no_entity", "stop_when_no_lexical_hit",
                      "lexical_hit_min_score"})
        apply("web", config.web, {"enabled", "backend", "max_results", "fetch_pages", "prefer_snippet", "min_snippet_chars", "timeout"})
        return config

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["retrieval"]["channels"] = list(self.retrieval.channels)
        return payload