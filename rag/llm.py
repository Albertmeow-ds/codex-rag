"""LLM adapters: Ollama native, OpenAI-compatible (chat + responses wire), and a null LLM."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol, Sequence


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str: ...


class NullLLM:
    """Sentinel used when no generator is reachable; the pipeline degrades gracefully."""

    name = "none"

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str:
        return ""


class OllamaChat:
    def __init__(
        self,
        model: str = "qwen3.8:27b",
        host: str = "http://127.0.0.1:11434",
        num_ctx: int = 16384,
        timeout: float = 600.0,
        enable_thinking: bool = False,
    ) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.last_usage: dict[str, int] = {}

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str:
        payload: dict = {
            "model": self.model,
            "stream": False,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        if not self.enable_thinking:
            payload["think"] = False
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ollama chat failed ({exc.code}): {exc.read().decode('utf-8', 'ignore')}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"ollama unreachable at {self.host}: {exc}") from exc
        self.last_usage = {
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        }
        return (data.get("response") or "").strip()


class OpenAICompatChat:
    """Supports both /chat/completions and the /responses wire API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        wire_api: str = "chat",
        timeout: float = 300.0,
    ) -> None:
        self.name = f"openai-compat:{model}"
        self.model = model
        self.base_url = (base_url or os.environ.get("RAG_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("RAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.wire_api = wire_api
        self.timeout = timeout
        self.last_usage: dict[str, int] = {}

    def _record_usage(self, data: dict) -> None:
        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        }

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"llm call failed ({exc.code}): {exc.read().decode('utf-8', 'ignore')}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"llm endpoint unreachable at {self.base_url}: {exc}") from exc

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1200) -> str:
        if self.wire_api == "responses":
            payload: dict = {"model": self.model, "input": prompt, "temperature": temperature, "store": False}
            if system:
                payload["instructions"] = system
            data = self._post("/responses", payload)
            self._record_usage(data)
            parts: list[str] = []
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        text = content.get("text") or content.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                elif item.get("type") == "output_text":
                    parts.append(item.get("text", ""))
            if not parts and isinstance(data.get("output_text"), str):
                parts.append(data["output_text"])
            return "".join(parts).strip()

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = self._post(
            "/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        self._record_usage(data)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message", {}).get("content") or "").strip()


def build_llm(
    name: str = "auto",
    ollama_model: str = "qwen3.8:27b",
    ollama_host: str = "http://127.0.0.1:11434",
    base_url: str = "",
    model: str = "",
    wire_api: str = "chat",
) -> LLM:
    """Resolve an LLM adapter.

    Reuses an already-loaded local server (Strata, vLLM, LM Studio) when a base_url is
    given, so no second model is loaded into VRAM.
    """

    name = (name or "auto").lower()
    if name in {"none", "null", "off"}:
        return NullLLM()
    if name.startswith("ollama:"):
        return OllamaChat(model=name.split(":", 1)[1], host=ollama_host)
    if name.startswith("openai:"):
        return OpenAICompatChat(model=name.split(":", 1)[1], base_url=base_url or None, wire_api=wire_api)
    if name == "openai":
        return OpenAICompatChat(model=model or "gpt-4o-mini", base_url=base_url or None, wire_api=wire_api)
    if name == "auto":
        resolved_url = base_url or os.environ.get("RAG_LLM_BASE_URL", "")
        resolved_model = model or os.environ.get("RAG_LLM_MODEL", "")
        resolved_wire = wire_api or os.environ.get("RAG_LLM_WIRE_API", "chat")
        if resolved_url or resolved_model:
            return OpenAICompatChat(
                model=resolved_model or "gpt-4o-mini",
                base_url=resolved_url or None,
                wire_api=resolved_wire,
            )
        return OllamaChat(model=ollama_model, host=ollama_host)
    raise ValueError(f"unknown llm: {name}")


_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json(text: str, default: object = None) -> object:
    if not text:
        return default
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return default


def extract_numbers(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def chunk_lines(text: str) -> Sequence[str]:
    return [line for line in text.split("\n") if line.strip()]