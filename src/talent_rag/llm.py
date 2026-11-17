"""Thin LLM clients built on the standard library only (no SDK needed).

* :class:`OpenAICompatibleClient` - OpenAI ``/chat/completions`` and every compatible server (Ollama's ``/v1``, vLLM,
  LM Studio, Groq, Together ...). Set ``OPENAI_API_KEY`` (and ``base_url`` for non-OpenAI servers).
* :class:`GeminiClient`           - Google Gemini REST API (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``).
* :class:`OllamaClient`           - a local Ollama server (e.g. Llama 3) - resumes never leave your machine.
* :class:`CallableLLM`            - wraps any Python function (tests, custom models).

Model names change often: pass the one you want, e.g. ``--explain openai:gpt-4o-mini`` or ``--explain ollama:llama3.1:8b``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_MODEL, DEFAULT_OPENAI_MODEL


class LLMError(RuntimeError):
    """Raised when an LLM call fails (network, authentication, bad response)."""


class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's text answer (we ask for JSON)."""


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float = 120.0, retries: int = 2, secret: str = "") -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    last = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {exc.code}: {body}"
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = f"connection problem: {exc}"
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    if secret:
        last = last.replace(secret, "***")
    raise LLMError(f"LLM request to {url.split('?')[0]} failed - {last}")


class OpenAICompatibleClient(LLMClient):
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0) -> None:
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"  # e.g. http://localhost:8000/v1 for vLLM / LM Studio
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.name = f"openai:{model}"

    def generate(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout, secret=self.api_key)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response format from {self.base_url}: {str(data)[:200]}") from exc


class GeminiClient(LLMClient):
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None, base_url: str = "https://generativelanguage.googleapis.com/v1beta", timeout: float = 120.0) -> None:
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        self.name = f"gemini:{model}"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        data = _post_json(f"{self.base_url}/models/{self.model}:generateContent", payload, {"x-goog-api-key": self.api_key}, self.timeout, secret=self.api_key)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:200]}") from exc


class OllamaClient(LLMClient):
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str | None = None, timeout: float = 300.0) -> None:
        self.model, self.timeout = model, timeout
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.name = f"ollama:{model}"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        data = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:200]}") from exc


class CallableLLM(LLMClient):
    def __init__(self, fn: Callable[[str, str], str], name: str = "callable") -> None:
        self.fn, self.name = fn, name

    def generate(self, system: str, user: str) -> str:
        return self.fn(system, user)


def make_llm(spec: str | None) -> LLMClient | None:
    """``none`` / ``template`` -> ``None`` (offline template explainer); ``openai:<model>``, ``gemini:<model>``, ``ollama:<model>``."""
    if not spec or spec.lower() in {"none", "template", "off"}:
        return None
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider == "openai":
        return OpenAICompatibleClient(model or DEFAULT_OPENAI_MODEL)
    if provider == "gemini":
        return GeminiClient(model or DEFAULT_GEMINI_MODEL)
    if provider == "ollama":
        return OllamaClient(model or DEFAULT_OLLAMA_MODEL)
    raise ValueError(f"Unknown LLM provider {provider!r}. Use none, openai:<model>, gemini:<model> or ollama:<model>.")
