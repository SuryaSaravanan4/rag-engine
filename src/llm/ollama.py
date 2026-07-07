from typing import Any, Iterator

from .base import BaseModelProvider


class OllamaProvider(BaseModelProvider):
    """LLM backend using a local Ollama instance.
    
    Fully offline — no API key needed. Requires Ollama running locally.
    https://ollama.com

    Args:
        model: Ollama model tag (e.g. "llama3", "mistral", "phi3").
        base_url: Ollama server URL. Default: http://localhost:11434.
        temperature: Sampling temperature.
        max_tokens: Maps to Ollama's `num_predict` option. None (default)
            leaves generation length unbounded, matching Ollama's own default.
        timeout: Request timeout in seconds — applies to both the initial
            connection and each read (including between streamed chunks).
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        return options

    def complete(self, system: str, user: str) -> str:
        import requests
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": self._options(),
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        import json
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": self._options(),
            "stream": True,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content")
            if content:
                yield content
