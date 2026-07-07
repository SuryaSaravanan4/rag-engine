from typing import TYPE_CHECKING, Iterator

from .base import BaseModelProvider

if TYPE_CHECKING:
    import openai


class OpenAIProvider(BaseModelProvider):
    """LLM backend using the OpenAI Chat Completions API.

    Args:
        model: OpenAI model string. Default: gpt-4o-mini.
        max_tokens: Max tokens in the response.
        temperature: Sampling temperature.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self._client: "openai.OpenAI | None" = None

    def _load(self) -> None:
        from ..env import require_env
        api_key = self.api_key or require_env("OPENAI_API_KEY")
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        if not self._client:
            self._load()
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response contained no text content")
        return content

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        if not self._client:
            self._load()
        assert self._client is not None
        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
