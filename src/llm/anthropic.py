from typing import TYPE_CHECKING, Iterator

from .base import BaseModelProvider

if TYPE_CHECKING:
    import anthropic


class AnthropicProvider(BaseModelProvider):
    """LLM backend using the Anthropic Messages API.

    Args:
        model: Claude model string. Default: claude-sonnet-4-6.
        max_tokens: Max tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self._client: "anthropic.Anthropic | None" = None

    def _load(self) -> None:
        from ..env import require_env
        api_key = self.api_key or require_env("ANTHROPIC_API_KEY")
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        if not self._client:
            self._load()
        assert self._client is not None
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(block, "text", "") for block in msg.content)
        if not text:
            raise ValueError("Anthropic response contained no text content")
        return text

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        if not self._client:
            self._load()
        assert self._client is not None
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream
