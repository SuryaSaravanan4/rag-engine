from typing import TYPE_CHECKING, Any, Iterator

from .base import BaseModelProvider

if TYPE_CHECKING:
    import anthropic


# Claude models that removed the sampling parameters (temperature/top_p/top_k).
# Sending temperature to any of these returns a 400 — response depth is
# controlled by output_config.effort instead. Anything not listed here still
# accepts temperature, so unknown/custom model strings keep the old behavior.
#
# This list needs updating as models ship: the removal applies to every
# frontier Claude model from Opus 4.7 and Sonnet 5 onward, so a newer model
# will reject temperature before it appears here. If a request 400s on a
# sampling parameter, add the model prefix.
_NO_SAMPLING_PARAMS = (
    "claude-fable-",
    "claude-mythos-",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)


def accepts_sampling_params(model: str) -> bool:
    """Return True if `model` still accepts temperature/top_p/top_k."""
    return not model.startswith(_NO_SAMPLING_PARAMS)


class AnthropicProvider(BaseModelProvider):
    """LLM backend using the Anthropic Messages API.

    Args:
        model: Claude model string. Default: claude-sonnet-5.
        max_tokens: Max tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic). Silently
            dropped for models that no longer accept it (see
            `accepts_sampling_params`) so one config can drive any backend.
        effort: Optional response-depth hint — "low", "medium", "high",
            "xhigh", or "max". The replacement for temperature on current
            models; sent only when set. Not supported on Sonnet 4.5 or
            Haiku 4.5, which reject it.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        temperature: float | None = 0.2,
        effort: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.effort = effort
        self.api_key = api_key
        self._client: "anthropic.Anthropic | None" = None

    def _request_kwargs(self, system: str, user: str) -> dict[str, Any]:
        """Build the Messages API payload, omitting params this model rejects."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None and accepts_sampling_params(self.model):
            kwargs["temperature"] = self.temperature
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def _load(self) -> None:
        from ..env import require_env
        api_key = self.api_key or require_env("ANTHROPIC_API_KEY")
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        if not self._client:
            self._load()
        assert self._client is not None
        msg = self._client.messages.create(**self._request_kwargs(system, user))
        text = "".join(getattr(block, "text", "") for block in msg.content)
        if not text:
            raise ValueError("Anthropic response contained no text content")
        return text

    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        if not self._client:
            self._load()
        assert self._client is not None
        with self._client.messages.stream(**self._request_kwargs(system, user)) as stream:
            yield from stream.text_stream
