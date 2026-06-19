from .base import BaseModelProvider


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
        self._client = None

    def _load(self):
        # TODO:
        # import anthropic, os
        # self._client = anthropic.Anthropic(api_key=self.api_key or os.environ["ANTHROPIC_API_KEY"])
        raise NotImplementedError("AnthropicProvider not yet implemented")

    def complete(self, system: str, user: str) -> str:
        # TODO:
        # if not self._client: self._load()
        # msg = self._client.messages.create(
        #     model=self.model,
        #     max_tokens=self.max_tokens,
        #     temperature=self.temperature,
        #     system=system,
        #     messages=[{"role": "user", "content": user}],
        # )
        # return msg.content[0].text
        raise NotImplementedError
