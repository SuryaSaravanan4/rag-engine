from .base import BaseModelProvider


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
        self._client = None

    def _load(self):
        # TODO:
        # import openai, os
        # self._client = openai.OpenAI(api_key=self.api_key or os.environ["OPENAI_API_KEY"])
        raise NotImplementedError("OpenAIProvider not yet implemented")

    def complete(self, system: str, user: str) -> str:
        # TODO:
        # if not self._client: self._load()
        # resp = self._client.chat.completions.create(
        #     model=self.model,
        #     max_tokens=self.max_tokens,
        #     temperature=self.temperature,
        #     messages=[
        #         {"role": "system", "content": system},
        #         {"role": "user", "content": user},
        #     ],
        # )
        # return resp.choices[0].message.content
        raise NotImplementedError
