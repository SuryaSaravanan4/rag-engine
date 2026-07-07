from typing import TYPE_CHECKING

from .base import BaseEmbedder
from ..env import require_env

if TYPE_CHECKING:
    import openai


class OpenAIEmbedder(BaseEmbedder):
    """Embedder backed by the OpenAI Embeddings API.

    Args:
        model: OpenAI embedding model name.
               Default: text-embedding-3-small (1536-dim, cheap).
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._client: "openai.OpenAI | None" = None  # lazy-loaded on first use

    def _load(self) -> None:
        api_key = self.api_key or require_env("OPENAI_API_KEY")
        import openai
        self._client = openai.OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks via the OpenAI Embeddings API."""
        if self._client is None:
            self._load()
        assert self._client is not None
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string via the OpenAI Embeddings API."""
        if self._client is None:
            self._load()
        assert self._client is not None
        response = self._client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
