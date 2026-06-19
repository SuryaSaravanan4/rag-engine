from .base import BaseEmbedder


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
        self._client = None  # lazy-loaded on first use

    def _load(self):
        # TODO: instantiate openai.OpenAI client
        # import openai, os
        # self._client = openai.OpenAI(api_key=self.api_key or os.environ["OPENAI_API_KEY"])
        raise NotImplementedError("OpenAIEmbedder not yet implemented")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # TODO: call client.embeddings.create, extract .data[i].embedding
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        # TODO: same as embed_documents but single text
        raise NotImplementedError
