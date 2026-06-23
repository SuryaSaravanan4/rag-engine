from __future__ import annotations
from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """Embedder backed by a local sentence-transformers model.

    Runs entirely offline — no API key required. Slower than
    API embedders for large corpora but free and private.

    Args:
        model_name: Any sentence-transformers compatible model name.
                    Default: all-MiniLM-L6-v2 (fast, 384-dim).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded on first use

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode a list of document chunks."""
        self._load()
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string."""
        self._load()
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()
