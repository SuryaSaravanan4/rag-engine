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

    def _load(self):
        # TODO: import and load SentenceTransformer model
        # from sentence_transformers import SentenceTransformer
        # self._model = SentenceTransformer(self.model_name)
        raise NotImplementedError("LocalEmbedder not yet implemented")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # TODO: batch encode texts, return as list of lists
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        # TODO: encode single query, return as list
        raise NotImplementedError
