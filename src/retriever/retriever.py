from __future__ import annotations

from ..embedder.base import BaseEmbedder
from .vector_store import VectorStore, SearchResult


class Retriever:
    """Ties an embedder to a vector store to answer top-k retrieval queries."""

    def __init__(self, embedder: BaseEmbedder, store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[SearchResult]:
        """Embed the query and return the top_k most relevant document chunks.

        Returns an empty list when the index is empty so callers don't need
        to guard against a cold-start index.

        Args:
            query: Natural-language question or code snippet to search against.

        Returns:
            Ranked list of SearchResult objects (closest first).
        """
        if self.store.is_empty():
            return []

        query_vector = self.embedder.embed_query(query)
        return self.store.search(query_vector, top_k=self.top_k)
