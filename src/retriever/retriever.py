from ..embedder.base import BaseEmbedder
from .vector_store import VectorStore, SearchResult


class Retriever:
    """Ties an embedder to a vector store to answer top-k retrieval queries.
    
    Args:
        embedder: Any BaseEmbedder implementation.
        store: A populated VectorStore.
        top_k: Number of chunks to return per query.
    """

    def __init__(self, embedder: BaseEmbedder, store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[SearchResult]:
        """Embed the query and return the top_k most relevant document chunks.
        
        Args:
            query: Raw natural language question from the user.

        Returns:
            Ranked list of SearchResult objects.
        """
        # TODO:
        # 1. query_vector = self.embedder.embed_query(query)
        # 2. return self.store.search(query_vector, self.top_k)
        raise NotImplementedError
