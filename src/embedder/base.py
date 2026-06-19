from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract interface for all embedder backends.
    
    Any embedder must implement embed_documents and embed_query.
    This keeps the retriever and pipeline decoupled from the
    specific embedding provider.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks.
        
        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense float vectors, one per input text.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.
        
        Args:
            text: The query to embed.

        Returns:
            A single dense float vector.
        """
        ...
