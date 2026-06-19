from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Document:
    """A chunked piece of source text with metadata."""
    text: str
    source: str          # filename or URL the chunk came from
    chunk_index: int     # position of this chunk within the source


@dataclass
class SearchResult:
    """A retrieved document chunk with its similarity score."""
    document: Document
    score: float         # lower = more similar for L2; higher for inner product


class VectorStore:
    """FAISS-backed vector store for document chunks.
    
    Supports adding documents, persisting the index to disk,
    loading it back, and top-k similarity search.
    """

    def __init__(self, index_dir: str = "data/processed"):
        self.index_dir = index_dir
        self._index = None       # faiss.Index — populated on add() or load()
        self._documents: list[Document] = []
        self._dim: int | None = None

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Add document chunks and their embedding vectors to the index.
        
        Args:
            documents: Chunked Document objects matching vectors 1-to-1.
            vectors: Dense float vectors from the embedder.
        """
        # TODO:
        # 1. On first call, infer dim and create faiss.IndexFlatL2(dim)
        # 2. Convert vectors to np.float32 array
        # 3. self._index.add(array)
        # 4. Append documents to self._documents
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top_k most similar documents to the query vector.
        
        Args:
            query_vector: Embedded query from the embedder.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by similarity (best first).
        """
        # TODO:
        # 1. Convert query_vector to np.float32 array shaped (1, dim)
        # 2. distances, indices = self._index.search(array, top_k)
        # 3. Map indices back to self._documents
        # 4. Return list of SearchResult(document, score)
        raise NotImplementedError

    def save(self) -> None:
        """Persist the FAISS index and document metadata to index_dir."""
        # TODO: faiss.write_index + pickle self._documents
        raise NotImplementedError

    def load(self) -> None:
        """Load a previously saved index and document list from index_dir."""
        # TODO: faiss.read_index + unpickle self._documents
        raise NotImplementedError
