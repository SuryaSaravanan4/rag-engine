from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class Document:
    """A chunked piece of source text with metadata."""
    text: str
    source: str
    chunk_index: int


@dataclass
class SearchResult:
    """A retrieved document chunk with its similarity score."""
    document: Document
    score: float


class VectorStore:
    """FAISS-backed vector store for document chunks.

    Supports adding documents, persisting the index to disk,
    loading it back, and top-k similarity search.
    """

    _INDEX_FILE = "index.faiss"
    _DOCS_FILE = "documents.pkl"

    def __init__(self, index_dir: str = "data/processed"):
        self.index_dir = index_dir
        self._index: faiss.Index | None = None
        self._documents: list[Document] = []
        self._dim: int | None = None

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Add document chunks and their embedding vectors to the index."""
        array = np.array(vectors, dtype=np.float32)
        if self._index is None:
            self._dim = array.shape[1]
            self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(array)
        self._documents.extend(documents)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return the top_k most similar documents to the query vector."""
        if self._index is None or self._index.ntotal == 0:
            return []
        array = np.array([query_vector], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(array, k)
        return [
            SearchResult(document=self._documents[i], score=float(distances[0][rank]))
            for rank, i in enumerate(indices[0])
            if i != -1
        ]

    def save(self) -> None:
        """Persist the FAISS index and document metadata to index_dir."""
        path = Path(self.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / self._INDEX_FILE))
        with open(path / self._DOCS_FILE, "wb") as f:
            pickle.dump(self._documents, f)

    def load(self) -> None:
        """Load a previously saved index and document list from index_dir."""
        path = Path(self.index_dir)
        self._index = faiss.read_index(str(path / self._INDEX_FILE))
        with open(path / self._DOCS_FILE, "rb") as f:
            self._documents = pickle.load(f)
        if self._index.ntotal > 0:
            self._dim = self._index.d
