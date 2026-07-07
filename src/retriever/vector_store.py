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

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source: str | list[str] | None = None,
    ) -> list[SearchResult]:
        """Return the top_k most similar documents to the query vector.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Maximum number of results to return.
            source: If given, restrict results to document(s) whose `source`
                matches this path (or is in this list of paths). Since FAISS's
                flat index has no native filtering, this over-searches the
                full index and filters the ranked results — O(ntotal) per
                query rather than O(top_k log ntotal), so filtered queries
                won't scale as well as unfiltered ones on very large corpora.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        allowed = {source} if isinstance(source, str) else (set(source) if source is not None else None)
        array = np.array([query_vector], dtype=np.float32)
        k = self._index.ntotal if allowed is not None else min(top_k, self._index.ntotal)
        distances, indices = self._index.search(array, k)
        results = []
        for rank, i in enumerate(indices[0]):
            if i == -1:
                continue
            document = self._documents[i]
            if allowed is not None and document.source not in allowed:
                continue
            results.append(SearchResult(document=document, score=float(distances[0][rank])))
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        """Persist the FAISS index and document metadata to index_dir."""
        if self._index is None:
            raise ValueError("Cannot save an empty VectorStore — call add() first.")
        path = Path(self.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / self._INDEX_FILE))
        with open(path / self._DOCS_FILE, "wb") as f:
            pickle.dump(self._documents, f)

    def is_empty(self) -> bool:
        """Return True if the index has no vectors."""
        return self._index is None or self._index.ntotal == 0

    def load(self) -> None:
        """Load a previously saved index and document list from index_dir."""
        path = Path(self.index_dir)
        self._index = faiss.read_index(str(path / self._INDEX_FILE))
        with open(path / self._DOCS_FILE, "rb") as f:
            self._documents = pickle.load(f)
        if self._index.ntotal > 0:
            self._dim = self._index.d
