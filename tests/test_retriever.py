"""Tests for the retriever and vector store."""
import pytest
from src.retriever.vector_store import Document, SearchResult, VectorStore


def test_vector_store_raises_on_search_before_add():
    store = VectorStore()
    with pytest.raises((NotImplementedError, Exception)):
        store.search([0.1, 0.2, 0.3], top_k=3)


# TODO: add tests once VectorStore is implemented
# def test_add_and_search_returns_top_k():
#     store = VectorStore()
#     docs = [Document(text=f"doc {i}", source="test.txt", chunk_index=i) for i in range(10)]
#     vecs = [[float(i)] * 128 for i in range(10)]
#     store.add(docs, vecs)
#     results = store.search([0.0] * 128, top_k=3)
#     assert len(results) == 3
#     assert all(isinstance(r, SearchResult) for r in results)
