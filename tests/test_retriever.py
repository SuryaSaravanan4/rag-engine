"""Tests for the retriever and vector store."""
import pytest

from src.retriever.retriever import Retriever
from src.retriever.vector_store import Document, SearchResult, VectorStore


def _make_store():
    store = VectorStore()
    docs = [Document(text=f"doc {i}", source=f"file{i % 2}.txt", chunk_index=i) for i in range(6)]
    vecs = [[float(i)] * 4 for i in range(6)]
    store.add(docs, vecs)
    return store, docs


def test_vector_store_is_empty_before_add():
    store = VectorStore()
    assert store.is_empty()
    assert store.search([0.0, 0.0, 0.0], top_k=3) == []


def test_vector_store_save_raises_before_add(tmp_path):
    store = VectorStore(index_dir=str(tmp_path))
    with pytest.raises(ValueError, match="empty VectorStore"):
        store.save()


def test_vector_store_search_returns_top_k_closest():
    store, docs = _make_store()
    assert not store.is_empty()

    results = store.search([0.0] * 4, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    # Closest vector to [0,0,0,0] is doc 0, then doc 1, then doc 2.
    assert [r.document.chunk_index for r in results] == [0, 1, 2]
    assert results[0].score <= results[1].score <= results[2].score


def test_vector_store_search_filters_by_source():
    store, docs = _make_store()

    results = store.search([0.0] * 4, top_k=3, source="file0.txt")

    assert len(results) == 3
    assert all(r.document.source == "file0.txt" for r in results)


def test_vector_store_search_filters_by_source_list():
    store, docs = _make_store()

    results = store.search([0.0] * 4, top_k=10, source=["file0.txt"])

    assert all(r.document.source == "file0.txt" for r in results)
    assert len(results) == 3  # only 3 of the 6 docs are file0.txt


class _StubEmbedder:
    def embed_query(self, text):
        return [0.0] * 4


def test_retriever_returns_empty_list_for_empty_store():
    store = VectorStore()
    retriever = Retriever(_StubEmbedder(), store, top_k=5)
    assert retriever.retrieve("anything") == []


def test_retriever_delegates_to_store_search():
    store, docs = _make_store()
    retriever = Retriever(_StubEmbedder(), store, top_k=2)

    results = retriever.retrieve("a question")

    assert len(results) == 2
    assert [r.document.chunk_index for r in results] == [0, 1]


def test_retriever_passes_source_filter_through():
    store, docs = _make_store()
    retriever = Retriever(_StubEmbedder(), store, top_k=2)

    results = retriever.retrieve("a question", source="file1.txt")

    assert all(r.document.source == "file1.txt" for r in results)
