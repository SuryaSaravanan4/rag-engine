"""Tests for the embedder module."""
import sys
import types

import pytest

from src.embedder import get_embedder
from src.embedder.base import BaseEmbedder


def test_base_embedder_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        get_embedder("cohere")


def test_local_embedder_encodes_documents_and_query(monkeypatch):
    from src.embedder.local import LocalEmbedder

    class _FakeArray(list):
        """Minimal numpy-array stand-in: supports .tolist() and indexing to a sub-array."""

        def tolist(self):
            return [row.tolist() if isinstance(row, _FakeArray) else row for row in self]

        def __getitem__(self, item):
            result = list.__getitem__(self, item)
            return _FakeArray(result) if isinstance(result, list) else result

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts, convert_to_numpy=True):
            return _FakeArray([_FakeArray([float(len(t))] * 3) for t in texts])

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    docs = embedder.embed_documents(["hello", "hi there"])
    assert docs == [[5.0, 5.0, 5.0], [8.0, 8.0, 8.0]]

    query_vec = embedder.embed_query("hello")
    assert query_vec == [5.0, 5.0, 5.0]


def test_openai_embedder_raises_clear_error_without_api_key(monkeypatch):
    from src.embedder.openai import OpenAIEmbedder

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAIEmbedder()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        embedder.embed_documents(["hello"])


def test_openai_embedder_calls_embeddings_api(monkeypatch):
    from src.embedder.openai import OpenAIEmbedder

    captured = {}

    class FakeEmbeddings:
        def create(self, input, model):
            captured["input"] = input
            captured["model"] = model
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])

    class FakeClient:
        embeddings = FakeEmbeddings()

    fake_openai = types.SimpleNamespace(OpenAI=lambda api_key: FakeClient())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    docs = embedder.embed_documents(["a", "b"])
    assert docs == [[0.1, 0.2], [0.1, 0.2]]
    assert captured["model"] == "text-embedding-3-small"

    query_vec = embedder.embed_query("a")
    assert query_vec == [0.1, 0.2]
