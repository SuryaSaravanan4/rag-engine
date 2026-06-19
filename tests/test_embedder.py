"""Tests for the embedder module."""
import pytest
from src.embedder.base import BaseEmbedder


def test_base_embedder_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_local_embedder_raises_not_implemented():
    from src.embedder.local import LocalEmbedder
    e = LocalEmbedder()
    with pytest.raises(NotImplementedError):
        e.embed_documents(["hello"])


def test_openai_embedder_raises_not_implemented():
    from src.embedder.openai import OpenAIEmbedder
    e = OpenAIEmbedder()
    with pytest.raises(NotImplementedError):
        e.embed_documents(["hello"])


# TODO: add integration tests once implementations are complete
# def test_local_embedder_returns_vectors():
#     e = LocalEmbedder()
#     vecs = e.embed_documents(["hello world", "foo bar"])
#     assert len(vecs) == 2
#     assert all(isinstance(v, list) for v in vecs)
