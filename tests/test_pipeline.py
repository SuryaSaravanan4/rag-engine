"""Tests for the ingest and query pipelines."""
import importlib

import pytest

from src.pipeline.ingest import chunk_text, ingest
from src.pipeline.query import build_augmented_prompt, query, stream_query
from src.retriever.vector_store import VectorStore

# src/pipeline/__init__.py does `from .ingest import ingest`, which rebinds the
# `ingest` attribute on the `src.pipeline` package to that function — shadowing
# the `src.pipeline.ingest` *module*. Attribute-chain access (and monkeypatch's
# string-path resolution) would hit the function, so fetch the real modules via
# sys.modules (what importlib.import_module returns) instead.
ingest_module = importlib.import_module("src.pipeline.ingest")
query_module = importlib.import_module("src.pipeline.query")


def test_build_augmented_prompt_includes_query_and_context():
    prompt = build_augmented_prompt("What is X?", ["Context about X.", "More about X."])
    assert "What is X?" in prompt
    assert "Context about X." in prompt
    assert "More about X." in prompt


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_splits_with_overlap():
    text = "a" * 10
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    # step = 4: windows at 0, 4, 8 -> "aaaaaa", "aaaaaa", "aa"
    assert chunks == ["aaaaaa", "aaaaaa", "aa"]


def test_chunk_text_drops_empty_trailing_chunks():
    assert chunk_text("hello", chunk_size=5, overlap=0) == ["hello"]


class _FakeEmbedder:
    """Deterministic embedder: vector = [len(text)] * 4."""

    def embed_documents(self, texts):
        return [[float(len(t))] * 4 for t in texts]

    def embed_query(self, text):
        return [float(len(text))] * 4


class _FakeProvider:
    def __init__(self):
        self.last_call = None

    def complete(self, system, user):
        self.last_call = (system, user)
        return "the answer"

    def stream_complete(self, system, user):
        self.last_call = (system, user)
        yield "the "
        yield "answer"


def _config(tmp_path, index_dir):
    return {
        "embedder": {"provider": "local", "model": "all-MiniLM-L6-v2", "openai_model": "text-embedding-3-small"},
        "retriever": {"top_k": 2, "chunk_size": 20, "chunk_overlap": 0},
        "llm": {"provider": "anthropic", "model": "claude-test", "max_tokens": 100, "temperature": 0.0},
        "pipeline": {"data_dir": str(tmp_path), "index_dir": str(index_dir)},
    }


def test_ingest_chunks_embeds_and_persists_index(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("This is a short test document about widgets.", encoding="utf-8")
    index_dir = tmp_path / "processed"
    config = _config(tmp_path, index_dir)

    monkeypatch.setattr(ingest_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())

    ingest(str(tmp_path), config)

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "documents.pkl").exists()

    store = VectorStore(index_dir=str(index_dir))
    store.load()
    assert not store.is_empty()


def test_ingest_raises_on_missing_input_dir(tmp_path):
    config = _config(tmp_path, tmp_path / "processed")
    with pytest.raises(FileNotFoundError):
        ingest(str(tmp_path / "does-not-exist"), config)


def test_ingest_uses_relative_path_as_source_for_same_named_files(tmp_path, monkeypatch):
    # Two different files named "notes.md" in different subdirectories must not
    # collapse to the same `source` — otherwise --source filtering can't tell them apart.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "notes.md").write_text("Notes from A about widgets.", encoding="utf-8")
    (tmp_path / "b" / "notes.md").write_text("Notes from B about gadgets.", encoding="utf-8")
    index_dir = tmp_path / "processed"
    config = _config(tmp_path, index_dir)

    monkeypatch.setattr(ingest_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())
    ingest(str(tmp_path), config)

    store = VectorStore(index_dir=str(index_dir))
    store.load()
    sources = {doc.source for doc in store._documents}
    assert sources == {"a/notes.md", "b/notes.md"}


def test_query_retrieves_context_and_calls_provider(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("Widgets are small mechanical parts used in gadgets.", encoding="utf-8")
    index_dir = tmp_path / "processed"
    config = _config(tmp_path, index_dir)

    monkeypatch.setattr(ingest_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())
    ingest(str(tmp_path), config)

    monkeypatch.setattr(query_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())
    fake_provider = _FakeProvider()
    monkeypatch.setattr(query_module, "get_provider", lambda provider, **kwargs: fake_provider)

    answer = query("What are widgets?", config)

    assert answer == "the answer"
    system, user = fake_provider.last_call
    assert "What are widgets?" in user


def test_stream_query_yields_pieces_from_provider(tmp_path, monkeypatch):
    (tmp_path / "doc.txt").write_text("Widgets are small mechanical parts used in gadgets.", encoding="utf-8")
    index_dir = tmp_path / "processed"
    config = _config(tmp_path, index_dir)

    monkeypatch.setattr(ingest_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())
    ingest(str(tmp_path), config)

    monkeypatch.setattr(query_module, "get_embedder", lambda provider, **kwargs: _FakeEmbedder())
    fake_provider = _FakeProvider()
    monkeypatch.setattr(query_module, "get_provider", lambda provider, **kwargs: fake_provider)

    pieces = list(stream_query("What are widgets?", config))

    assert pieces == ["the ", "answer"]
