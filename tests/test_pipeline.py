"""Tests for the ingest and query pipelines."""
import pytest
from src.pipeline.query import build_augmented_prompt
from src.pipeline.ingest import chunk_text


def test_build_augmented_prompt_includes_query():
    prompt = build_augmented_prompt("What is X?", ["Context about X.", "More about X."])
    assert "What is X?" in prompt
    assert "Context about X." in prompt


def test_chunk_text_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        chunk_text("some text", chunk_size=100)


# TODO: add integration tests once pipeline is implemented
