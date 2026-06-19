"""Tests for LLM provider backends."""
import pytest
from src.llm import get_provider


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("grok")


def test_anthropic_provider_raises_not_implemented():
    from src.llm.anthropic import AnthropicProvider
    p = AnthropicProvider()
    with pytest.raises(NotImplementedError):
        p.complete("system", "user")


def test_openai_provider_raises_not_implemented():
    from src.llm.openai import OpenAIProvider
    p = OpenAIProvider()
    with pytest.raises(NotImplementedError):
        p.complete("system", "user")


# TODO: add mock-based tests once implementations are complete
# def test_anthropic_complete_calls_api(mocker):
#     mock_client = mocker.patch("anthropic.Anthropic")
#     ...
