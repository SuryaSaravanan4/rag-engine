"""Tests for LLM provider backends."""
import sys
import types

import pytest

from src.llm import get_provider
from src.llm.ollama import OllamaProvider


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("grok")


def test_get_provider_dispatches_by_name():
    from src.llm.anthropic import AnthropicProvider
    from src.llm.openai import OpenAIProvider

    assert isinstance(get_provider("anthropic"), AnthropicProvider)
    assert isinstance(get_provider("openai"), OpenAIProvider)
    assert isinstance(get_provider("ollama"), OllamaProvider)


def test_anthropic_complete_raises_clear_error_without_api_key(monkeypatch):
    from src.llm.anthropic import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider.complete("system", "user")


def test_openai_complete_raises_clear_error_without_api_key(monkeypatch):
    from src.llm.openai import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider.complete("system", "user")


def test_anthropic_complete_calls_api(monkeypatch):
    from src.llm.anthropic import AnthropicProvider

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="hello from claude")])

    class FakeClient:
        messages = FakeMessages()

    fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: FakeClient())
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = AnthropicProvider(model="claude-test")
    result = provider.complete("system prompt", "user prompt")

    assert result == "hello from claude"
    assert captured["model"] == "claude-test"
    assert captured["system"] == "system prompt"
    assert captured["messages"] == [{"role": "user", "content": "user prompt"}]


def test_accepts_sampling_params_by_model():
    from src.llm.anthropic import accepts_sampling_params

    # Models that removed temperature — sending it returns a 400.
    assert not accepts_sampling_params("claude-opus-4-8")
    assert not accepts_sampling_params("claude-opus-4-7")
    assert not accepts_sampling_params("claude-sonnet-5")
    assert not accepts_sampling_params("claude-fable-5")

    # Models that still accept it, plus unknown strings (conservative default).
    assert accepts_sampling_params("claude-sonnet-4-6")
    assert accepts_sampling_params("claude-opus-4-6")
    assert accepts_sampling_params("claude-haiku-4-5")
    assert accepts_sampling_params("claude-test")


def _fake_anthropic_client(monkeypatch, captured):
    """Install a fake anthropic SDK that records create()/stream() kwargs."""

    class FakeStreamCtx:
        text_stream = iter(["ok"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok")])

        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStreamCtx()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=lambda api_key: FakeClient()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_anthropic_omits_temperature_for_models_that_reject_it(monkeypatch):
    """Regression: temperature on Opus 4.7+/Sonnet 5+ returns a 400 from the API."""
    from src.llm.anthropic import AnthropicProvider

    captured = {}
    _fake_anthropic_client(monkeypatch, captured)

    AnthropicProvider(model="claude-opus-4-8", temperature=0.2).complete("sys", "user")
    assert "temperature" not in captured

    # Streaming builds the same payload, so it must drop it too.
    captured.clear()
    list(AnthropicProvider(model="claude-opus-4-8", temperature=0.2).stream_complete("sys", "user"))
    assert "temperature" not in captured


def test_anthropic_sends_temperature_for_models_that_accept_it(monkeypatch):
    from src.llm.anthropic import AnthropicProvider

    captured = {}
    _fake_anthropic_client(monkeypatch, captured)

    AnthropicProvider(model="claude-sonnet-4-6", temperature=0.4).complete("sys", "user")
    assert captured["temperature"] == 0.4


def test_anthropic_sends_effort_only_when_set(monkeypatch):
    from src.llm.anthropic import AnthropicProvider

    captured = {}
    _fake_anthropic_client(monkeypatch, captured)

    AnthropicProvider(model="claude-opus-4-8").complete("sys", "user")
    assert "output_config" not in captured

    captured.clear()
    AnthropicProvider(model="claude-opus-4-8", effort="medium").complete("sys", "user")
    assert captured["output_config"] == {"effort": "medium"}


def test_anthropic_stream_complete_yields_text(monkeypatch):
    from src.llm.anthropic import AnthropicProvider

    class FakeStreamCtx:
        text_stream = iter(["hel", "lo"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeMessages:
        def stream(self, **kwargs):
            return FakeStreamCtx()

    class FakeClient:
        messages = FakeMessages()

    fake_anthropic = types.SimpleNamespace(Anthropic=lambda api_key: FakeClient())
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = AnthropicProvider()
    assert list(provider.stream_complete("sys", "user")) == ["hel", "lo"]


def test_openai_complete_calls_api(monkeypatch):
    from src.llm.openai import OpenAIProvider

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hello from gpt"))]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    fake_openai = types.SimpleNamespace(OpenAI=lambda api_key: FakeClient())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider(model="gpt-test")
    result = provider.complete("system prompt", "user prompt")

    assert result == "hello from gpt"
    assert captured["model"] == "gpt-test"
    assert captured["messages"][0] == {"role": "system", "content": "system prompt"}
    assert captured["messages"][1] == {"role": "user", "content": "user prompt"}


def test_openai_stream_complete_skips_empty_deltas(monkeypatch):
    from src.llm.openai import OpenAIProvider

    def fake_chunk(content):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))])

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return iter([fake_chunk("he"), fake_chunk(None), fake_chunk("llo")])

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    fake_openai = types.SimpleNamespace(OpenAI=lambda api_key: FakeClient())
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = OpenAIProvider()
    assert list(provider.stream_complete("sys", "user")) == ["he", "llo"]


def test_ollama_complete_posts_to_chat_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hello from llama"}}

    def fake_post(url, json, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    fake_requests = types.SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    result = provider.complete("system prompt", "user prompt")

    assert result == "hello from llama"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["stream"] is False


def test_ollama_complete_passes_timeout_and_num_predict(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hello"}}

    def fake_post(url, json, **kwargs):
        captured["json"] = json
        captured["kwargs"] = kwargs
        return FakeResponse()

    fake_requests = types.SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    provider = OllamaProvider(max_tokens=256, timeout=5.0)
    provider.complete("system prompt", "user prompt")

    assert captured["kwargs"]["timeout"] == 5.0
    assert captured["json"]["options"]["num_predict"] == 256


def test_ollama_complete_omits_num_predict_when_max_tokens_unset(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hello"}}

    def fake_post(url, json, **kwargs):
        captured["json"] = json
        return FakeResponse()

    fake_requests = types.SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    provider = OllamaProvider()
    provider.complete("system prompt", "user prompt")

    assert "num_predict" not in captured["json"]["options"]


def test_ollama_stream_complete_parses_ndjson_lines(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter([
                b'{"message": {"content": "hel"}}',
                b"",
                b'{"message": {"content": "lo"}}',
                b'{"message": {"content": ""}, "done": true}',
            ])

    def fake_post(url, json, **kwargs):
        assert kwargs.get("stream") is True
        assert json["stream"] is True
        return FakeResponse()

    fake_requests = types.SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    provider = OllamaProvider()
    assert list(provider.stream_complete("sys", "user")) == ["hel", "lo"]
