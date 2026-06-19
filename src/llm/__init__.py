from .base import BaseModelProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .ollama import OllamaProvider


def get_provider(provider: str, **kwargs) -> BaseModelProvider:
    """Factory — returns the right LLM backend for the given provider string."""
    if provider == "anthropic":
        return AnthropicProvider(**kwargs)
    elif provider == "openai":
        return OpenAIProvider(**kwargs)
    elif provider == "ollama":
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Choose 'anthropic', 'openai', or 'ollama'.")


__all__ = ["BaseModelProvider", "AnthropicProvider", "OpenAIProvider", "OllamaProvider", "get_provider"]
