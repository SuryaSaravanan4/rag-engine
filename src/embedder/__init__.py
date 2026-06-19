from .base import BaseEmbedder
from .local import LocalEmbedder
from .openai import OpenAIEmbedder


def get_embedder(provider: str, **kwargs) -> BaseEmbedder:
    """Factory function — returns the right embedder for the given provider string."""
    if provider == "local":
        return LocalEmbedder(**kwargs)
    elif provider == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder provider: '{provider}'. Choose 'local' or 'openai'.")


__all__ = ["BaseEmbedder", "LocalEmbedder", "OpenAIEmbedder", "get_embedder"]
