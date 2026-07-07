from abc import ABC, abstractmethod
from typing import Iterator


class BaseModelProvider(ABC):
    """Abstract interface for LLM backends.

    Isolates provider-specific request/response handling behind
    a single interface — making backends runtime-swappable via config.
    """

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Send a prompt to the model and return the response text.

        Args:
            system: System prompt (instructions, persona, constraints).
            user: User message (the augmented query with retrieved context).

        Returns:
            The model's response as a plain string.
        """
        ...

    @abstractmethod
    def stream_complete(self, system: str, user: str) -> Iterator[str]:
        """Send a prompt to the model and yield the response incrementally.

        Args:
            system: System prompt (instructions, persona, constraints).
            user: User message (the augmented query with retrieved context).

        Yields:
            Successive text fragments as they arrive from the backend.
        """
        ...
