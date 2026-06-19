from .base import BaseModelProvider


class OllamaProvider(BaseModelProvider):
    """LLM backend using a local Ollama instance.
    
    Fully offline — no API key needed. Requires Ollama running locally.
    https://ollama.com

    Args:
        model: Ollama model tag (e.g. "llama3", "mistral", "phi3").
        base_url: Ollama server URL. Default: http://localhost:11434.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        # TODO:
        # import requests
        # payload = {
        #     "model": self.model,
        #     "messages": [
        #         {"role": "system", "content": system},
        #         {"role": "user", "content": user},
        #     ],
        #     "options": {"temperature": self.temperature},
        #     "stream": False,
        # }
        # resp = requests.post(f"{self.base_url}/api/chat", json=payload)
        # resp.raise_for_status()
        # return resp.json()["message"]["content"]
        raise NotImplementedError("OllamaProvider not yet implemented")
