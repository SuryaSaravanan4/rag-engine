"""Query pipeline — retrieve context, augment prompt, generate answer.

Usage:
    python -m src.pipeline.query "What does the /users endpoint return?"
"""
from __future__ import annotations
import argparse
from typing import Iterator

from ..embedder import get_embedder
from ..retriever.vector_store import VectorStore
from ..retriever.retriever import Retriever
from ..llm import get_provider


SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions
based strictly on the provided context. If the context does not contain
enough information to answer the question, say so clearly rather than guessing.
Do not reference information outside the provided context."""


def build_augmented_prompt(query: str, context_chunks: list[str]) -> str:
    """Combine retrieved chunks with the user query into a single prompt."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def _build_prompt(question: str, config: dict, source: str | list[str] | None) -> str:
    """Embed the question, retrieve context, and build the augmented prompt."""
    emb_cfg = config["embedder"]
    emb_provider = emb_cfg["provider"]
    emb_kwargs = {"model_name": emb_cfg["model"]} if emb_provider == "local" else {"model": emb_cfg["openai_model"]}
    embedder = get_embedder(emb_provider, **emb_kwargs)

    store = VectorStore(index_dir=config["pipeline"]["index_dir"])
    store.load()

    retriever = Retriever(embedder, store, top_k=config["retriever"]["top_k"])
    results = retriever.retrieve(question, source=source)
    chunks = [r.document.text for r in results]

    return build_augmented_prompt(question, chunks)


def _build_provider(config: dict):
    llm_cfg = config["llm"]
    llm_provider = llm_cfg["provider"]
    if llm_provider == "ollama":
        llm_kwargs = {
            "model": llm_cfg["model"],
            "base_url": llm_cfg.get("ollama_base_url", "http://localhost:11434"),
            "temperature": llm_cfg["temperature"],
            "max_tokens": llm_cfg.get("max_tokens"),
        }
    else:
        llm_kwargs = {
            "model": llm_cfg["model"],
            "max_tokens": llm_cfg["max_tokens"],
            "temperature": llm_cfg["temperature"],
        }
        # effort is Anthropic-only — the replacement for temperature on models
        # that no longer accept it. Same leaky-edge tradeoff as ollama above.
        if llm_provider == "anthropic" and llm_cfg.get("effort") is not None:
            llm_kwargs["effort"] = llm_cfg["effort"]
    return get_provider(llm_provider, **llm_kwargs)


def query(question: str, config: dict, source: str | list[str] | None = None) -> str:
    """Full RAG pipeline: embed query → retrieve → augment → generate.

    Args:
        question: Natural language question from the user.
        config: Parsed config.yaml as a dict.
        source: If given, restrict retrieval to chunk(s) from this source
            path (or list of paths), relative to the ingested input directory.

    Returns:
        The model's answer, grounded in retrieved context.
    """
    prompt = _build_prompt(question, config, source)
    provider = _build_provider(config)
    return provider.complete(SYSTEM_PROMPT, prompt)


def stream_query(question: str, config: dict, source: str | list[str] | None = None) -> Iterator[str]:
    """Same as query(), but yields the answer incrementally as it's generated."""
    prompt = _build_prompt(question, config, source)
    provider = _build_provider(config)
    yield from provider.stream_complete(SYSTEM_PROMPT, prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG engine.")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", help="Restrict retrieval to chunks from this source path (relative to --input, e.g. subdir/file.md)")
    parser.add_argument("--stream", action="store_true", help="Stream the answer as it's generated")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.stream:
        for piece in stream_query(args.question, config, source=args.source):
            print(piece, end="", flush=True)
        print()
    else:
        answer = query(args.question, config, source=args.source)
        print(answer)
