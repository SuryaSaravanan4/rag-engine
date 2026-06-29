"""Query pipeline — retrieve context, augment prompt, generate answer.

Usage:
    python -m src.pipeline.query "What does the /users endpoint return?"
"""
from __future__ import annotations
import argparse

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


def query(question: str, config: dict) -> str:
    """Full RAG pipeline: embed query → retrieve → augment → generate.
    
    Args:
        question: Natural language question from the user.
        config: Parsed config.yaml as a dict.

    Returns:
        The model's answer, grounded in retrieved context.
    """
    emb_cfg = config["embedder"]
    emb_provider = emb_cfg["provider"]
    emb_kwargs = {"model_name": emb_cfg["model"]} if emb_provider == "local" else {"model": emb_cfg["openai_model"]}
    embedder = get_embedder(emb_provider, **emb_kwargs)

    store = VectorStore(index_dir=config["pipeline"]["index_dir"])
    store.load()

    retriever = Retriever(embedder, store, top_k=config["retriever"]["top_k"])
    results = retriever.retrieve(question)
    chunks = [r.document.text for r in results]

    prompt = build_augmented_prompt(question, chunks)

    llm_cfg = config["llm"]
    llm_provider = llm_cfg["provider"]
    if llm_provider == "ollama":
        llm_kwargs = {
            "model": llm_cfg["model"],
            "base_url": llm_cfg.get("ollama_base_url", "http://localhost:11434"),
            "temperature": llm_cfg["temperature"],
        }
    else:
        llm_kwargs = {
            "model": llm_cfg["model"],
            "max_tokens": llm_cfg["max_tokens"],
            "temperature": llm_cfg["temperature"],
        }
    provider = get_provider(llm_provider, **llm_kwargs)
    return provider.complete(SYSTEM_PROMPT, prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG engine.")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    answer = query(args.question, config)
    print(answer)
