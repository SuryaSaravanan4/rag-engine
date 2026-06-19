"""Query pipeline — retrieve context, augment prompt, generate answer.

Usage:
    python -m src.pipeline.query "What does the /users endpoint return?"
"""
from __future__ import annotations
import argparse


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
    # TODO:
    # 1. embedder = get_embedder(config["embedder"]["provider"], ...)
    # 2. store = VectorStore(config["pipeline"]["index_dir"])
    # 3. store.load()
    # 4. retriever = Retriever(embedder, store, top_k=config["retriever"]["top_k"])
    # 5. results = retriever.retrieve(question)
    # 6. chunks = [r.document.text for r in results]
    # 7. prompt = build_augmented_prompt(question, chunks)
    # 8. provider = get_provider(config["llm"]["provider"], ...)
    # 9. return provider.complete(SYSTEM_PROMPT, prompt)
    raise NotImplementedError


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
