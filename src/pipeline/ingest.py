"""Document ingestion pipeline.

Usage:
    python -m src.pipeline.ingest --input data/raw/ --config config.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ..embedder import get_embedder
from ..retriever.vector_store import VectorStore, Document


def load_text_file(path: Path) -> str:
    """Read a plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters.

    Uses a sliding window with step = chunk_size - overlap so context
    spanning a chunk boundary is preserved in both adjacent chunks.

    Args:
        text: Source text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks; final chunk may be shorter than chunk_size.
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    step = chunk_size - overlap
    return [c for c in (text[i:i + chunk_size] for i in range(0, len(text), step)) if c.strip()]


def ingest(input_dir: str, config: dict) -> None:
    """Load all documents from input_dir, chunk, embed, and persist the index.

    Args:
        input_dir: Path to folder containing .txt, .md, or .pdf files.
        config: Parsed config.yaml as a dict.
    """
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    chunk_size = config["retriever"]["chunk_size"]
    chunk_overlap = config["retriever"]["chunk_overlap"]

    all_documents: list[Document] = []
    all_texts: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            raw = load_text_file(path)
        elif suffix == ".pdf":
            raw = load_pdf(path)
        else:
            continue

        for i, chunk in enumerate(chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)):
            all_documents.append(Document(text=chunk, source=path.name, chunk_index=i))
            all_texts.append(chunk)

    if not all_texts:
        print("No supported files found in input directory.")
        return

    emb_cfg = config["embedder"]
    provider = emb_cfg["provider"]
    # Config uses "model" for local and "openai_model" for openai — map to constructor kwarg.
    kwargs = {"model_name": emb_cfg["model"]} if provider == "local" else {"model": emb_cfg["openai_model"]}
    embedder = get_embedder(provider, **kwargs)

    vectors = embedder.embed_documents(all_texts)
    store = VectorStore(index_dir=config["pipeline"]["index_dir"])
    store.add(all_documents, vectors)
    store.save()
    print(f"Ingested {len(all_texts)} chunks from {len(set(d.source for d in all_documents))} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--input", required=True, help="Directory of source documents")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    ingest(args.input, config)
