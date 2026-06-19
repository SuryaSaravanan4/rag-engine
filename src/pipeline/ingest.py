"""Document ingestion pipeline.

Usage:
    python -m src.pipeline.ingest --input data/raw/ --config config.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path


def load_text_file(path: Path) -> str:
    """Read a plain text or markdown file."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    # TODO:
    # import fitz
    # doc = fitz.open(str(path))
    # return "\n".join(page.get_text() for page in doc)
    raise NotImplementedError("PDF loading not yet implemented")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters.
    
    Overlap ensures that context spanning a chunk boundary isn't lost.
    """
    # TODO: sliding window split — step = chunk_size - overlap
    raise NotImplementedError


def ingest(input_dir: str, config: dict) -> None:
    """Load all documents from input_dir, chunk, embed, and persist the index.
    
    Args:
        input_dir: Path to folder containing .txt, .md, or .pdf files.
        config: Parsed config.yaml as a dict.
    """
    # TODO:
    # 1. Walk input_dir, dispatch to load_text_file or load_pdf
    # 2. chunk_text each document
    # 3. embedder = get_embedder(config["embedder"]["provider"], ...)
    # 4. vectors = embedder.embed_documents(all_chunks)
    # 5. store = VectorStore(config["pipeline"]["index_dir"])
    # 6. store.add(documents, vectors)
    # 7. store.save()
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--input", required=True, help="Directory of source documents")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    ingest(args.input, config)
    print("Ingestion complete.")
