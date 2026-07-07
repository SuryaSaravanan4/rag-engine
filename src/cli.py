"""Unified CLI for rag-engine.

Usage:
    rag ingest --input data/raw/
    rag query "What does the /users endpoint return?"
    rag query "..." --stream --source api-spec.md
"""
from __future__ import annotations
import argparse

import yaml

from .pipeline.ingest import ingest
from .pipeline.query import query, stream_query


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _run_ingest(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    ingest(args.input, config)


def _run_query(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.stream:
        for piece in stream_query(args.question, config, source=args.source):
            print(piece, end="", flush=True)
        print()
    else:
        print(query(args.question, config, source=args.source))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Retrieval-augmented generation over your own documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a folder of documents into the vector index")
    ingest_parser.add_argument("--input", required=True, help="Directory of source documents (.txt, .md, .pdf)")
    ingest_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ingest_parser.set_defaults(func=_run_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question against the ingested corpus")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    query_parser.add_argument("--source", help="Restrict retrieval to chunks from this source path (relative to --input, e.g. subdir/file.md)")
    query_parser.add_argument("--stream", action="store_true", help="Stream the answer as it's generated")
    query_parser.set_defaults(func=_run_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
