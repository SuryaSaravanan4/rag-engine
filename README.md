# rag-engine

A lightweight, modular Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python.

Feed it a corpus of documents. Ask it a question. It retrieves the most relevant context and conditions an LLM's response on real source material — no hallucinated endpoints, no parametric guessing.

---

## Why I Built This

At CoverMyMeds I designed a query layer that conditioned LLM responses on retrieved OpenAPI specs rather than parametric memory, constraining outputs to real, current API definitions. This project is that same architecture, extracted and generalized — a clean implementation I can reason about, extend, and share publicly.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │            rag-engine                │
                        └─────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                           │
     ┌────────▼────────┐      ┌─────────▼──────────┐    ┌──────────▼────────┐
     │    Embedder      │      │     Retriever       │    │   LLM Provider    │
     │                  │      │                     │    │                   │
     │ Converts docs +  │      │ Vector store (FAISS)│    │ Pluggable backend │
     │ queries into     │─────▶│ Top-k similarity    │───▶│ OpenAI / Anthropic│
     │ dense vectors    │      │ search              │    │ / Ollama          │
     └──────────────────┘      └─────────────────────┘    └───────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │      Pipeline        │
                              │                      │
                              │ Orchestrates embed → │
                              │ retrieve → augment → │
                              │ generate             │
                              └──────────────────────┘
```

---

## Features

- **Modular embedder** — swap between local (`sentence-transformers`) and API-based (`OpenAI text-embedding-3-small`) embeddings via config
- **FAISS vector store** — fast approximate nearest-neighbor search, persisted to disk
- **Pluggable LLM backends** — one config flag switches between OpenAI, Anthropic Claude, and local Ollama
- **Clean pipeline interface** — `pipeline.query(question)` handles the full retrieve → augment → generate flow
- **Streaming responses** — `--stream` prints tokens as they're generated instead of waiting for the full answer
- **Metadata filtering** — `--source` restricts retrieval to chunks from a specific ingested file
- **CLI** — a single `rag` command (`rag ingest`, `rag query`) for ingesting a folder of `.txt`/`.md`/`.pdf` files and querying from the terminal

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-engine.git
cd rag-engine
pip install -r requirements.txt
pip install -e .   # installs the `rag` CLI command

# 2. Set your API key (or use Ollama for fully local)
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Ingest a corpus
rag ingest --input data/raw/

# 4. Query
rag query "What does the /users endpoint return?"

# Stream the answer, or restrict retrieval to one source file
rag query "What does the /users endpoint return?" --stream
rag query "What does the /users endpoint return?" --source openapi.yaml
```

Without installing the package, the same commands work as modules:
`python -m src.pipeline.ingest --input data/raw/` and `python -m src.pipeline.query "..."`.

---

## Project Structure

```
rag-engine/
├── src/
│   ├── cli.py                # Unified `rag` CLI (ingest / query subcommands)
│   ├── embedder/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract Embedder interface
│   │   ├── local.py         # sentence-transformers embedder
│   │   └── openai.py        # OpenAI embeddings API
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── vector_store.py  # FAISS wrapper (index, search, persist, source filtering)
│   │   └── retriever.py     # Top-k retriever logic
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract ModelProvider interface (complete + stream_complete)
│   │   ├── openai.py        # OpenAI chat completions
│   │   ├── anthropic.py     # Anthropic Messages API
│   │   └── ollama.py        # Local Ollama backend
│   └── pipeline/
│       ├── __init__.py
│       ├── ingest.py        # Document loading + chunking + embedding
│       └── query.py         # Query → retrieve → augment → generate (+ streaming)
├── tests/
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_llm.py
│   └── test_pipeline.py
├── data/
│   ├── raw/                 # Drop source documents here
│   └── processed/           # FAISS index + metadata stored here
├── docs/
│   └── design.md            # Architecture decisions + tradeoffs
├── config.yaml              # Runtime config (model, top-k, chunk size)
├── pyproject.toml            # Packaging + `rag` console script entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

Edit `config.yaml` to switch providers without touching code:

```yaml
embedder:
  provider: local              # local | openai
  model: all-MiniLM-L6-v2     # ignored if provider: openai

retriever:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 64

llm:
  provider: anthropic          # openai | anthropic | ollama
  model: claude-sonnet-4-6
  max_tokens: 1024
  temperature: 0.2
```

---

## Roadmap

- [x] Project scaffold + architecture
- [x] Embedder module (local + OpenAI)
- [x] FAISS vector store wrapper
- [x] Document ingestion pipeline (txt, md, pdf)
- [x] Pluggable LLM backends
- [x] End-to-end query pipeline
- [x] CLI interface
- [x] Unit tests
- [x] Streaming responses
- [x] Metadata filtering on retrieval

---

## Tech Stack

- **Python 3.11+**
- **FAISS** — vector similarity search
- **sentence-transformers** — local embeddings
- **OpenAI / Anthropic Python SDKs** — LLM backends
- **PyMuPDF** — PDF parsing
- **PyYAML** — config

---

## License

MIT
