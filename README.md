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
- **CLI** — ingest a folder of `.txt`/`.md`/`.pdf` files and query from the terminal

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-engine.git
cd rag-engine
pip install -r requirements.txt

# 2. Set your API key (or use Ollama for fully local)
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Ingest a corpus
python -m src.pipeline.ingest --input data/raw/

# 4. Query
python -m src.pipeline.query "What does the /users endpoint return?"
```

---

## Project Structure

```
rag-engine/
├── src/
│   ├── embedder/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract Embedder interface
│   │   ├── local.py         # sentence-transformers embedder
│   │   └── openai.py        # OpenAI embeddings API
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── vector_store.py  # FAISS wrapper (index, search, persist)
│   │   └── retriever.py     # Top-k retriever logic
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract ModelProvider interface
│   │   ├── openai.py        # OpenAI chat completions
│   │   ├── anthropic.py     # Anthropic Messages API
│   │   └── ollama.py        # Local Ollama backend
│   └── pipeline/
│       ├── __init__.py
│       ├── ingest.py        # Document loading + chunking + embedding
│       └── query.py         # Query → retrieve → augment → generate
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
- [ ] Embedder module (local + OpenAI)
- [ ] FAISS vector store wrapper
- [ ] Document ingestion pipeline (txt, md, pdf)
- [ ] Pluggable LLM backends
- [ ] End-to-end query pipeline
- [ ] CLI interface
- [ ] Unit tests
- [ ] Streaming responses
- [ ] Metadata filtering on retrieval

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
