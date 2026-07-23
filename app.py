"""Gradio front-end for the RAG engine — a live, hosted demo.

Wraps the same modules the CLI uses (embedder → FAISS retriever → LLM backend)
in a web UI. Designed to run on Hugging Face Spaces:

- Embeddings are local (`sentence-transformers`), so retrieval needs no API key
  and no per-visitor cost — the FAISS index over the bundled demo corpus is
  built once at startup.
- Generation calls Claude by default, reading ANTHROPIC_API_KEY from the
  environment (set it as a Space secret). A visitor can instead paste their own
  Anthropic/OpenAI key in the UI. With no key at all, retrieval still runs and
  the demo shows the exact chunks it would have fed the model.

Run locally with:  python app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import yaml

from src.embedder import get_embedder
from src.llm.anthropic import AnthropicProvider
from src.llm.openai import OpenAIProvider
from src.pipeline.ingest import chunk_text, load_pdf, load_text_file
from src.pipeline.query import SYSTEM_PROMPT, build_augmented_prompt
from src.retriever.retriever import Retriever
from src.retriever.vector_store import Document, SearchResult, VectorStore

ROOT = Path(__file__).parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CORPUS_DIR = ROOT / "demo" / "corpus"

ANTHROPIC = "Anthropic (Claude)"
OPENAI = "OpenAI (GPT)"

# One local embedder, loaded once and reused for both indexing and queries so
# the sentence-transformers model is downloaded and held in memory a single time.
EMBEDDER = get_embedder("local", model_name=CONFIG["embedder"]["model"])

EXAMPLE_QUESTIONS = [
    "How do I make sure a network retry doesn't charge a customer twice?",
    "What's the difference between a refund and a dispute?",
    "How do I verify that a webhook actually came from Acme?",
    "Can I authorize a payment now and capture it later? For how long?",
    "What are the rate limits, and what should I do when I hit one?",
]


def _read(path: Path) -> str | None:
    """Load a supported document to text, or None if the type is unsupported."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf(path)
    return None


def build_store(files: list[Path]) -> tuple[VectorStore, int, int]:
    """Chunk, embed, and index a list of files into a fresh in-memory store.

    Returns the store plus (chunk count, source-file count) for status display.
    """
    chunk_size = CONFIG["retriever"]["chunk_size"]
    overlap = CONFIG["retriever"]["chunk_overlap"]

    documents: list[Document] = []
    texts: list[str] = []
    for path in files:
        raw = _read(path)
        if raw is None:
            continue
        for i, chunk in enumerate(chunk_text(raw, chunk_size=chunk_size, overlap=overlap)):
            documents.append(Document(text=chunk, source=path.name, chunk_index=i))
            texts.append(chunk)

    if not texts:
        raise gr.Error("No supported documents found (.txt, .md, .pdf).")

    vectors = EMBEDDER.embed_documents(texts)
    store = VectorStore()
    store.add(documents, vectors)
    return store, len(texts), len({d.source for d in documents})


def make_provider(provider_name: str, api_key: str) -> AnthropicProvider | OpenAIProvider | None:
    """Build the selected LLM backend, or None if no key is available.

    A key typed into the UI wins; otherwise fall back to the environment
    (a Space secret). Returns None when neither is set so the caller can
    degrade gracefully to retrieval-only.
    """
    key = (api_key or "").strip() or None
    llm_cfg = CONFIG["llm"]

    if provider_name == ANTHROPIC:
        key = key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        kwargs: dict = {
            "model": llm_cfg["model"],
            "max_tokens": llm_cfg["max_tokens"],
            "temperature": llm_cfg.get("temperature"),
            "api_key": key,
        }
        if llm_cfg.get("effort") is not None:
            kwargs["effort"] = llm_cfg["effort"]
        return AnthropicProvider(**kwargs)

    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAIProvider(
        model="gpt-4o-mini",
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg.get("temperature") or 0.2,
        api_key=key,
    )


def format_sources(results: list[SearchResult]) -> str:
    """Render retrieved chunks as markdown — the transparency panel of the demo."""
    blocks = []
    for rank, r in enumerate(results, 1):
        doc = r.document
        snippet = " ".join(doc.text.split())
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        blocks.append(
            f"**{rank}. `{doc.source}` — chunk {doc.chunk_index}**  \n"
            f"<sub>L2 distance {r.score:.3f} · lower is closer</sub>  \n\n{snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(question: str, store: VectorStore, provider_name: str, api_key: str, top_k: int):
    """Full RAG turn: retrieve top-k, then generate a grounded answer."""
    if not question or not question.strip():
        raise gr.Error("Enter a question first.")

    retriever = Retriever(EMBEDDER, store, top_k=int(top_k))
    results = retriever.retrieve(question)
    if not results:
        return "The index is empty — ingest some documents first.", ""

    sources_md = format_sources(results)
    prompt = build_augmented_prompt(question, [r.document.text for r in results])

    provider = make_provider(provider_name, api_key)
    if provider is None:
        note = (
            "### ⚠️ Retrieval ran, generation is off\n\n"
            "No API key is configured, so the demo didn't call an LLM. The panel "
            "below shows the exact context the engine retrieved and *would* have "
            "conditioned the answer on. Add an API key (Space secret, or the box "
            "in **Settings**) to see the grounded answer itself."
        )
        return note, sources_md

    try:
        text = provider.complete(SYSTEM_PROMPT, prompt)
    except Exception as exc:  # surface the real backend error to the user
        raise gr.Error(f"Generation failed: {exc}") from exc
    return text, sources_md


def rebuild_from_uploads(paths: list[str] | None):
    """Replace the active index with one built from user-uploaded files."""
    if not paths:
        raise gr.Error("Upload at least one .txt, .md, or .pdf file.")
    store, n_chunks, n_files = build_store([Path(p) for p in paths])
    status = f"✅ Indexed **{n_chunks} chunks** from **{n_files} file(s)**. Ask away below."
    return store, status


def restore_demo_corpus():
    """Reset the active index back to the bundled demo corpus."""
    status = (
        f"↩️ Restored the demo corpus — **{DEMO_CHUNKS} chunks** from "
        f"**{DEMO_FILES} file(s)** (`{CORPUS_DIR.name}/`)."
    )
    return DEMO_STORE, status


# Build the demo index once at startup.
DEMO_STORE, DEMO_CHUNKS, DEMO_FILES = build_store(sorted(CORPUS_DIR.glob("*")))

_env_keys = [
    name for name, var in (("Anthropic", "ANTHROPIC_API_KEY"), ("OpenAI", "OPENAI_API_KEY"))
    if os.environ.get(var)
]
_key_status = (
    f"🔑 Detected key(s) in the environment: **{', '.join(_env_keys)}**."
    if _env_keys
    else "🔑 No API key in the environment — paste one under **Settings** to enable answers."
)

INTRO = f"""
# 🔎 RAG Engine — live demo

A from-scratch Retrieval-Augmented Generation pipeline: **local embeddings →
FAISS similarity search → an LLM answer grounded strictly in what was retrieved.**
No answer comes from the model's parametric memory; if the corpus doesn't say
it, the engine says it doesn't know.

**Loaded corpus:** a small mock *Acme Payments API* reference
({DEMO_FILES} docs, {DEMO_CHUNKS} chunks). Ask about auth, charges, refunds, or
webhooks — or load your own documents under **Settings → Your own documents**.

Every answer ships with the retrieved chunks it was built from, so you can see
the grounding, not just trust it.

{_key_status}

<sub>Embedder `{CONFIG['embedder']['model']}` (local) · default model
`{CONFIG['llm']['model']}` · [source on GitHub](https://github.com/SuryaSaravanan4/rag-engine)</sub>
"""


with gr.Blocks(title="RAG Engine — live demo", theme=gr.themes.Soft()) as demo:
    active_store = gr.State(DEMO_STORE)

    gr.Markdown(INTRO)

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. How do I stop a retry from double-charging a customer?",
            scale=5,
            autofocus=True,
        )
        ask = gr.Button("Ask", variant="primary", scale=1)

    gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[question], label="Try one")

    answer_out = gr.Markdown(label="Answer")
    with gr.Accordion("Retrieved context — what the LLM was actually given", open=False):
        sources_out = gr.Markdown()

    with gr.Accordion("Settings", open=False):
        with gr.Row():
            provider = gr.Radio(
                choices=[ANTHROPIC, OPENAI],
                value=ANTHROPIC,
                label="LLM backend",
            )
            top_k = gr.Slider(
                minimum=1, maximum=10, value=CONFIG["retriever"]["top_k"], step=1,
                label="Chunks to retrieve (top-k)",
            )
        api_key = gr.Textbox(
            label="Your API key (optional)",
            placeholder="sk-ant-…  or  sk-…  — used only for your requests, never stored",
            type="password",
        )

        gr.Markdown("#### Your own documents")
        gr.Markdown(
            "Replace the demo corpus with your own `.txt`, `.md`, or `.pdf` files. "
            "They're embedded in memory for this session only."
        )
        uploads = gr.File(
            file_count="multiple",
            file_types=[".txt", ".md", ".pdf"],
            label="Documents",
        )
        with gr.Row():
            rebuild = gr.Button("Index these files")
            restore = gr.Button("Restore demo corpus")
        corpus_status = gr.Markdown()

    inputs = [question, active_store, provider, api_key, top_k]
    ask.click(answer, inputs=inputs, outputs=[answer_out, sources_out])
    question.submit(answer, inputs=inputs, outputs=[answer_out, sources_out])

    rebuild.click(rebuild_from_uploads, inputs=[uploads], outputs=[active_store, corpus_status])
    restore.click(restore_demo_corpus, inputs=None, outputs=[active_store, corpus_status])


if __name__ == "__main__":
    demo.queue().launch()
