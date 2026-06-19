# Design Notes

## Why RAG instead of fine-tuning?

Fine-tuning bakes knowledge into model weights — it's expensive, slow to update, and can't reference documents added after training. RAG keeps the knowledge external and live: swap the corpus, re-ingest, and the model immediately answers from new information. For a dynamic corpus like an internal API catalog (my motivation for this project), RAG is the right call.

## Chunking strategy

Fixed-size character windows with overlap (default 512 chars, 64 overlap). Tradeoffs:

- **Too small:** Context gets fragmented; a single concept might span multiple chunks.
- **Too large:** Retrieval becomes coarse; the model gets a lot of irrelevant context alongside relevant content.
- **Overlap:** Prevents losing context at chunk boundaries. A sentence split across chunk N and N+1 still appears intact in at least one chunk.

Sentence-boundary chunking (split on `.`) is a natural next step.

## Embedding model choice

`all-MiniLM-L6-v2` is the default local model: 384 dimensions, ~80ms/batch on CPU, good performance on semantic similarity benchmarks, and runs offline. OpenAI `text-embedding-3-small` is better for retrieval quality but costs money and requires a network call per ingest.

## FAISS index type

`IndexFlatL2` for now — exact L2 search, no approximation error. Scales to ~1M vectors on a laptop without issue. For larger corpora, `IndexIVFFlat` (inverted file index) reduces search time at the cost of a training step and small recall loss.

## LLM provider abstraction

The `BaseModelProvider` interface has one method: `complete(system, user) -> str`. Every backend implements this signature. The pipeline never imports a specific provider — it calls `get_provider(config["llm"]["provider"])` and gets back a `BaseModelProvider`. Adding a new backend (e.g. Gemini) means writing one file and adding one branch to the factory. No changes to pipeline code.

This is the same pattern I used at CoverMyMeds for the internal API Discovery Platform, where five LLM backends were runtime-swappable via a single config change.
