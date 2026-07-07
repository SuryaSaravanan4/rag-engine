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

**Caveat — metadata filtering breaks that scaling claim.** `VectorStore.search(source=...)` has no native filtered index to lean on, so it ranks the *entire* index and then discards non-matching results (`k = ntotal` instead of `top_k`). That's O(n) per filtered query regardless of corpus size — fine at the current scale, but the first thing to revisit if `--source` becomes a hot path on a large corpus (e.g. a per-source id map, or a metadata-aware store like Chroma/Qdrant).

**Caveat — L2 distance and embedding normalization.** `sentence-transformers`' default `.encode()` output isn't unit-normalized, while OpenAI's embeddings are. L2 ranking is equivalent to cosine ranking only for normalized vectors, so switching the `embedder.provider` config can subtly shift which chunks rank highest for the same query — it's not a bug, just a semantic difference to be aware of when comparing local vs. OpenAI embedding quality.

## Ingestion is full-rebuild, not incremental

`ingest()` re-reads and re-embeds every file under `data_dir` on every run — there's no upsert or change-detection story. Fine for a demo-sized corpus; the first real cost anyone hits scaling this up. The fix is straightforward (hash each file, skip re-embedding unchanged ones) but deliberately deferred until it's actually needed.

## No retry/backoff on network calls

None of the three LLM backends or two embedders retry on a transient 429/5xx — a single rate-limit response fails the whole request. Acceptable for a v0.1 CLI tool; the natural hardening step before running this unattended (e.g. behind a cron job or server).

## LLM provider abstraction

The `BaseModelProvider` interface has one method: `complete(system, user) -> str`. Every backend implements this signature. The pipeline never imports a specific provider — it calls `get_provider(config["llm"]["provider"])` and gets back a `BaseModelProvider`. Adding a new backend (e.g. Gemini) means writing one file and adding one branch to the factory. No changes to pipeline code.

This is the same pattern I used at CoverMyMeds for the internal API Discovery Platform, where five LLM backends were runtime-swappable via a single config change.
