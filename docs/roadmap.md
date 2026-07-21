# Roadmap

Work that is deliberately not done yet, in the order it is worth doing. Each
item names the tradeoff being accepted today and what would change if it were
addressed — these are scope cuts, not oversights.

---

## 1. Retrieval evaluation harness

**The gap:** the test suite covers plumbing — that chunking splits, that FAISS
round-trips, that each backend builds the right request. Nothing measures
whether retrieval actually surfaces the right chunk. For a project whose entire
value proposition is retrieval quality, that is the load-bearing missing piece.

Without it, several design decisions can't be evaluated, only asserted:

- Is 512/64 fixed-window chunking better or worse than sentence-aware chunking
  for a given corpus? (see item 4)
- How much does switching `embedder.provider` between `local` and `openai`
  actually change which chunks rank highest?
- Does the L2-vs-cosine mismatch on unnormalized local embeddings degrade
  ranking in practice, or is it theoretical?

**Shape:** a `evals/` directory with 20–30 question/expected-source pairs over a
committed sample corpus, recall@k and MRR, and a `rag eval` subcommand. Small
enough to run in CI, specific enough to turn every question above into a number.

## 2. Retry and backoff on network calls

**Today:** no retry anywhere — LLM completions, OpenAI embeddings, and the
Ollama HTTP call each fail the whole request on a single transient 429 or 5xx.
Fine when a human is watching a CLI and can re-run; the first real problem when
anything runs unattended or ingests a large corpus.

**Shape:** exponential backoff with jitter on the three network paths, retrying
only on rate limits, 5xx, and connection errors. Bounded attempts so a genuine
outage still fails fast rather than hanging.

## 3. Incremental ingestion

**Today:** every `rag ingest` re-reads and re-embeds the entire corpus, even
unchanged files, and rebuilds the index from scratch. There is no upsert or
dedup story. Fine at demo scale; the first cost anyone hits scaling up, and it
makes the OpenAI embedder needlessly expensive to re-run.

**Shape:** hash each source file, skip unchanged ones, and support removing a
source's chunks from the index without a full rebuild.

## 4. Sentence-aware chunking

**Today:** fixed-size character windows (512 chars, 64 overlap). Simple and
fast, but chunks can split mid-sentence or mid-concept; the overlap mitigates
boundary loss without eliminating it.

Worth doing **after** item 1, not before — the point of sentence-aware chunking
is better retrieval, and without an eval harness there is no way to confirm it
delivers any.

## 5. Metadata filtering that scales

**Today:** `--source` filtering over-searches the full index (`k = ntotal`) and
discards non-matching hits, because FAISS's flat index has no native filter.
Correct, but O(n) per query — which quietly undercuts the "scales to ~1M
vectors" property the moment filtering is used.

**Shape:** a per-source id map to search within, or a move to a metadata-aware
store (Chroma, Qdrant) if filtering becomes a primary access pattern.

---

## Smaller cleanups

- **`ingest()` prints directly to stdout**, mixing library and CLI concerns.
  Low risk while this is only ever run from a terminal; real cleanup if it is
  ever imported into something that isn't. Move to a logger or an injected
  progress callback.

- **The Anthropic sampling-parameter model list needs maintenance.** Newer
  Claude models removed `temperature`/`top_p`/`top_k`, and the backend gates on
  a hardcoded prefix list — so a model newer than the list will reject
  `temperature` before it is listed there. Querying the Models API for
  capabilities at load time would remove the staleness, at the cost of a
  network call on startup.

- **Broaden the ruff rule set.** Config is currently line-length only. Import
  sorting, pyupgrade, and bugbear are all reasonable additions — added one at a
  time, each verified against the real codebase rather than enabled wholesale.

- **CI runs Ubuntu only** (Python 3.11/3.12). Nothing in the codebase is
  OS-sensitive beyond path handling, which already goes through `pathlib`, so
  this is low value until something platform-specific lands.
