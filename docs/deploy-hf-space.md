# Deploying the live demo to Hugging Face Spaces

`app.py` is a Gradio front-end for the engine. Hugging Face Spaces hosts Gradio
apps for free on CPU, which is all this demo needs — embeddings run locally and
only the LLM call goes out to an API.

The whole repo *is* the Space: HF reads the metadata block at the top of
`README.md`, installs `requirements.txt`, and runs `app.py`.

---

## What HF reads

- **`README.md` front-matter** — the YAML block at the very top declares
  `sdk: gradio` and `app_file: app.py`. Keep it there.
- **`requirements.txt`** — includes `gradio` plus the engine's runtime deps.
- **`app.py`** — builds the FAISS index over `demo/corpus/` at startup and
  serves the UI.

## One decision: who pays for generation?

Retrieval (local embeddings + FAISS) is free and needs no key. Only the final
LLM call costs money. Pick one:

- **You pay (simplest, works instantly).** Set `ANTHROPIC_API_KEY` as a Space
  secret. Every visitor's question calls your key. Fine for a low-traffic
  portfolio demo; watch usage if it gets shared widely.
- **Visitors bring their own key.** Set *no* secret. The app still loads and
  runs retrieval; the answer panel tells visitors to paste their own key under
  **Settings**. Zero cost to you, a little friction for them.
- **Both.** Set the secret *and* leave the key box — visitors can override yours
  with their own. This is what the app supports out of the box.

---

## Option A — deploy from the web UI (no local setup)

1. Go to <https://huggingface.co/new-space>.
2. Name it (e.g. `rag-engine`), **SDK: Gradio**, hardware **CPU basic** (free),
   visibility **Public**.
3. On the new Space, open the **Files** tab → **Add file** → upload the repo
   contents (or drag the folder). Include `app.py`, `requirements.txt`,
   `config.yaml`, `README.md`, `src/`, and `demo/`. You can skip `tests/`,
   `.venv/`, and the `.*_cache/` folders.
4. **Settings → Variables and secrets → New secret**: name `ANTHROPIC_API_KEY`,
   value your key (only if you chose "you pay" or "both" above).
5. The Space builds automatically. First build takes a few minutes (it installs
   torch for local embeddings). When it says **Running**, the demo is live at
   `https://huggingface.co/spaces/<you>/rag-engine`.

## Option B — deploy with git (keeps GitHub and the Space in sync)

```bash
# 1. Create the Space first (Option A steps 1–2), then add it as a remote.
#    Auth: use your HF username and an access token (Settings → Access Tokens)
#    as the password, or `huggingface-cli login`.
git remote add space https://huggingface.co/spaces/<you>/rag-engine

# 2. Push. HF builds on push.
git push space main

# 3. Set the ANTHROPIC_API_KEY secret in the Space UI (Option A step 4).
```

To update the live demo later, push again: `git push space main`.

---

## After it's live

Add the link to your resume / GitHub README:

> **RAG Engine** — Python, FAISS, sentence-transformers, Gradio · [live demo](https://huggingface.co/spaces/<you>/rag-engine)

## Troubleshooting

- **Build fails installing dependencies** — the free CPU tier has limited RAM
  during build; retry from **Settings → Factory reboot**. torch + faiss are the
  heavy installs.
- **"No API key" note on every answer** — the `ANTHROPIC_API_KEY` secret isn't
  set (or is misspelled). Retrieval is working; only generation is gated.
- **Slow first response** — the sentence-transformers model downloads on the
  first startup, then is cached for the life of the Space.
- **Model rejects `temperature`** — already handled: the Anthropic backend drops
  sampling params for models that removed them (Sonnet 5+, Opus 4.7+).
