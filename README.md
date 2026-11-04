# Ziggo RAG assistant

A question-answering API over Ziggo product pages. Run instructions, design choices, and diagrams are added as the project is built.

## Secrets

The Anthropic API key is read from the environment. `.env.1password` holds `op://` references to the 1Password items; it contains no secrets and is committed. Run commands through the 1Password CLI so the references are resolved at process start:

```sh
op run --env-file=.env.1password -- uv run uvicorn ziggo_rag.api:app
```

Without 1Password, copy `.env.1password` to `.env`, replace the `op://` values with real ones, and load it with `uv run --env-file .env ...`. `.env` is gitignored.
