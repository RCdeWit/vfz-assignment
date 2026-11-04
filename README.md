# Ziggo RAG assistant

A question-answering API over Ziggo product pages. Run instructions, design choices, and diagrams are added as the project is built.

## Secrets

The Anthropic API key is read from the environment. `.env.1password` holds `op://` references to the 1Password items; it contains no secrets and is committed. Run commands through the 1Password CLI so the references are resolved at process start:

```sh
op run --env-file=.env.1password -- docker compose up --build
```

Or write a resolved `.env`, which compose picks up on its own. `.env` is gitignored.

```sh
op inject -i .env.1password -o .env
docker compose up --build
```

Without 1Password, copy `.env.1password` to `.env` and replace the `op://` value with the key.

## Running

`docker compose up --build` starts Chroma and the API on port 8000. The first build downloads the embedding model and CPU-only torch; the API image is about 2.3 GB.

```sh
curl localhost:8000/health
```

Returns 200 when Chroma answers and the API key is set, else 503 with the status of each component.
