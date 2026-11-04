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

## Page data

At startup the API checks the Chroma collection. When it is empty, the API scrapes the Ziggo product pages listed in `src/ziggo_rag/config.py`: each seed URL and the pages linked from it under the same path, up to `CRAWL_DEPTH` link hops away (default 2; the seeds are depth 0). Each page is reduced to its title and heading sections. The log reports every request, then a summary with page, section, and byte counts, and ends with `source=live`.

ziggo.nl sits behind bot protection, so a seed fetch can fail with a non-200 status or a timeout. The API then loads the committed snapshot in `data/pages/` and logs `source=snapshot` with the failing seed. Refresh the snapshot with:

```sh
uv run python -m ziggo_rag.scrape            # uses CRAWL_DEPTH
uv run python -m ziggo_rag.scrape --depth 1  # override for one run
```

Each extra depth level multiplies the page count and the run time; requests are one second apart.
