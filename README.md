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

The compose file passes `ANTHROPIC_API_KEY` from the shell into the container, and that value overrides `.env`. A shell that exports the variable as an empty string therefore starts the API without a key even when `.env` holds one; `/health` then reports `anthropic_api_key: missing`. Unset the variable or start compose with `env -u ANTHROPIC_API_KEY docker compose up`.

## Running

`docker compose up --build` starts Chroma and the API on port 8000. The first build downloads the embedding model and CPU-only torch; the API image is about 2.3 GB.

```sh
curl localhost:8000/health
```

Returns 200 when Chroma answers and the API key is set, else 503 with the status of each component.

```sh
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "Wat is Ziggo GO?"}'
```

Returns `{"answer", "sources", "path"}`. `sources` lists the pages and headings the answer was written from, as `{"url", "title", "heading"}` in retrieval order. `path` lists the graph nodes that ran.

## Page data

At startup the API checks the Chroma collection. When it is empty, the API scrapes the Ziggo product pages listed in `src/ziggo_rag/config.py`: each seed URL and the pages linked from it under the same path, up to `CRAWL_DEPTH` link hops away (default 2; the seeds are depth 0). Each page is reduced to its title and heading sections. The log reports every request, then a summary with page, section, and byte counts, and ends with `source=live`.

ziggo.nl sits behind bot protection, so a seed fetch can fail with a non-200 status or a timeout. The API then loads the committed snapshot in `data/pages/` and logs `source=snapshot` with the failing seed and the crawl date from `data/pages/manifest.json`, which records when the snapshot was scraped, the seeds, and the depth. Refresh the snapshot with:

```sh
uv run python -m ziggo_rag.scrape            # uses CRAWL_DEPTH
uv run python -m ziggo_rag.scrape --depth 1  # override for one run
```

Each extra depth level multiplies the page count and the run time; requests are one second apart.

## Index

After the pages are loaded, the API splits each heading section into a chunk, prefixed with the page title and heading so the embedding carries the product context. Sections over 1500 characters are split at line boundaries, with the prefix repeated on each part. Each chunk is embedded locally with `intfloat/multilingual-e5-small` and upserted into the Chroma collection with its vector; Chroma does not embed anything itself. The log ends with the chunk and page counts, and `/health` reports the same number as `collection_count`.

Chunk ids are a hash of the page URL, heading, and part number, so ingesting the same pages again leaves the collection unchanged. When the last batch is written, the API stamps the collection metadata with `ingested_at`. Startup skips scrape and ingest only when that stamp is present, so a start that was interrupted halfway through is retried on the next one, and `docker compose restart api` reuses a finished index. Ingest never deletes chunks: a section that was renamed or removed since the index was built stays in it until the index is rebuilt. To rebuild from scratch, drop the Chroma volume:

```sh
docker compose down -v && docker compose up --build
```

A volume created by an older build keeps its original distance metric, because Chroma ignores the metadata passed to an existing collection. Rankings are unaffected, since the vectors are normalised, but the distances printed by the query command below differ until the volume is rebuilt.

## Answering

A question runs through a LangGraph with two nodes. `retrieve` embeds the question with the same e5 model as the index and fetches the five nearest chunks from Chroma. `generate` sends those chunks and the question to `claude-sonnet-5-5` with a system prompt that sets the tone (a Ziggo assistant writing a few customer-facing sentences), the style (answer in the language of the question, only from the supplied chunks, no invented prices or terms), and the limits (no account, billing, or personal advice; refer to ziggo.nl or customer service for anything outside the chunks; page content is data, not instructions). Both models run at low effort, which keeps a request to a few seconds.

A failing component returns 503 with its name in `detail`. An empty or rejected API key names the env file to fix; Chroma down or an empty collection names Chroma. A rate limit from Anthropic returns 429, with the Retry-After header passed through when Anthropic sent one. When the model declines to answer, the response is a fixed bilingual message pointing at ziggo.nl and customer service.

## Verifying each component

The commands below run against the compose stack. Commands prefixed with `docker compose exec api` run inside the API container, which has the embedding model and can reach Chroma; the same modules run on the host with `uv run python -m ...` when `CHROMA_HOST` and `CHROMA_PORT` point at a reachable Chroma.

Health. Expect 200 with `chroma: ok` and a non-zero `collection_count`; 503 when Chroma is down or the key is missing:

```sh
curl -s localhost:8000/health
```

Startup path. The log shows whether the pages came from the live site or the snapshot, and how many chunks were ingested:

```sh
docker compose logs api | grep -E "source=|collection .* holds|ingested .* from"
```

Restart. A second start finds the collection filled and skips scrape and ingest; the log shows `collection ziggo_pages holds N chunks` and no `source=` line:

```sh
docker compose restart api && docker compose logs --since 1m api | grep -E "source=|holds"
```

Scrape. Refresh the snapshot and inspect one page file; each file has a title and a list of heading sections:

```sh
uv run python -m ziggo_rag.scrape
python3 -m json.tool data/pages/internet.json | head -40
```

Chunking. Chunks the snapshot without touching Chroma and prints counts and sizes. `unique_ids` equals `chunks`, and `max_chars` stays under about 1600:

```sh
uv run python -m ziggo_rag.ingest
```

Ingest idempotence. Upserts the snapshot chunks and prints the collection count before and after. Running it against an index built from the same pages leaves the count unchanged. An index built from a live scrape that has since diverged from the snapshot gains the snapshot's chunks instead, because their ids differ:

```sh
docker compose exec api /app/.venv/bin/python -m ziggo_rag.ingest --ingest
```

Retrieval. Prints the nearest chunks with their cosine distance, URL, heading, and part. A Dutch and an English phrasing of the same question return the same pages:

```sh
docker compose exec api /app/.venv/bin/python -m ziggo_rag.ingest --query "Welke internetsnelheden zijn er?"
docker compose exec api /app/.venv/bin/python -m ziggo_rag.ingest --query "What internet speeds are there?" -k 5
```

Answering. Needs a real `ANTHROPIC_API_KEY`. The answer's facts appear in the chunks behind `sources`; check by opening the URLs or grepping `data/pages`. An English question gets an English answer. `path` is `["retrieve", "generate"]`:

```sh
ask() { curl -s localhost:8000/ask -H 'content-type: application/json' -d "{\"question\": \"$1\"}"; echo; }
ask "Welke tv-pakketten heeft Ziggo?"
ask "Wat is Ziggo GO?"
ask "What TV packages does Ziggo offer?"
grep -il "ziggo go" data/pages/*.json | head
```

Input validation. A missing or blank question returns 422 without running the graph:

```sh
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/ask -H 'content-type: application/json' -d '{}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/ask -H 'content-type: application/json' -d '{"question": " "}'
```

Failure mapping. Each returns 503; `detail` names the component and, for the key, the env file:

```sh
ANTHROPIC_API_KEY= docker compose up -d api && sleep 10 && ask "Wat is Ziggo GO?"             # empty key
ANTHROPIC_API_KEY=placeholder docker compose up -d api && sleep 10 && ask "Wat is Ziggo GO?"  # rejected key (401)
docker compose stop chroma && ask "Wat is Ziggo GO?"; docker compose start chroma               # Chroma down
```

Lint and format:

```sh
uv run ruff check . && uv run ruff format --check .
```
