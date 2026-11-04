"""HTTP API: health check, startup scrape and ingest, and, in later phases, the /ask endpoint."""

import logging
from contextlib import asynccontextmanager

import chromadb
from chromadb.api.models.Collection import Collection
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ziggo_rag.config import settings
from ziggo_rag.ingest import INGESTED_AT_KEY, chunk_pages, get_collection, ingest, is_ingested
from ziggo_rag.llm import embedding_model
from ziggo_rag.scrape import Page, ScrapeError, load_snapshot, read_manifest, scrape_pages

log = logging.getLogger("ziggo_rag")


def chroma_client() -> chromadb.ClientAPI:
    """Connect to the Chroma service named in the settings."""
    s = settings()
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs every Chroma request at INFO, which drowns out the startup log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Load the embedding model before serving so the first request does not
    # pay the multi-second load.
    embedding_model()
    log.info("embedding model loaded")
    collection = startup_collection()
    if collection is not None and not is_ingested(collection):
        try:
            pages = load_pages()
            count = ingest(chunk_pages(pages), collection)
            log.info("ingested %d chunks from %d pages", count, len(pages))
        except Exception as e:
            # Logged, not raised: the server starts without a usable index so
            # /health can show the collection count. The ingested_at stamp was
            # not written, so the next start runs scrape and ingest again.
            log.error("startup ingest failed, index incomplete: %s", e)
    yield


def startup_collection() -> Collection | None:
    """Return the Chroma collection, or None when Chroma cannot be reached.

    Returning None lets startup complete so /health reports the outage
    instead of the process crashing. The log line states the chunk count and
    whether a previous ingest finished.
    """
    s = settings()
    try:
        collection = get_collection(chroma_client())
        count = collection.count()
    except Exception as e:  # chromadb raises several unrelated types on connection failure
        log.warning("chroma unreachable at startup, skipping scrape and ingest: %s", e)
        return None
    ingested_at = (collection.metadata or {}).get(INGESTED_AT_KEY, "never")
    log.info(
        "collection %s holds %d chunks, ingested_at=%s", s.chroma_collection, count, ingested_at
    )
    return collection


def load_pages() -> list[Page]:
    """Scrape the seed URLs, or load the committed snapshot when a seed fetch fails.

    Logs ``source=live`` or ``source=snapshot`` so a run can be checked for
    which path it took; the snapshot line includes the crawl date from its
    manifest.
    """
    s = settings()
    try:
        pages = scrape_pages(s.seed_urls, max_depth=s.crawl_depth)
        log.info("pages loaded source=live count=%d", len(pages))
    except ScrapeError as e:
        log.warning("live scrape failed: %s", e)
        pages = load_snapshot(s.pages_dir)
        scraped_at = read_manifest(s.pages_dir).get("scraped_at", "unknown")
        log.info("pages loaded source=snapshot count=%d scraped_at=%s", len(pages), scraped_at)
    return pages


app = FastAPI(title="Ziggo RAG assistant", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    """Report Chroma reachability, API key presence, and collection size.

    Returns 200 when Chroma answers and the key is set, else 503 with the
    status of each component so the caller can see which one is missing.
    """
    s = settings()
    status: dict[str, object] = {
        "chroma": "unreachable",
        "anthropic_api_key": "set" if s.anthropic_api_key else "missing",
        "collection_count": None,
    }
    try:
        client = chroma_client()
        client.heartbeat()
        status["chroma"] = "ok"
        status["collection_count"] = get_collection(client).count()
    except Exception as e:  # chromadb raises several unrelated types on connection failure
        log.warning("chroma health check failed: %s", e)

    healthy = status["chroma"] == "ok" and status["anthropic_api_key"] == "set"
    return JSONResponse(status, status_code=200 if healthy else 503)
