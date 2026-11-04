"""HTTP API: health check, startup page loading, and, in later phases, the /ask endpoint."""

import logging
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ziggo_rag.config import settings
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
    if collection_is_empty():
        pages = load_pages()
        log.info("%d pages ready for ingest", len(pages))
    yield


def collection_is_empty() -> bool:
    """Report whether the Chroma collection has no chunks yet.

    Returns False when Chroma cannot be reached, so startup completes and
    /health reports the outage instead of the process crashing.
    """
    s = settings()
    try:
        count = chroma_client().get_or_create_collection(s.chroma_collection).count()
    except Exception as e:  # chromadb raises several unrelated types on connection failure
        log.warning("chroma unreachable at startup, skipping scrape: %s", e)
        return False
    log.info("collection %s holds %d chunks", s.chroma_collection, count)
    return count == 0


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
        status["collection_count"] = client.get_or_create_collection(s.chroma_collection).count()
    except Exception as e:  # chromadb raises several unrelated types on connection failure
        log.warning("chroma health check failed: %s", e)

    healthy = status["chroma"] == "ok" and status["anthropic_api_key"] == "set"
    return JSONResponse(status, status_code=200 if healthy else 503)
