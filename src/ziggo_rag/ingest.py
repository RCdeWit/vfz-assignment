"""Turn scraped pages into chunks, embed them locally, and upsert them into Chroma.

One chunk per heading section. The page title and heading are prepended to the
embedded text so a section that only reads "Vanaf € 45 per maand" still
carries its product context. Sections over MAX_SECTION_CHARS are split at line
boundaries, with the title and heading prefix repeated on each part.

Vectors are computed here with the e5 model from ``llm.py`` and passed to
Chroma explicitly. Chroma's own default embedding function (all-MiniLM-L6-v2,
English only) is never used, so queries and passages share one model.
"""

import hashlib
import logging
from typing import TypedDict

import chromadb
from chromadb.api.models.Collection import Collection

from ziggo_rag.config import settings
from ziggo_rag.llm import embed_passages
from ziggo_rag.scrape import Page

log = logging.getLogger("ziggo_rag.ingest")

# Section text above this length is split into parts. Chunk text is the
# section part plus the title and heading prefix, so a chunk can be a little
# longer than this.
MAX_SECTION_CHARS = 1500
BATCH_SIZE = 100


class ChunkMetadata(TypedDict):
    url: str
    title: str
    heading: str
    part: int


class Chunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


def get_collection(client: chromadb.ClientAPI) -> Collection:
    """Return the configured collection, creating it with cosine distance if missing.

    The e5 vectors are L2-normalised, so cosine and the Chroma default (squared
    L2) rank identically; cosine makes the returned distances readable as
    ``1 - similarity``.
    """
    return client.get_or_create_collection(
        settings().chroma_collection, metadata={"hnsw:space": "cosine"}
    )


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    """Split every section of every page into chunks with stable ids and metadata.

    The id is the sha1 of the url, heading, the heading's occurrence index on
    the page, and the part number. The same page content always yields the
    same ids, so upserting twice leaves the collection unchanged. The
    occurrence index is needed because a page can repeat a heading (two
    "Veelgestelde vragen" blocks); without it the second section would
    overwrite the first.
    """
    chunks: list[Chunk] = []
    for page in pages:
        heading_counts: dict[str, int] = {}
        for section in page["sections"]:
            heading = section["heading"]
            occurrence = heading_counts.get(heading, 0)
            heading_counts[heading] = occurrence + 1
            prefix = f"{page['title']}\n{heading}\n\n"
            for part, text in enumerate(_split(section["text"], MAX_SECTION_CHARS)):
                chunks.append(
                    {
                        "id": _chunk_id(page["url"], heading, occurrence, part),
                        "text": prefix + text,
                        "metadata": {
                            "url": page["url"],
                            "title": page["title"],
                            "heading": heading,
                            "part": part,
                        },
                    }
                )
    return chunks


def _split(text: str, limit: int) -> list[str]:
    """Split text into parts of at most ``limit`` characters at the last newline before the limit.

    The scraper joins paragraphs, list items, and table cells with single
    newlines, so a newline is a paragraph boundary. A single line longer than
    the limit is cut at the limit.
    """
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def _chunk_id(url: str, heading: str, occurrence: int, part: int) -> str:
    key = "\x1f".join([url, heading, str(occurrence), str(part)])
    return hashlib.sha1(key.encode()).hexdigest()


def ingest(chunks: list[Chunk], collection: Collection) -> int:
    """Embed the chunks and upsert them into the collection in batches; return the count.

    Upsert rather than add so a re-run with the same pages does not fail on
    existing ids.
    """
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embed_passages([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        log.info("ingested %d/%d chunks", min(start + BATCH_SIZE, len(chunks)), len(chunks))
    return len(chunks)
