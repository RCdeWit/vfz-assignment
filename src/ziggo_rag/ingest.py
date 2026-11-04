"""Turn scraped pages into chunks, embed them locally, and upsert them into Chroma.

One chunk per heading section. The page title and heading are prepended to the
embedded text so a section that only reads "Vanaf € 45 per maand" still
carries its product context. Sections over MAX_SECTION_CHARS are split at line
boundaries, with the title and heading prefix repeated on each part.

Vectors are computed here with the e5 model from ``llm.py`` and passed to
Chroma explicitly. Chroma's own default embedding function (all-MiniLM-L6-v2,
English only) is never used, so queries and passages share one model.

Run the module directly to check the pipeline by hand. Without arguments it
chunks the committed snapshot and prints size statistics. ``--ingest`` upserts
the snapshot chunks into Chroma and prints the collection count. ``--query``
embeds a question and prints the nearest chunks.
"""

import argparse
import hashlib
import logging
import sys
from datetime import UTC, datetime
from typing import TypedDict

import chromadb
from chromadb.api.models.Collection import Collection

from ziggo_rag.config import settings
from ziggo_rag.llm import embed_passages, embed_queries
from ziggo_rag.scrape import Page, load_snapshot

log = logging.getLogger("ziggo_rag.ingest")

# Section text above this length is split into parts. Chunk text is the
# section part plus the title and heading prefix, so a chunk can be a little
# longer than this. The limit keeps chunks under the e5 model's 512-token
# input; longer input is truncated without an error, so the tail of a chunk
# would be stored but never embedded. On the committed snapshot the longest
# chunk is about 460 tokens.
MAX_SECTION_CHARS = 1500
BATCH_SIZE = 100
# Collection metadata key written after the last batch. Startup only skips
# ingest when it is present, so a run that died halfway is retried.
INGESTED_AT_KEY = "ingested_at"


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


def is_ingested(collection: Collection) -> bool:
    """Report whether a previous ingest ran to completion on this collection.

    A non-zero count is not enough: a process killed between batches leaves a
    partial index that would otherwise be served as complete.
    """
    return INGESTED_AT_KEY in (collection.metadata or {})


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
    """Embed the chunks, upsert them in batches, mark the collection complete, and return the count.

    Upsert rather than add so a re-run with the same pages does not fail on
    existing ids. Nothing is deleted: a chunk whose section was renamed or
    removed since the last run stays in the collection. Rebuild from an empty
    volume to drop stale chunks.
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
    # modify replaces the whole metadata dict, so existing keys are carried
    # over, except the hnsw: settings: Chroma rejects an update that names the
    # distance function even when the value is unchanged. The index keeps its
    # distance function regardless of the metadata.
    kept = {k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")}
    stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    collection.modify(metadata={**kept, INGESTED_AT_KEY: stamp})
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk the committed snapshot; optionally ingest it or query Chroma."
    )
    parser.add_argument("--ingest", action="store_true", help="upsert the snapshot chunks")
    parser.add_argument("--query", metavar="TEXT", help="print the nearest chunks for TEXT")
    parser.add_argument("-k", type=int, default=3, help="results per query (default 3)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # These libraries log model loading and every HTTP request at INFO, which
    # hides the few lines this command prints.
    for name in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)
    s = settings()

    if args.query is None or args.ingest:
        pages = load_snapshot(s.pages_dir)
        chunks = chunk_pages(pages)
        lengths = sorted(len(c["text"]) for c in chunks)
        print(
            f"pages={len(pages)} sections={sum(len(p['sections']) for p in pages)} "
            f"chunks={len(chunks)} unique_ids={len({c['id'] for c in chunks})} "
            f"split_parts={sum(c['metadata']['part'] > 0 for c in chunks)} "
            f"median_chars={lengths[len(lengths) // 2]} max_chars={lengths[-1]}"
        )
    if args.ingest or args.query is not None:
        client = chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)
        collection = get_collection(client)
    if args.ingest:
        print(f"count before={collection.count()}")
        ingest(chunks, collection)
        print(f"count after={collection.count()}")
    if args.query is not None:
        result = collection.query(query_embeddings=embed_queries([args.query]), n_results=args.k)
        for m, distance in zip(result["metadatas"][0], result["distances"][0], strict=True):
            print(f"{distance:.3f} {m['url']} | {m['heading']} | part {m['part']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
