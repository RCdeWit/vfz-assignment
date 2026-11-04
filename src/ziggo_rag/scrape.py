"""Bounded crawl of ziggo.nl product pages, cleaned into heading sections.

The crawl fetches each seed URL and follows links whose path sits under one
of the seed paths, up to a maximum depth (the seeds are depth 0). Each fetched
page becomes one record: ``{"url", "title", "sections": [{"heading", "text"}]}``.
Run the module directly to write those records to ``data/pages/`` as the
committed snapshot, with a ``manifest.json`` recording the crawl time and
parameters; ``--depth`` overrides the configured depth. The snapshot also
holds the catalogue page from ``catalogue.py``, which carries the
subscription prices the product pages do not render.

ziggo.nl renders most product detail (package names, speeds, prices) through
client-side templates. The server response carries that text inside
``<template>`` elements, which BeautifulSoup excludes from ``get_text`` by
default, so the cleaner reads template strings as ordinary text. Text nodes
that still contain an unrendered placeholder such as ``{{ vmpc.getPrice(2.5) }}``
are dropped.
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, TemplateString

from ziggo_rag.catalogue import fetch_catalogue
from ziggo_rag.config import settings
from ziggo_rag.pages import TIMEOUT_S, USER_AGENT, Page, ScrapeError, Section

log = logging.getLogger("ziggo_rag.scrape")

DELAY_S = 1.0
# Shorter sections are icon captions and button labels ("Meer info").
MIN_SECTION_CHARS = 40
MANIFEST_NAME = "manifest.json"

DROP_TAGS = ["nav", "footer", "header", "script", "style", "noscript"]
DROP_ATTR = re.compile(r"cookie|consent|banner|modal", re.I)
HEADING_TAGS = ("h1", "h2", "h3")
TEXT_TAGS = ("p", "li", "td", "th")
# Exact types, so Comment, Script, and Stylesheet strings stay excluded.
TEXT_TYPES = (NavigableString, TemplateString)


class _FetchFailed(Exception):
    """Raised by _fetch for any URL that did not yield an HTML 200 response."""


@dataclass
class _Stats:
    """Counters logged as the end-of-run summary."""

    started: float = field(default_factory=time.perf_counter)
    requests: int = 0
    bytes: int = 0
    failed: int = 0
    redirected: int = 0
    duplicates: int = 0
    pages_by_depth: dict[int, int] = field(default_factory=dict)
    sections: int = 0
    chars: int = 0

    @property
    def pages(self) -> int:
        return sum(self.pages_by_depth.values())

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def scrape_pages(seeds: list[str], max_depth: int = 1) -> list[Page]:
    """Crawl the seeds breadth-first to max_depth; return one cleaned Page per unique URL.

    The seeds enter a FIFO queue at depth 0. Each queued URL is fetched once:
    a set of normalised URLs (no query, no fragment, no trailing slash) stops
    a link from being queued twice, and a page reached through a redirect is
    keyed by its final URL, so two routes to one page yield one record. Links
    are collected only from pages below max_depth, and only when their path
    equals a seed path or sits below it, so the crawl stays inside the
    product sections. A seed at the site root is fetched for its content but
    adds no prefix, because "/" would admit the whole site.

    Raises ScrapeError when any seed _fetch fails. Deeper pages that fail are
    logged and skipped. Logs every request and a summary with counts per
    depth at the end.
    """
    stats = _Stats()
    prefixes = [p for p in (urlsplit(_normalize_url(s)).path for s in seeds) if p != "/"]
    queue: list[tuple[str, int]] = [(_normalize_url(s), 0) for s in seeds]
    seen: set[str] = {url for url, _ in queue}
    pages: dict[str, Page] = {}

    log.info(
        "starting crawl: %d seeds, max depth %d, path prefixes %s", len(seeds), max_depth, prefixes
    )
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "nl-NL,nl;q=0.9"}
    with httpx.Client(headers=headers, timeout=TIMEOUT_S, follow_redirects=True) as client:
        # The queue grows while iterating; index access reads appended entries.
        i = 0
        while i < len(queue):
            url, depth = queue[i]
            i += 1
            log.info("[%d/%d] depth %d %s", i, len(queue), depth, url)
            try:
                final_url, html = _fetch(client, url, stats)
            except _FetchFailed as e:
                if depth == 0:
                    raise ScrapeError(f"seed {url}: {e}") from e
                stats.failed += 1
                log.warning("skipping %s: %s", url, e)
                continue

            final_url = _normalize_url(final_url)
            if final_url != url:
                stats.redirected += 1
                seen.add(final_url)
                log.info("redirected to %s", final_url)
                if not _in_scope(final_url, prefixes):
                    log.info("redirect target is outside the seed paths, skipping")
                    continue
            if final_url in pages:
                stats.duplicates += 1
                log.info("already scraped as %s, skipping", final_url)
                continue

            soup = BeautifulSoup(html, "html.parser")
            if depth < max_depth:
                links = _in_scope_links(soup, final_url, prefixes)
                new = [link for link in links if link not in seen]
                seen.update(new)
                queue.extend((link, depth + 1) for link in new)
                log.info("found %d in-scope links, %d new queued", len(links), len(new))

            page = _clean_page(final_url, soup)
            pages[final_url] = page
            stats.pages_by_depth[depth] = stats.pages_by_depth.get(depth, 0) + 1
            page_chars = sum(len(s["text"]) for s in page["sections"])
            stats.sections += len(page["sections"])
            stats.chars += page_chars
            log.info(
                "cleaned %r: %d sections, %d chars",
                page["title"],
                len(page["sections"]),
                page_chars,
            )

    _log_summary(stats)
    return list(pages.values())


def _fetch(client: httpx.Client, url: str, stats: _Stats) -> tuple[str, str]:
    """GET url and return (final URL after redirects, HTML body).

    Raises _FetchFailed on a transport error, timeout, non-200 status, or a
    non-HTML content type. Sleeps DELAY_S before every request after the first.
    """
    if stats.requests:
        time.sleep(DELAY_S)
    stats.requests += 1
    t0 = time.perf_counter()
    try:
        resp = client.get(url)
    except httpx.HTTPError as e:
        raise _FetchFailed(f"{type(e).__name__}: {e}") from e
    elapsed = time.perf_counter() - t0
    stats.bytes += len(resp.content)
    content_type = resp.headers.get("content-type", "")
    log.info(
        "GET %s -> %d %s, %d bytes, %.2fs",
        url,
        resp.status_code,
        content_type.split(";")[0] or "no content-type",
        len(resp.content),
        elapsed,
    )
    if resp.status_code != 200:
        raise _FetchFailed(f"status {resp.status_code}")
    if "text/html" not in content_type:
        raise _FetchFailed(f"content-type {content_type!r} is not HTML")
    return str(resp.url), resp.text


def _normalize_url(url: str) -> str:
    """Drop query and fragment, lowercase scheme and host, strip a trailing slash."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def _in_scope(url: str, prefixes: list[str]) -> bool:
    """Report whether the URL path equals a seed path or sits below it.

    Matches on path segments, so the prefix ``/internet`` does not admit
    ``/internet-en-mobiel``.
    """
    path = urlsplit(url).path
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def _in_scope_links(soup: BeautifulSoup, base_url: str, prefixes: list[str]) -> list[str]:
    """Return the unique same-host links in the document whose path is in scope."""
    host = urlsplit(base_url).netloc.lower()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        url = _normalize_url(urljoin(base_url, a["href"]))
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.netloc != host:
            continue
        if _in_scope(url, prefixes) and url not in links:
            links.append(url)
    return links


def _clean_page(url: str, soup: BeautifulSoup) -> Page:
    """Reduce a parsed document to its title and heading sections.

    Takes ``<main>`` when present, else ``<body>``. Removes navigation, footer,
    header, scripts, styles, and any element whose id or class matches
    cookie, consent, banner, or modal. Walks the rest in document order: each
    h1/h2/h3 starts a section; paragraph, list item, and table cell text is
    appended to the current one. Repeated lines within a page are kept once,
    because the templates repeat blocks for different viewport sizes. Sections
    under MIN_SECTION_CHARS are dropped.
    """
    title = _title(soup)
    root = soup.find("main") or soup.body or soup
    for el in root.find_all(DROP_TAGS):
        el.decompose()
    for el in root.find_all(attrs={"class": DROP_ATTR}) + root.find_all(id=DROP_ATTR):
        if not el.decomposed:
            el.decompose()

    sections: list[Section] = []
    heading = ""
    lines: list[str] = []
    seen_lines: set[str] = set()

    def flush() -> None:
        text = "\n".join(lines)
        if len(text) >= MIN_SECTION_CHARS:
            sections.append({"heading": heading, "text": text})

    for el in root.find_all([*HEADING_TAGS, *TEXT_TAGS]):
        # A p inside an li, or a heading inside a td, is already part of the
        # ancestor's text.
        if el.find_parent(TEXT_TAGS) is not None:
            continue
        text = _element_text(el)
        if not text:
            continue
        if el.name in HEADING_TAGS:
            flush()
            heading = text
            lines = []
        elif text not in seen_lines:
            seen_lines.add(text)
            lines.append(text)
    flush()

    if not title:
        title = sections[0]["heading"] if sections else urlsplit(url).path
    return {"url": url, "title": title, "sections": sections}


def _element_text(el: Tag) -> str:
    """Join the element's text nodes, including template content, with single spaces.

    Text nodes containing ``{{`` are unrendered placeholders and are dropped.
    """
    parts = []
    for s in el.find_all(string=True):
        if type(s) not in TEXT_TYPES or "{{" in s:
            continue
        cleaned = re.sub(r"\s+", " ", s).strip()
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def _title(soup: BeautifulSoup) -> str:
    if soup.title is None:
        return ""
    title = re.sub(r"\s+", " ", soup.title.get_text()).strip()
    return re.sub(r"\s*\|\s*Ziggo$", "", title)


def _log_summary(stats: _Stats) -> None:
    """Log the crawl counters as one block."""
    by_depth = ", ".join(f"depth {d}: {n}" for d, n in sorted(stats.pages_by_depth.items()))
    log.info("crawl finished in %.1fs", stats.elapsed)
    log.info("  requests: %d (%.1f KB downloaded)", stats.requests, stats.bytes / 1024)
    log.info(
        "  pages: %d (%s); redirected: %d, duplicates skipped: %d, failed: %d",
        stats.pages,
        by_depth,
        stats.redirected,
        stats.duplicates,
        stats.failed,
    )
    log.info(
        "  sections: %d (%.1f per page), text: %d chars",
        stats.sections,
        stats.sections / stats.pages if stats.pages else 0.0,
        stats.chars,
    )


def write_snapshot(pages: list[Page], directory: Path, manifest: dict[str, object]) -> list[Path]:
    """Write one JSON file per page plus manifest.json, replacing earlier files.

    The manifest holds the crawl time and parameters in one file so a refresh
    changes only the pages whose content changed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    written = []
    for page in sorted(pages, key=lambda p: p["url"]):
        path = directory / f"{_snapshot_name(page['url'])}.json"
        _write_json(path, page)
        written.append(path)
    _write_json(directory / MANIFEST_NAME, {**manifest, "pages": len(pages)})
    return written


def load_snapshot(directory: Path) -> list[Page]:
    """Read the page files written by write_snapshot. Raises FileNotFoundError when none exist."""
    files = sorted(f for f in directory.glob("*.json") if f.name != MANIFEST_NAME)
    if not files:
        raise FileNotFoundError(f"no page files in {directory}")
    return [json.loads(f.read_text()) for f in files]


def read_manifest(directory: Path) -> dict[str, object]:
    """Return the snapshot manifest, or an empty dict when the snapshot has none."""
    path = directory / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _snapshot_name(url: str) -> str:
    """Map a URL path to a file stem: ``/televisie/ziggo-go`` becomes ``televisie__ziggo-go``."""
    path = urlsplit(url).path.strip("/") or "index"
    return re.sub(r"[^a-z0-9_-]", "-", path.replace("/", "__").lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape the seed URLs and write data/pages/.")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="link-following depth, seeds are depth 0 (default: CRAWL_DEPTH setting)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request at INFO; _fetch already logs each one with status and size.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    s = settings()
    depth = s.crawl_depth if args.depth is None else args.depth
    manifest = {
        "scraped_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "seeds": s.seed_urls,
        "max_depth": depth,
    }
    manifest["catalogue_url"] = s.catalogue_url
    try:
        pages = scrape_pages(s.seed_urls, max_depth=depth)
        pages.append(fetch_catalogue())
    except ScrapeError as e:
        log.error("scrape failed: %s", e)
        return 1
    written = write_snapshot(pages, s.pages_dir, manifest)
    size = sum(p.stat().st_size for p in written)
    log.info("wrote %d page files (%.1f KB) to %s", len(written), size / 1024, s.pages_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
