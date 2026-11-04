"""Types and HTTP constants shared by the page scraper and the catalogue fetch."""

from typing import TypedDict

# ziggo.nl answers a 403 to the default httpx User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
TIMEOUT_S = 10.0


class Section(TypedDict):
    heading: str
    text: str


class Page(TypedDict):
    url: str
    title: str
    sections: list[Section]


class ScrapeError(Exception):
    """Raised when a seed URL or the catalogue cannot be fetched.

    Covers a non-200 status, a timeout, no connection, and a catalogue
    response in an unexpected shape.
    """
