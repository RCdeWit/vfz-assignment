"""Runtime settings, read from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    generate_model: str = "claude-sonnet-5-5"
    grade_model: str = "claude-haiku-5-5"

    embedding_model: str = "intfloat/multilingual-e5-small"
    # Seconds a request to /ask may take in total. Each model call gets the
    # time left as its own timeout, so a hung Anthropic connection returns a
    # 504 at this limit instead of holding the request for several calls.
    request_timeout: float = 30.0

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "ziggo_pages"

    # The homepage is scraped for its content but, unlike the other seeds, does
    # not widen the crawl: links are only followed under the other seed paths.
    seed_urls: list[str] = [
        "https://www.ziggo.nl/",
        "https://www.ziggo.nl/internet",
        "https://www.ziggo.nl/televisie",
        "https://www.ziggo.nl/televisie/ziggo-go",
        "https://www.ziggo.nl/alles-in-1",
        "https://www.ziggo.nl/mobiel",
    ]
    # How many link hops to follow from the seeds. Depth 1 yields about 40
    # pages; depth 2 reaches the per-package and per-feature pages below them.
    crawl_depth: int = 2
    # JSON catalogue behind the configurator at /tv-internet; the only source
    # of internet subscription prices. Fetched after the crawl.
    catalogue_url: str = "https://www.ziggo.nl/services/blueriq/b2c/info"
    # Committed scrape snapshot, loaded when a live scrape fails.
    pages_dir: Path = Path("data/pages")


@lru_cache
def settings() -> Settings:
    """Return the process-wide settings, read once on first call."""
    return Settings()
