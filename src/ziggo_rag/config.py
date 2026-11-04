"""Runtime settings, read from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    generate_model: str = "claude-sonnet-5-5"
    grade_model: str = "claude-haiku-5-5"

    embedding_model: str = "intfloat/multilingual-e5-small"

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "ziggo_pages"

    seed_urls: list[str] = [
        "https://www.ziggo.nl/internet",
        "https://www.ziggo.nl/televisie",
        "https://www.ziggo.nl/ziggo-go",
        "https://www.ziggo.nl/alles-in-1",
        "https://www.ziggo.nl/mobiel",
    ]


@lru_cache
def settings() -> Settings:
    """Return the process-wide settings, read once on first call."""
    return Settings()
