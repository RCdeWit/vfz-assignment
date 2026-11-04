"""Model access: Anthropic chat model factories and the local embedding model.

Embeddings are computed locally with multilingual-e5-small rather than by a
hosted API. Anthropic has no embedding endpoint, and a local model keeps the
reviewer to one API key. The model is multilingual because the source pages are
Dutch and questions may be English.
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer

from ziggo_rag.config import settings


@lru_cache
def embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model once per process."""
    return SentenceTransformer(settings().embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for storage in the vector store."""
    # e5 models were trained with these prefixes; omitting them lowers
    # retrieval quality because queries and passages land in different regions.
    return _embed([f"passage: {t}" for t in texts])


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Embed user queries for similarity search against stored passages."""
    return _embed([f"query: {t}" for t in texts])


def _embed(texts: list[str]) -> list[list[float]]:
    # The tqdm progress bar writes carriage-return lines into the container log.
    vectors = embedding_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def generate_model() -> ChatAnthropic:
    """Return the chat model that writes customer-facing answers."""
    return _chat(settings().generate_model, temperature=0.2)


def grade_model() -> ChatAnthropic:
    """Return the chat model for retrieval grading and query rewriting."""
    return _chat(settings().grade_model, temperature=0.0)


def _chat(model: str, temperature: float) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=1024,
        api_key=settings().anthropic_api_key or None,
    )
