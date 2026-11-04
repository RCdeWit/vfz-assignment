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
    # encode shows a tqdm progress bar by default. A container log has no
    # terminal, so each bar update lands as a separate line of partial output.
    vectors = embedding_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


@lru_cache
def generate_model() -> ChatAnthropic:
    """Return the chat model that writes customer-facing answers, built once per process."""
    return _chat(settings().generate_model)


@lru_cache
def grade_model() -> ChatAnthropic:
    """Return the chat model for grading and query rewriting, built once per process."""
    return _chat(settings().grade_model)


def _chat(model: str) -> ChatAnthropic:
    # One attempt plus one retry within about a minute per call. A request
    # makes up to four calls in sequence, so a hung connection holds a request
    # thread for at most about four minutes. The SDK default is a ten minute
    # timeout with two retries, which would make that two hours.
    # Claude 5.5 models reject a non-default temperature with a 400 and always
    # run adaptive thinking, so output variation and depth are controlled with
    # effort instead. Low effort suits these calls: the answer is a few
    # sentences composed from supplied context, and grading is a yes or no.
    return ChatAnthropic(
        model=model,
        max_tokens=1024,
        effort="low",
        default_request_timeout=30.0,
        max_retries=1,
        api_key=settings().anthropic_api_key or None,
    )
