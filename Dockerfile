FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1

# Dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Download the embedding model at build time so the container needs no network
# access to Hugging Face at startup.
RUN HF_HUB_OFFLINE=0 /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY README.md ./
COPY src ./src
# Scrape snapshot, used when the live scrape fails at startup.
COPY data ./data
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "ziggo_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
