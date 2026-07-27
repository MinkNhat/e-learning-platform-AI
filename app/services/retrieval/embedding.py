import time
from functools import cache

import logfire

from app.gateway import get_embedding_client

BATCH_SIZE = 50
MAX_ATTEMPTS = 4

_embedding_dim: int | None = None


@cache
def _client():
    """Create one lazy Portkey client; routing is handled by its config."""
    client = get_embedding_client()
    logfire.info("[Embedding] Client initialized", gateway="portkey")
    return client


def get_embedding_dim() -> int:
    """Resolve the vector dimension from the Portkey embedding config."""
    if _embedding_dim is None:
        _embed_batch(["embedding dimension probe"])
    if _embedding_dim is None:
        raise RuntimeError("Portkey returned no embedding dimension.")
    return _embedding_dim


def _is_rate_limit_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 429:
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("429", "rate limit", "quota", "resource_exhausted")
    )


def _embed_batch(batch: list[str]) -> list[list[float]]:
    global _embedding_dim

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _client().embeddings.create(input=batch)
            vectors = [item.embedding for item in response.data]
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise RuntimeError("Embedding response contains inconsistent dimensions.")

            dimension = dimensions.pop()
            if _embedding_dim is None:
                _embedding_dim = dimension
                logfire.info(
                    "[Embedding] Dimension resolved",
                    dimension=dimension,
                )
            elif dimension != _embedding_dim:
                raise RuntimeError(
                    "Embedding dimension changed while the application was running: "
                    f"expected {_embedding_dim}, received {dimension}."
                )

            logfire.info(
                "[Embedding] Vectors created",
                input_count=len(batch),
                vector_count=len(vectors),
                dimension=dimension,
                attempt=attempt + 1,
            )
            return vectors
        except Exception as error:
            rate_limited = _is_rate_limit_error(error)
            if not rate_limited or attempt == MAX_ATTEMPTS - 1:
                logfire.exception(
                    "[ERROR][Embedding] Request failed",
                    error=str(error),
                    error_type=type(error).__name__,
                    input_count=len(batch),
                    attempt=attempt + 1,
                    max_attempts=MAX_ATTEMPTS,
                    rate_limited=rate_limited,
                    status_code=getattr(error, "status_code", None),
                )
                raise

            wait = 2**attempt
            logfire.warning(
                "[WARNING][Embedding] Rate limit reached; retrying",
                error=str(error),
                status_code=getattr(error, "status_code", None),
                input_count=len(batch),
                failed_attempt=attempt + 1,
                next_attempt=attempt + 2,
                max_attempts=MAX_ATTEMPTS,
                wait_seconds=wait,
            )
            time.sleep(wait)

    raise RuntimeError("Embedding request exhausted all retry attempts.")


def embed_query(query: str) -> list[float]:
    with logfire.span(
        "[Embedding] Embed query",
        query_length=len(query),
    ):
        return _embed_batch([query])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    batch_count = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    with logfire.span(
        "[Embedding] Embed texts",
        text_count=len(texts),
        batch_count=batch_count,
        batch_size=BATCH_SIZE,
    ):
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
