from functools import cache
from math import ceil

import logfire

from app.gateway import get_embedding_client

BATCH_SIZE = 32
BATCH_TOKEN_LIMIT = 20_000
INPUT_TOKEN_LIMIT = 8_192
ESTIMATED_CHARS_PER_TOKEN = 2.5
EMBEDDING_DIM = 1024
JINA_QUERY_TASK = "retrieval.query"
JINA_PASSAGE_TASK = "retrieval.passage"


@cache
def _client():
    """Create one lazy Portkey client; routing is handled by its config."""
    client = get_embedding_client()
    logfire.info("[Embedding] Client initialized", gateway="portkey")
    return client


def get_embedding_dim() -> int:
    """Return the configured Jina embedding dimension."""
    return EMBEDDING_DIM


def _estimate_tokens(text: str) -> int:
    return max(1, ceil(len(text) / ESTIMATED_CHARS_PER_TOKEN))


def _build_batches(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0

    for text in texts:
        token_count = _estimate_tokens(text)
        if token_count > INPUT_TOKEN_LIMIT:
            raise ValueError(
                "Embedding input exceeds the estimated Jina limit: "
                f"{token_count} > {INPUT_TOKEN_LIMIT} tokens."
            )

        if batch and (
            len(batch) >= BATCH_SIZE
            or batch_tokens + token_count > BATCH_TOKEN_LIMIT
        ):
            batches.append(batch)
            batch = []
            batch_tokens = 0

        batch.append(text)
        batch_tokens += token_count

    if batch:
        batches.append(batch)
    return batches


def _embed_batch(batch: list[str], task: str) -> list[list[float]]:
    try:
        response = _client().embeddings.create(
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
            task=task,
            normalized=True,
        )
        results = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in results]
        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding response count does not match input count: "
                f"expected {len(batch)}, received {len(vectors)}."
            )

        logfire.info(
            "[Embedding] Vectors created",
            provider="jina",
            task=task,
            input_count=len(batch),
            vector_count=len(vectors),
            dimension=EMBEDDING_DIM,
        )
        return vectors
    except Exception as error:
        logfire.exception(
            "[ERROR][Embedding] Request failed",
            error=str(error),
            error_type=type(error).__name__,
            provider="jina",
            task=task,
            input_count=len(batch),
            status_code=getattr(error, "status_code", None),
        )
        raise


def embed_query(query: str) -> list[float]:
    with logfire.span(
        "[Embedding] Embed query",
        query_length=len(query),
    ):
        return _embed_batch(
            _build_batches([query])[0],
            task=JINA_QUERY_TASK,
        )[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    batches = _build_batches(texts)
    with logfire.span(
        "[Embedding] Embed texts",
        text_count=len(texts),
        batch_count=len(batches),
        max_batch_size=BATCH_SIZE,
        batch_token_limit=BATCH_TOKEN_LIMIT,
    ):
        for batch in batches:
            all_embeddings.extend(
                _embed_batch(batch, task=JINA_PASSAGE_TASK)
            )
    return all_embeddings
