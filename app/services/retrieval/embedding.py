from functools import cache
from time import sleep

import logfire

from app.gateway import get_embedding_client

EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 16
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 60
QUERY_TASK = "retrieval.query"
PASSAGE_TASK = "retrieval.passage"


@cache
def _client():
    return get_embedding_client()


def _embed_batch(texts: list[str], task: str) -> list[list[float]]:
    for retry in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = _client().embeddings.create(
                input=texts,
                dimensions=EMBEDDING_DIM,
                encoding_format="float",
                task=task,
                normalized=True,
            )
            data = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in data]
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise
            if retry == RATE_LIMIT_RETRIES:
                raise
            logfire.warning(
                "[Embedding] Rate limited; retrying",
                wait_seconds=RATE_LIMIT_WAIT_SECONDS,
                retry=retry + 1,
                input_count=len(texts),
            )
            sleep(RATE_LIMIT_WAIT_SECONDS)


def embed_query(query: str) -> list[float]:
    return _embed_batch([query], QUERY_TASK)[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            _embed_batch(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                PASSAGE_TASK,
            )
        )
    return vectors
