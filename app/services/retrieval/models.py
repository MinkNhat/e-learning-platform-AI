from typing import TypedDict


class RetrievedChunk(TypedDict):
    id: str
    content: str
    source: str
    vector_score: float
    rerank_score: float | None
