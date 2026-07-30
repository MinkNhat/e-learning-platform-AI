from functools import cache
from pathlib import Path

import logfire
from flashrank import Ranker, RerankRequest

from app.services.retrieval.models import RetrievedChunk

RANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"
RANKER_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "flashrank"


@cache
def _ranker() -> Ranker:
    return Ranker(
        model_name=RANKER_MODEL,
        cache_dir=str(RANKER_CACHE_DIR),
        log_level="WARNING",
    )


def rerank_documents(
    query: str,
    documents: list[RetrievedChunk],
    top_n: int = 5,
) -> list[RetrievedChunk]:
    if not documents:
        return []

    with logfire.span(
        "[Reranker] Rank",
        document_count=len(documents),
        top_n=top_n,
    ):
        passages = [
            {"id": index, "text": document["content"]}
            for index, document in enumerate(documents)
        ]
        results = _ranker().rerank(RerankRequest(query=query, passages=passages))

        reranked: list[RetrievedChunk] = []
        for result in results[:top_n]:
            document = documents[int(result["id"])].copy()
            document["rerank_score"] = float(result["score"])
            reranked.append(document)
        return reranked
