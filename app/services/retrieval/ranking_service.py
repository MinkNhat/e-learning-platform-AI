import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        cache_dir = "/tmp/flashrank"
        with logfire.span(
            "[Reranker] Initialize model",
            engine="flashrank",
            cache_dir=cache_dir,
        ):
            try:
                # We use a specific cache directory to avoid permission issues in production
                _ranker = Ranker(cache_dir=cache_dir)
            except Exception as error:
                logfire.warning(
                    "[WARNING][Reranker] Custom cache unavailable; using default",
                    _exc_info=True,
                    error=str(error),
                    error_type=type(error).__name__,
                    cache_dir=cache_dir,
                )
                _ranker = Ranker()

            logfire.info("[Reranker] Model initialized", engine="flashrank")
    return _ranker


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.

    Why FlashRank?
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    with logfire.span(
        "[Reranker] Rank documents",
        query=query,
        document_count=len(documents),
        top_n=top_n,
    ) as rerank_span:
        try:
            ranker = _get_ranker()

            # FlashRank expects a list of dictionaries with 'id' and 'text'
            passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]

            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            # Results are returned sorted by highest semantic score first
            reranked_docs = [result["text"] for result in results[:top_n]]
            top_score = results[0]["score"] if results else None
            rerank_span.set_attributes(
                {
                    "outcome": "completed",
                    "result_count": len(reranked_docs),
                    "top_score": top_score,
                }
            )
            logfire.info(
                "[Reranker] Documents ranked",
                result_count=len(reranked_docs),
                top_score=top_score,
            )
            return reranked_docs

        except Exception as error:
            fallback_documents = documents[:top_n]
            rerank_span.set_attributes(
                {
                    "outcome": "fallback",
                    "result_count": len(fallback_documents),
                }
            )
            rerank_span.set_level("warning")
            logfire.warning(
                "[WARNING][Reranker] Ranking failed; using vector order",
                _exc_info=True,
                error=str(error),
                error_type=type(error).__name__,
                document_count=len(documents),
                result_count=len(fallback_documents),
            )
            return fallback_documents
