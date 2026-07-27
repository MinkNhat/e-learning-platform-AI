import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    with logfire.span(
        "[Qdrant] Search knowledge",
        collection=settings.QDRANT_COLLECTION,
        query=query,
        limit=limit,
    ) as search_span:
        try:
            query_vector = embed_query(query)

            # Using query_points - the modern standard for Qdrant
            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            results = []
            for res in response.points:
                results.append(
                    {
                        "content": res.payload.get("text", ""),
                        "source": res.payload.get("source", "Unknown"),
                        "score": res.score,
                    }
                )

            top_score = results[0]["score"] if results else None
            search_span.set_attributes(
                {
                    "outcome": "completed",
                    "result_count": len(results),
                    "top_score": top_score,
                }
            )
            logfire.info(
                "[Qdrant] Search completed",
                result_count=len(results),
                top_score=top_score,
                source_count=len({result["source"] for result in results}),
                vector_dimension=len(query_vector),
            )
            return results
        except Exception as error:
            search_span.set_attribute("outcome", "error")
            search_span.set_level("error")
            logfire.exception(
                "[ERROR][Qdrant] Search failed",
                error=str(error),
                error_type=type(error).__name__,
                collection=settings.QDRANT_COLLECTION,
                limit=limit,
            )
            return []
