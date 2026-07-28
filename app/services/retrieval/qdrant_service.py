import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.models import RetrievedChunk


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def search_learning_materials(
    query: str,
    limit: int = 8,
) -> list[RetrievedChunk]:
    """
    Perform a high-precision search across indexed learning materials.

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

            results: list[RetrievedChunk] = []
            for res in response.points:
                payload = res.payload or {}
                results.append(
                    {
                        "id": str(res.id),
                        "content": payload.get("text", ""),
                        "source": payload.get("source", "Unknown"),
                        "vector_score": res.score,
                        "rerank_score": None,
                    }
                )

            top_score = results[0]["vector_score"] if results else None
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
