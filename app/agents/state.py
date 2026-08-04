from typing import TypedDict

from app.services.retrieval.models import (
    QueryIntent,
    RecommendationFilters,
    RetrievalScope,
    RetrievalStatus,
    RetrievedChunk,
    Source,
)


class AgentState(TypedDict):
    messages: list[dict]
    current_query: str
    intent: QueryIntent
    retrieval_scope: RetrievalScope
    recommendation_filters: RecommendationFilters
    retrieved_chunks: list[RetrievedChunk]
    retrieval_status: RetrievalStatus
    sources: list[Source]
