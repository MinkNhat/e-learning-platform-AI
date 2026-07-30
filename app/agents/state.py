import operator
from typing import Annotated, TypedDict

from app.services.retrieval.models import (
    QueryIntent,
    RecommendationFilters,
    RetrievalScope,
    RetrievalStatus,
    RetrievedChunk,
    Source,
)


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[list[dict], operator.add]
    current_query: str
    intent: QueryIntent
    retrieval_scope: RetrievalScope
    requested_recommendation_filters: RecommendationFilters
    recommendation_filters: RecommendationFilters
    retrieved_chunks: list[RetrievedChunk]
    retrieval_status: RetrievalStatus
    sources: list[Source]
    final_answer: str
