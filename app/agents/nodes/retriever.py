import logfire

from app.agents.state import AgentState
from app.services.retrieval.models import (
    EntityType,
    QueryIntent,
    RetrievalStatus,
    RetrievedChunk,
)
from app.services.retrieval.qdrant_service import search_learning_materials
from app.services.retrieval.ranking_service import rerank_documents

CANDIDATE_LIMIT = 15
RESULT_LIMIT = 5
MIN_RERANK_SCORE = 0.01


def _lesson_candidates(
    query: str,
    state: AgentState,
) -> tuple[list[RetrievedChunk], RetrievalStatus]:
    scope = state.get("retrieval_scope", {})
    allowed_course_ids = scope.get("allowed_course_ids", [])
    if not allowed_course_ids:
        return [], RetrievalStatus.SCOPE_REQUIRED

    course_id = scope.get("course_id")
    if course_id and course_id not in allowed_course_ids:
        return [], RetrievalStatus.SCOPE_FORBIDDEN

    return search_learning_materials(
        query,
        entity_type=EntityType.LESSON_CHUNK,
        limit=CANDIDATE_LIMIT,
        allowed_course_ids=allowed_course_ids,
        course_id=course_id,
        module_id=scope.get("module_id"),
        lesson_id=scope.get("lesson_id"),
    ), RetrievalStatus.OK


def retrieve_node(state: AgentState):
    query = state["current_query"]
    intent = state["intent"]

    with logfire.span("[Retriever] Retrieve", intent=intent.value) as span:
        status = RetrievalStatus.OK
        if intent == QueryIntent.SYSTEM_QA:
            candidates = search_learning_materials(
                query,
                entity_type=EntityType.GENERAL_CHUNK,
                limit=CANDIDATE_LIMIT,
            )
        elif intent == QueryIntent.COURSE_RECOMMENDATION:
            candidates = search_learning_materials(
                query,
                entity_type=EntityType.COURSE_PROFILE,
                limit=CANDIDATE_LIMIT,
                recommendation_filters=state.get("recommendation_filters"),
            )
        elif intent == QueryIntent.LESSON_QA:
            candidates, status = _lesson_candidates(query, state)
        else:
            candidates = []

        if status != RetrievalStatus.OK:
            return {
                "retrieved_chunks": [],
                "retrieval_status": status,
            }

        documents = [
            document
            for document in rerank_documents(
                query,
                candidates,
                top_n=RESULT_LIMIT,
            )
            if (document["rerank_score"] or 0) >= MIN_RERANK_SCORE
        ]
        status = RetrievalStatus.OK if documents else RetrievalStatus.INSUFFICIENT_DATA
        span.set_attributes(
            {
                "candidate_count": len(candidates),
                "document_count": len(documents),
                "status": status.value,
            }
        )
        return {
            "retrieved_chunks": documents,
            "retrieval_status": status,
        }
