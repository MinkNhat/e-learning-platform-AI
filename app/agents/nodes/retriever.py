import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_learning_materials
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Perform vector search and semantic reranking for learning queries.
    """
    query = state["current_query"]

    candidate_limit = 15
    result_limit = 5
    with logfire.span(
        "[Retriever] Retrieve context",
        query=query,
        candidate_limit=candidate_limit,
        result_limit=result_limit,
    ):
        candidates = search_learning_materials(query, limit=candidate_limit)
        retrieved_chunks = rerank_documents(
            query,
            candidates,
            top_n=result_limit,
        )
        logfire.info(
            "[Retriever] Context ready",
            candidate_count=len(candidates),
            document_count=len(retrieved_chunks),
            source_count=len(
                {chunk["source"] for chunk in retrieved_chunks}
            ),
        )

    return {
        "retrieved_chunks": retrieved_chunks,
        "status": "Found relevant learning material.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
