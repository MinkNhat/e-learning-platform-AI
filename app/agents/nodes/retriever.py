import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
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
        raw_results = search_enterprise_knowledge(query, limit=candidate_limit)

        doc_contents = [doc["content"] for doc in raw_results]
        reranked_contents = rerank_documents(query, doc_contents, top_n=result_limit)

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
        logfire.info(
            "[Retriever] Context ready",
            candidate_count=len(raw_results),
            document_count=len(formatted_docs),
            source_count=len({doc["source"] for doc in raw_results}),
        )

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
