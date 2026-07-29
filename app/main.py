# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="rag-api",
)

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="E-learning RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "E-learning LangGraph RAG API is live."}


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "retrieved_chunks": [],
        "sources": [],
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    with logfire.span(
        "[RAG] Process request",
        thread_id=thread_id,
        query_preview=q[:200],
        query_length=len(q),
        message_count=len(initial_state["messages"]),
    ) as request_span:
        try:
            # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
            rail_fired, rail_response = guard(q)
            if rail_fired:
                request_span.set_attribute("outcome", "blocked")
                return {
                    "answer": rail_response,
                    "sources": [],
                }

            # Gate 2: LangGraph RAG pipeline
            # Run the graph synchronously to preserve Logfire context variables
            final_output = rag_agent.invoke(initial_state, config=config)
            retrieved_chunks = final_output.get("retrieved_chunks", [])
            sources = final_output.get("sources", [])

            request_span.set_attributes(
                {
                    "outcome": "completed",
                    "document_count": len(retrieved_chunks),
                    "source_count": len(sources),
                }
            )
            return {
                "answer": final_output.get("final_answer"),
                "sources": sources,
            }
        except Exception as error:
            request_span.set_attribute("outcome", "error")
            request_span.set_level("error")
            logfire.exception(
                "[ERROR][RAG] Request failed",
                error=str(error),
                error_type=type(error).__name__,
                thread_id=thread_id,
            )
            return {
                "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
                "sources": [],
            }
