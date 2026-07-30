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

from fastapi import Depends, FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.agents.graph import rag_agent
from app.config import settings
from app.guardrails import guard, initialize_rails
from app.security import RagPrincipal, require_rag_principal
from app.services.retrieval.models import (
    QueryIntent,
    RetrievalStatus,
)

# Initialize FastAPI
app = FastAPI(title="E-learning RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()


class RecommendationFiltersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str | None = Field(default=None, max_length=100)
    language: str | None = Field(default=None, max_length=100)
    category: str | None = Field(default=None, max_length=128)
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @field_validator("level", "language", "category")
    @classmethod
    def normalize_filter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).casefold()
        return normalized or None

    @model_validator(mode="after")
    def validate_price_range(self):
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price.")
        return self


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=10_000)
    recommendation_filters: RecommendationFiltersRequest | None = None

    @field_validator("q")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("q must not be blank.")
        return normalized


@app.get("/")
def home():
    return {"message": "API trợ lý học tập đang hoạt động."}


@app.post("/query")
def query(
    request: QueryRequest,
    principal: RagPrincipal = Depends(require_rag_principal),
):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = f"{principal.sub}:{principal.conversation_id}"
    retrieval_scope = principal.retrieval_scope.model_dump(
        exclude_none=True,
    )
    requested_filters = (
        request.recommendation_filters.model_dump(exclude_none=True)
        if request.recommendation_filters
        else {}
    )

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "intent": QueryIntent.CONVERSATIONAL,
        "retrieval_scope": retrieval_scope,
        "requested_recommendation_filters": requested_filters,
        "recommendation_filters": {},
        "retrieved_chunks": [],
        "retrieval_status": RetrievalStatus.OK,
        "sources": [],
        "final_answer": "",
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    with logfire.span(
        "[RAG] Process request",
        thread_id=thread_id,
        query_preview=q[:200],
        query_length=len(q),
        message_count=len(initial_state["messages"]),
        allowed_course_count=len(retrieval_scope.get("allowed_course_ids", [])),
    ) as request_span:
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
