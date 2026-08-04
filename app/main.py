import os
from typing import Annotated, Literal

import logfire
from dotenv import load_dotenv

# Configure tracing before importing application modules.
load_dotenv()
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="rag-api",
)

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.guardrails import initialize_rails
from app.security import RagPrincipal, require_rag_principal
from app.services.query_stream import stream_query

# Initialize FastAPI
app = FastAPI(title="E-learning RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("history content must not be blank.")
        return normalized


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=10_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)

    @field_validator("q")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("q must not be blank.")
        return normalized


def _initial_state(request: QueryRequest, principal: RagPrincipal) -> dict:
    retrieval_scope = principal.retrieval_scope.model_dump(exclude_none=True)
    messages = [message.model_dump() for message in request.history]
    messages.append({"role": "user", "content": request.q})

    return {
        "messages": messages,
        "retrieval_scope": retrieval_scope,
    }


@app.get("/")
def home():
    return {"message": "API trợ lý học tập đang hoạt động."}


@app.post("/query/stream")
def query_stream(
    request: QueryRequest,
    principal: Annotated[RagPrincipal, Depends(require_rag_principal)],
):
    return StreamingResponse(
        stream_query(_initial_state(request, principal)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
