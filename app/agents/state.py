from typing import TypedDict, List, Annotated
import operator

from app.services.retrieval.models import RetrievedChunk


class AgentState(TypedDict):
    # Using Annotated with operator.add ensures that messages
    # are appended to the history rather than replaced.
    messages: Annotated[List[dict], operator.add]
    current_query: str
    retrieved_chunks: List[RetrievedChunk]
    sources: List[str]
    plan: List[str]
    status: str
    final_answer: str
