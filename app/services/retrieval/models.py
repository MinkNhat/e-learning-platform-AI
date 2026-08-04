from enum import StrEnum
from typing import NotRequired, TypedDict


class EntityType(StrEnum):
    GENERAL_CHUNK = "general_chunk"
    COURSE_PROFILE = "course_profile"
    LESSON_CHUNK = "lesson_chunk"


class QueryIntent(StrEnum):
    CONVERSATIONAL = "conversational"
    SYSTEM_QA = "system_qa"
    COURSE_RECOMMENDATION = "course_recommendation"
    LESSON_QA = "lesson_qa"


class RetrievalStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"
    SCOPE_REQUIRED = "scope_required"
    SCOPE_FORBIDDEN = "scope_forbidden"


class Source(TypedDict):
    id: str
    label: str
    entity_type: str
    course_id: NotRequired[str]
    course_title: NotRequired[str]
    module_id: NotRequired[str]
    module_title: NotRequired[str]
    lesson_id: NotRequired[str]
    lesson_title: NotRequired[str]


class RetrievalScope(TypedDict, total=False):
    allowed_course_ids: list[str]
    course_id: str
    module_id: str
    lesson_id: str


class RecommendationFilters(TypedDict, total=False):
    level: str
    language: str
    min_price: float
    max_price: float


class RetrievedChunk(TypedDict):
    id: str
    content: str
    source: Source
    entity_type: str
    course_id: str | None
    module_id: str | None
    lesson_id: str | None
    vector_score: float
    rerank_score: float | None
