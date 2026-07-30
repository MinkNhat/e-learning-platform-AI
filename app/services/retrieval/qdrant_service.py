import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query
from app.services.retrieval.models import (
    EntityType,
    RecommendationFilters,
    RetrievedChunk,
    Source,
)

MIN_VECTOR_SCORE = 0.35

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    check_compatibility=False,
)


def _match(field: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(
        key=field,
        match=models.MatchValue(value=value),
    )


def _build_filter(
    entity_type: EntityType,
    allowed_course_ids: list[str] | None,
    course_id: str | None,
    module_id: str | None,
    lesson_id: str | None,
    filters: RecommendationFilters | None,
) -> models.Filter:
    must: list[models.Condition] = [_match("entity_type", entity_type.value)]

    if allowed_course_ids:
        must.append(
            models.FieldCondition(
                key="course_id",
                match=models.MatchAny(any=allowed_course_ids),
            )
        )

    for field, value in (
        ("course_id", course_id),
        ("module_id", module_id),
        ("lesson_id", lesson_id),
    ):
        if value:
            must.append(_match(field, value))

    filters = filters or {}
    for input_name, payload_name in (
        ("level", "level_normalized"),
        ("language", "languages_normalized"),
        ("category", "category_keys"),
    ):
        if value := filters.get(input_name):
            must.append(_match(payload_name, " ".join(value.casefold().split())))

    min_price = filters.get("min_price")
    max_price = filters.get("max_price")
    if min_price is not None or max_price is not None:
        must.append(
            models.FieldCondition(
                key="price",
                range=models.Range(gte=min_price, lte=max_price),
            )
        )

    return models.Filter(must=must)


def _source(payload: dict, entity_type: str) -> Source:
    source: Source = {
        "id": str(payload.get("source_id") or payload.get("entity_id")),
        "label": str(payload.get("source_label") or payload.get("source")),
        "entity_type": entity_type,
    }
    for key in (
        "course_id",
        "course_title",
        "module_id",
        "module_title",
        "lesson_id",
        "lesson_title",
    ):
        if payload.get(key):
            source[key] = str(payload[key])
    return source


def _optional_id(payload: dict, key: str) -> str | None:
    return str(payload[key]) if payload.get(key) else None


def search_learning_materials(
    query: str,
    *,
    entity_type: EntityType,
    limit: int = 8,
    allowed_course_ids: list[str] | None = None,
    course_id: str | None = None,
    module_id: str | None = None,
    lesson_id: str | None = None,
    recommendation_filters: RecommendationFilters | None = None,
) -> list[RetrievedChunk]:
    with logfire.span(
        "[Qdrant] Search",
        entity_type=entity_type.value,
        limit=limit,
    ) as span:
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embed_query(query),
            query_filter=_build_filter(
                entity_type,
                allowed_course_ids,
                course_id,
                module_id,
                lesson_id,
                recommendation_filters,
            ),
            limit=limit,
            score_threshold=MIN_VECTOR_SCORE,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "id": str(point.id),
                    "content": str(payload.get("text") or ""),
                    "source": _source(payload, entity_type.value),
                    "entity_type": entity_type.value,
                    "course_id": _optional_id(payload, "course_id"),
                    "module_id": _optional_id(payload, "module_id"),
                    "lesson_id": _optional_id(payload, "lesson_id"),
                    "vector_score": point.score,
                    "rerank_score": None,
                }
            )

        span.set_attribute("result_count", len(results))
        return results
