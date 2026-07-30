from dataclasses import dataclass, field
from typing import Any

from app.services.retrieval.models import EntityType


@dataclass(frozen=True, slots=True)
class IndexChunk:
    """Normalized unit embedded and stored as one Qdrant point."""

    entity_type: EntityType
    entity_id: str
    content: str
    embedding_text: str
    source_id: str
    source_label: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id must not be empty.")
        if not self.content.strip():
            raise ValueError("content must not be empty.")
        if not self.embedding_text.strip():
            raise ValueError("embedding_text must not be empty.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative.")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in self.metadata.items() if value is not None
        }
        payload.update(
            {
                "text": self.content,
                "source": self.source_label,
                "source_id": self.source_id,
                "source_label": self.source_label,
                "entity_type": self.entity_type.value,
                "entity_id": self.entity_id,
                "chunk_index": self.chunk_index,
            }
        )
        return payload
