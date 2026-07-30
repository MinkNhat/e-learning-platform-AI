import sys
from collections import Counter
from pathlib import Path

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.course_chunks import build_mongodb_chunks
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.ingestion.models import IndexChunk
from app.ingestion.mongodb_reader import MongoCourseReader
from app.services.retrieval.embedding import EMBEDDING_DIM, embed_texts
from app.services.retrieval.models import EntityType

INDEX_BATCH_SIZE = 16
QDRANT_TIMEOUT_SECONDS = 120
SOURCE_SAMPLE_SIZE = 8

FILE_LOADERS = {
    ".pdf": parse_pdf,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
    ".docx": parse_office,
    ".pptx": parse_office,
}

PAYLOAD_INDEXES = {
    "entity_type": models.PayloadSchemaType.KEYWORD,
    "course_id": models.PayloadSchemaType.KEYWORD,
    "module_id": models.PayloadSchemaType.KEYWORD,
    "lesson_id": models.PayloadSchemaType.KEYWORD,
    "level_normalized": models.PayloadSchemaType.KEYWORD,
    "languages_normalized": models.PayloadSchemaType.KEYWORD,
    "category_keys": models.PayloadSchemaType.KEYWORD,
    "price": models.PayloadSchemaType.FLOAT,
}

logfire.configure(
    service_name="ingestion-service",
    send_to_logfire="if-token-present",
)

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=QDRANT_TIMEOUT_SECONDS,
    check_compatibility=False,
)


def build_file_chunks(file_path: Path) -> list[IndexChunk]:
    loader = FILE_LOADERS[file_path.suffix.lower()]
    source = file_path.name
    return [
        IndexChunk(
            entity_type=EntityType.GENERAL_CHUNK,
            entity_id=source,
            content=content,
            embedding_text=f"Tài liệu: {source}\n\n{content}",
            source_id=source,
            source_label=source,
            chunk_index=index,
            metadata={"file_extension": file_path.suffix.lower()},
        )
        for index, content in enumerate(chunk_text(loader(str(file_path))))
    ]


def _load_file_chunks(base_dir: Path) -> tuple[list[IndexChunk], int]:
    files = sorted(
        path
        for path in base_dir.iterdir()
        if path.is_file() and path.suffix.lower() in FILE_LOADERS
    )
    return [
        chunk for file_path in files for chunk in build_file_chunks(file_path)
    ], len(files)


def _recreate_collection() -> None:
    name = settings.QDRANT_COLLECTION
    if qdrant_client.collection_exists(name):
        qdrant_client.delete_collection(
            name,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )

    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        ),
        timeout=QDRANT_TIMEOUT_SECONDS,
    )
    for field_name, field_schema in PAYLOAD_INDEXES.items():
        qdrant_client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )


def _index_chunks(chunks: list[IndexChunk]) -> None:
    batch_count = (len(chunks) + INDEX_BATCH_SIZE - 1) // INDEX_BATCH_SIZE
    for batch_number, start in enumerate(
        range(0, len(chunks), INDEX_BATCH_SIZE),
        start=1,
    ):
        batch = chunks[start : start + INDEX_BATCH_SIZE]
        sources = list(dict.fromkeys(chunk.source_label for chunk in batch))
        with logfire.span(
            "[Ingestion] Index batch",
            batch_number=batch_number,
            batch_count=batch_count,
            point_count=len(batch),
            source_sample=sources[:SOURCE_SAMPLE_SIZE],
        ):
            vectors = embed_texts([chunk.embedding_text for chunk in batch])
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk.to_payload(),
                )
                for point_id, (chunk, vector) in enumerate(
                    zip(batch, vectors, strict=True),
                    start=start,
                )
            ]
            qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
                wait=True,
                timeout=QDRANT_TIMEOUT_SECONDS,
            )


def run_ingestion(base_dir: Path) -> dict[str, object]:
    if not base_dir.is_dir():
        raise ValueError(f"File source directory does not exist: {base_dir}")

    with logfire.span("[Ingestion] Rebuild collection") as span:
        file_chunks, file_count = _load_file_chunks(base_dir)
        mongo_chunks = build_mongodb_chunks(MongoCourseReader().read())
        chunks = file_chunks + mongo_chunks
        if not chunks:
            raise RuntimeError("No indexable content was found.")

        entity_counts = Counter(chunk.entity_type.value for chunk in chunks)
        logfire.info(
            "[Ingestion] Chunks ready",
            file_count=file_count,
            point_count=len(chunks),
            entity_counts=dict(entity_counts),
        )

        _recreate_collection()
        _index_chunks(chunks)

        actual_count = qdrant_client.count(
            collection_name=settings.QDRANT_COLLECTION,
            exact=True,
        ).count
        if actual_count != len(chunks):
            raise RuntimeError(f"Expected {len(chunks)} points, found {actual_count}.")

        result = {
            "collection": settings.QDRANT_COLLECTION,
            "point_count": actual_count,
            "entity_counts": dict(entity_counts),
        }
        span.set_attributes(result)
        logfire.info("[Ingestion] Collection rebuilt", **result)
        return result


if __name__ == "__main__":
    # python -m app.ingestion.processor DATA
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "DATA")
    summary = run_ingestion(directory)
    print(f"Rebuilt {summary['collection']} with {summary['point_count']} points.")
