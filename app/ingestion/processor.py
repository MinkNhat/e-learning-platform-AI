import sys
import uuid
from pathlib import Path

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".docx", ".pptx"}

logfire.configure(service_name="ingestion-service")

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def _point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))


def _parse_file(file_path: Path) -> str:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return parse_pdf(str(file_path))
    if extension in {".html", ".htm"}:
        return parse_html(str(file_path))
    if extension == ".txt":
        return parse_text(str(file_path))
    if extension in {".docx", ".pptx"}:
        from app.ingestion.loaders.office import parse_office

        return parse_office(str(file_path))
    raise ValueError(f"Unsupported file format: {extension}")


def process_file(file_path: Path) -> int:
    """Parse, chunk, embed, and index one learning-material file."""
    source = file_path.name
    with logfire.span(
        "[Ingestion] Process file",
        file=source,
        file_path=str(file_path),
    ) as file_span:
        text = _parse_file(file_path)
        chunks = chunk_text(text)
        if not chunks:
            file_span.set_attribute("outcome", "skipped")
            logfire.warning(
                "[Ingestion] File skipped; no text extracted",
                file=source,
            )
            return 0

        embeddings = embed_texts(chunks)
        points = [
            models.PointStruct(
                id=_point_id(source, index),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    "source_id": source,
                    "source_label": source,
                    "chunk_index": index,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
        ]
        qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )

        file_span.set_attributes(
            {
                "outcome": "indexed",
                "character_count": len(text),
                "chunk_count": len(chunks),
            }
        )
        logfire.info(
            "[Ingestion] File indexed",
            file=source,
            character_count=len(text),
            chunk_count=len(chunks),
        )
        return len(points)


def run_ingestion(base_dir: Path) -> None:
    """Rebuild the e-learning collection from supported files in base_dir."""
    with logfire.span(
        "[Ingestion] Run",
        base_directory=str(base_dir),
    ) as ingestion_span:
        if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
            logfire.info(
                "[Qdrant] Collection deleted",
                collection=settings.QDRANT_COLLECTION,
            )

        embedding_dim = get_embedding_dim()
        qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logfire.info(
            "[Qdrant] Collection created",
            collection=settings.QDRANT_COLLECTION,
            vector_dimension=embedding_dim,
            distance="cosine",
        )

        files = sorted(
            path
            for path in base_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        point_count = sum(process_file(path) for path in files)

        ingestion_span.set_attributes(
            {
                "outcome": "completed",
                "file_count": len(files),
                "point_count": point_count,
            }
        )
        logfire.info(
            "[Ingestion] Run completed",
            base_directory=str(base_dir),
            file_count=len(files),
            point_count=point_count,
        )


if __name__ == "__main__":
    # python -m app.ingestion.processor DATA
    target_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "DATA")

    if not target_dir.is_dir():
        print(f"Error: directory '{target_dir}' does not exist.")
        sys.exit(1)

    run_ingestion(target_dir)
