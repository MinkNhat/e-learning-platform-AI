import os
import sys
import uuid

import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

logfire.configure(service_name="enterprise-ingestion-service")

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def process_file(file_path: str, filename: str, source_type: str):
    """Parse, chunk, embed, and index a document in Qdrant."""
    with logfire.span(
        "[Ingestion] Process file",
        file=filename,
        file_path=file_path,
        source_type=source_type,
    ) as file_span:
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office

                full_text = parse_office(file_path)
            else:
                file_span.set_attributes(
                    {"outcome": "skipped", "skip_reason": "unsupported_format"}
                )
                logfire.warning(
                    "[WARNING][Ingestion] File skipped; unsupported format",
                    file=filename,
                    file_path=file_path,
                    format=ext,
                    source_type=source_type,
                )
                return

            if not full_text or not full_text.strip():
                file_span.set_attributes(
                    {"outcome": "skipped", "skip_reason": "empty_text"}
                )
                logfire.warning(
                    "[WARNING][Ingestion] File skipped; no text extracted",
                    file=filename,
                    file_path=file_path,
                    format=ext,
                    source_type=source_type,
                )
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                file_span.set_attributes(
                    {"outcome": "skipped", "skip_reason": "no_chunks"}
                )
                logfire.warning(
                    "[WARNING][Ingestion] File skipped; no chunks created",
                    file=filename,
                    character_count=len(full_text),
                    source_type=source_type,
                )
                return

            # 3. Embed and index in Qdrant
            with logfire.span(
                "[Ingestion] Embed and index file",
                file=filename,
                chunk_count=len(chunks),
                collection=settings.QDRANT_COLLECTION,
            ):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )

            file_span.set_attributes(
                {
                    "outcome": "indexed",
                    "character_count": len(full_text),
                    "chunk_count": len(chunks),
                    "point_count": len(points),
                }
            )
            logfire.info(
                "[Ingestion] File indexed",
                file=filename,
                source_type=source_type,
                character_count=len(full_text),
                chunk_count=len(chunks),
                point_count=len(points),
                collection=settings.QDRANT_COLLECTION,
            )

        except Exception as error:
            file_span.set_attribute("outcome", "error")
            file_span.set_level("error")
            logfire.exception(
                "[ERROR][Ingestion] File processing failed",
                error=str(error),
                error_type=type(error).__name__,
                file=filename,
                file_path=file_path,
                source_type=source_type,
            )


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span(
        "[Ingestion] Scan directory",
        directory=dir_path,
        source_type=source_type,
    ):
        files = [
            file
            for file in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, file))
        ]
        logfire.info(
            "[Ingestion] Directory scanned",
            directory=dir_path,
            source_type=source_type,
            file_count=len(files),
        )
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str | None = None,
    wipe: bool = False,
):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span(
        "[Ingestion] Run",
        base_directory=base_dir,
        explicit_source_type=explicit_source_type,
        wipe=wipe,
    ) as ingestion_span:

        # Wipe collection if requested
        if wipe:
            with logfire.span(
                "[Qdrant] Reset collection",
                collection=settings.QDRANT_COLLECTION,
            ):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(
                        "[Qdrant] Collection deleted",
                        collection=settings.QDRANT_COLLECTION,
                    )

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                "[Qdrant] Collection created",
                collection=settings.QDRANT_COLLECTION,
                vector_dimension=dim,
                distance="cosine",
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(
                "[Ingestion] Source selected",
                directory=base_dir,
                source_type=source_type,
                selection="explicit" if explicit_source_type else "inferred",
            )
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

        ingestion_span.set_attributes(
            {
                "outcome": "completed",
                "directory_count": len(subdirs) or 1,
            }
        )
        logfire.info(
            "[Ingestion] Run completed",
            base_directory=base_dir,
            directory_count=len(subdirs) or 1,
        )


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        logfire.error(
            "[ERROR][Ingestion] Input path does not exist",
            path=target_dir,
        )
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(
        target_dir,
        explicit_source_type=explicit_type,
        wipe=wipe_requested,
    )
