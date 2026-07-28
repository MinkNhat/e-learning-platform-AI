import re
import unicodedata

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 150


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Normalize and recursively split text without exceeding chunk_size."""
    with logfire.span(
        "[Chunking] Split text",
        text_length=len(text),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        clean_text = normalize_text(text)
        if not clean_text:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " ", ""],
            keep_separator="end",
            length_function=len,
        )
        chunks = splitter.split_text(clean_text)
        max_chunk_length = max(map(len, chunks), default=0)
        if max_chunk_length > chunk_size:
            raise RuntimeError(
                f"Chunk exceeds hard limit: {max_chunk_length} > {chunk_size}."
            )

        logfire.info(
            "[Chunking] Text split",
            source_length=len(text),
            normalized_length=len(clean_text),
            chunk_count=len(chunks),
            max_chunk_length=max_chunk_length,
        )
        return chunks
