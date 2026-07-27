import logfire


def chunk_text(text: str, chunk_size: int = 1500) -> list[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    with logfire.span(
        "[Chunking] Split text",
        text_length=len(text),
        chunk_size=chunk_size,
    ):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(
            "[Chunking] Text split",
            paragraph_count=len(paragraphs),
            chunk_count=len(valid_chunks),
            max_chunk_length=max(map(len, valid_chunks), default=0),
        )
        return valid_chunks
