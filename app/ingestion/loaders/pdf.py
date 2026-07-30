import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """Extract PDF text locally with pypdf."""
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format="pdf",
    ) as span:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        span.set_attributes(
            {
                "page_count": len(reader.pages),
                "character_count": len(text),
            }
        )
        return text
