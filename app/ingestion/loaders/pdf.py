import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format="pdf",
    ):
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        text_parts: list[str] = []
        blank_pages: list[int] = []

        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
            else:
                blank_pages.append(index + 1)

        recovered_page_count = 0
        # Fallback: use pdfplumber for any pages pypdf returned blank
        if blank_pages:
            logfire.info(
                "[Parser] Retry blank PDF pages",
                file=file_path,
                blank_page_count=len(blank_pages),
                blank_pages=blank_pages,
                fallback_parser="pdfplumber",
            )
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    for page_number in blank_pages:
                        page = pdf.pages[page_number - 1]
                        fallback_text = page.extract_text() or ""
                        if fallback_text.strip():
                            text_parts.append(fallback_text)
                            recovered_page_count += 1
            except Exception as error:
                logfire.warning(
                    "[WARNING][Parser] PDF fallback failed",
                    _exc_info=True,
                    error=str(error),
                    error_type=type(error).__name__,
                    file=file_path,
                    fallback_parser="pdfplumber",
                    blank_page_count=len(blank_pages),
                )

        full_text = "\n".join(text_parts)

        if not full_text.strip():
            logfire.warning(
                "[WARNING][Parser] PDF contains no extractable text",
                file=file_path,
                page_count=total_pages,
                blank_page_count=len(blank_pages),
            )
        else:
            logfire.info(
                "[Parser] File parsed",
                file=file_path,
                format="pdf",
                page_count=total_pages,
                blank_page_count=len(blank_pages),
                recovered_page_count=recovered_page_count,
                character_count=len(full_text),
            )

        return full_text
