import logfire
from unstructured.partition.auto import partition


def parse_office(file_path: str) -> str:
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    file_format = file_path.rsplit(".", 1)[-1].lower()
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format=file_format,
    ):
        # Unstructured automatically detects if it's docx or pptx
        elements = partition(filename=file_path)
        full_text = "\n".join(str(element) for element in elements)

        logfire.info(
            "[Parser] File parsed",
            file=file_path,
            format=file_format,
            element_count=len(elements),
            character_count=len(full_text),
        )
        return full_text
