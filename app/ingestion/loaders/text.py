import logfire


def parse_text(file_path: str) -> str:
    """
    Parses plain text files.
    """
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format="text",
    ):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()

        logfire.info(
            "[Parser] File parsed",
            file=file_path,
            format="text",
            character_count=len(text),
        )
        return text
