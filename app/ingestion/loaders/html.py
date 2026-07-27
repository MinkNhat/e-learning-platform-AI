import logfire
from bs4 import BeautifulSoup


def parse_html(file_path: str) -> str:
    """
    Parses HTML content using BeautifulSoup.
    Cleans scripts, styles, and extracts readable text for RAG.
    """
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format="html",
    ):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        soup = BeautifulSoup(content, "html.parser")

        # 1. Remove Junk (Scripts, Styles, Metadata)
        for script in soup(["script", "style", "meta", "noscript"]):
            script.decompose()

        # 2. Extract Text
        text = soup.get_text(separator="\n")

        # 3. Clean Whitespace (Collapse multiple newlines)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        logfire.info(
            "[Parser] File parsed",
            file=file_path,
            format="html",
            source_character_count=len(content),
            character_count=len(clean_text),
        )
        return clean_text
