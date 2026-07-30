import logfire
from bs4 import BeautifulSoup


def parse_html_content(content: str) -> str:
    """Convert HTML into readable text while retaining document structure."""
    soup = BeautifulSoup(content or "", "html.parser")

    for element in soup(["script", "style", "meta", "noscript", "template"]):
        element.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for row in soup.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            row.replace_with(f"\n{' | '.join(cells)}\n")

    for item in soup.find_all("li"):
        item.insert_before("\n- ")
        item.append("\n")

    for block in soup.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "blockquote",
            "pre",
            "section",
            "article",
        ]
    ):
        block.insert_before("\n")
        block.append("\n")

    text = soup.get_text(separator=" ")
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


def parse_html(file_path: str) -> str:
    """Parse an HTML file with the same structured representation as lessons."""
    with logfire.span(
        "[Parser] Parse file",
        file=file_path,
        format="html",
    ):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        clean_text = parse_html_content(content)

        logfire.info(
            "[Parser] File parsed",
            file=file_path,
            format="html",
            source_character_count=len(content),
            character_count=len(clean_text),
        )
        return clean_text
