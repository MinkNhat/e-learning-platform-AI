import re

VIETNAMESE_CHARACTERS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)


def prefers_english_fallback(text: str) -> bool:
    """Detect clear English for deterministic messages; default to Vietnamese."""
    normalized = " ".join(text.casefold().split())
    if re.search(
        r"\b(trả lời|phản hồi|viết|nói)\b.*\btiếng anh\b",
        normalized,
    ):
        return True
    if any(character in VIETNAMESE_CHARACTERS for character in normalized):
        return False

    english_prefixes = (
        "what ",
        "how ",
        "why ",
        "when ",
        "where ",
        "explain ",
        "summarize ",
        "recommend ",
        "compare ",
        "can you ",
        "could you ",
        "please ",
    )
    if normalized.startswith(english_prefixes):
        return True

    tokens = set(re.findall(r"[a-z]+", normalized))
    english_markers = {
        "the",
        "this",
        "that",
        "is",
        "are",
        "lesson",
        "course",
        "help",
        "learn",
        "study",
        "answer",
        "english",
    }
    return len(tokens & english_markers) >= 2
