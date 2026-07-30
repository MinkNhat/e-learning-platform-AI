from collections import defaultdict
from typing import Any

from app.ingestion.chunking.splitter import chunk_text, normalize_text
from app.ingestion.loaders.html import parse_html_content
from app.ingestion.models import IndexChunk
from app.ingestion.mongodb_reader import MongoSnapshot, reference_id
from app.services.retrieval.models import EntityType


def _plain_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "<" in text and ">" in text:
        text = parse_html_content(text)
    return normalize_text(text)


def _normalized_keyword(value: Any) -> str:
    return _plain_text(value).casefold()


def _order(document: dict[str, Any]) -> int:
    try:
        return int(document.get("order") or 0)
    except (TypeError, ValueError):
        return 0


def _objective_lines(objectives: Any) -> list[str]:
    if not isinstance(objectives, list):
        return []
    return [
        f"- {cleaned}"
        for objective in objectives
        if (cleaned := _plain_text(objective))
    ]


def _course_metadata(
    course: dict[str, Any],
    category: dict[str, Any] | None,
) -> dict[str, Any]:
    languages = [
        _plain_text(language)
        for language in course.get("languages") or []
        if _plain_text(language)
    ]
    category_name = _plain_text((category or {}).get("name"))
    category_keys = list(
        dict.fromkeys(
            value
            for value in (
                reference_id(course.get("category")),
                category_name.casefold(),
                str((category or {}).get("slug") or "").casefold(),
            )
            if value
        )
    )
    price = course.get("price")
    return {
        "course_id": reference_id(course.get("_id")),
        "course_title": _plain_text(course.get("title")),
        "category_keys": category_keys,
        "level_normalized": _normalized_keyword(course.get("level")) or None,
        "languages_normalized": [language.casefold() for language in languages],
        "price": float(price) if price is not None else None,
    }


def _build_course_profile(
    course: dict[str, Any],
    category: dict[str, Any] | None,
) -> list[IndexChunk]:
    course_id = reference_id(course.get("_id"))
    title = _plain_text(course.get("title")) or "Khóa học chưa đặt tên"
    category_name = _plain_text((category or {}).get("name"))
    languages = [
        _plain_text(language)
        for language in course.get("languages") or []
        if _plain_text(language)
    ]

    lines = [f"Khóa học: {title}"]
    fields = (
        ("Mô tả ngắn", course.get("shortDescription")),
        ("Mô tả", course.get("description")),
        ("Yêu cầu", course.get("requirement")),
    )
    for label, value in fields:
        cleaned = _plain_text(value)
        if cleaned:
            lines.extend(["", f"{label}: {cleaned}"])

    objectives = _objective_lines(course.get("objectives"))
    if objectives:
        lines.extend(["", "Mục tiêu:", *objectives])
    if category_name:
        lines.extend(["", f"Danh mục: {category_name}"])
    if level := _plain_text(course.get("level")):
        lines.append(f"Trình độ: {level}")
    if languages:
        lines.append(f"Ngôn ngữ: {', '.join(languages)}")
    if course.get("price") is not None:
        lines.append(f"Giá: {course.get('price')}")

    metadata = _course_metadata(course, category)
    chunks = chunk_text("\n".join(lines))
    return [
        IndexChunk(
            entity_type=EntityType.COURSE_PROFILE,
            entity_id=course_id,
            content=content,
            embedding_text=(
                f"Khóa học: {title}\nLoại nội dung: hồ sơ khóa học\n\n{content}"
            ),
            source_id=course_id,
            source_label=title,
            chunk_index=index,
            metadata=metadata,
        )
        for index, content in enumerate(chunks)
    ]


def _build_lesson_chunks(
    course: dict[str, Any],
    module: dict[str, Any],
    lesson: dict[str, Any],
) -> list[IndexChunk]:
    course_title = _plain_text(course.get("title")) or "Khóa học chưa đặt tên"
    module_title = _plain_text(module.get("name")) or "Chương chưa đặt tên"
    lesson_id = reference_id(lesson.get("_id"))
    lesson_title = _plain_text(lesson.get("name")) or "Bài học chưa đặt tên"
    clean_content = parse_html_content(str(lesson.get("content") or ""))
    chunks = chunk_text(clean_content)

    metadata = {
        "course_id": reference_id(course.get("_id")),
        "course_title": course_title,
        "module_id": reference_id(module.get("_id")),
        "module_title": module_title,
        "lesson_id": lesson_id,
        "lesson_title": lesson_title,
    }
    source_label = f"{course_title} > {module_title} > {lesson_title}"
    return [
        IndexChunk(
            entity_type=EntityType.LESSON_CHUNK,
            entity_id=lesson_id,
            content=content,
            embedding_text=(
                f"Khóa học: {course_title}\n"
                f"Chương: {module_title}\n"
                f"Bài học: {lesson_title}\n"
                "Loại nội dung: nội dung bài học\n\n"
                f"{content}"
            ),
            source_id=lesson_id,
            source_label=source_label,
            chunk_index=index,
            metadata=metadata,
        )
        for index, content in enumerate(chunks)
    ]


def build_mongodb_chunks(snapshot: MongoSnapshot) -> list[IndexChunk]:
    courses_by_id = {
        reference_id(course.get("_id")): course for course in snapshot.courses
    }
    modules_by_course: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for module in snapshot.modules:
        modules_by_course[reference_id(module.get("course"))].append(module)

    lessons_by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lesson in snapshot.lessons:
        lessons_by_module[reference_id(lesson.get("module"))].append(lesson)

    chunks: list[IndexChunk] = []
    for course_id, course in courses_by_id.items():
        category = snapshot.categories.get(reference_id(course.get("category")))
        chunks.extend(_build_course_profile(course, category))

        for module in sorted(modules_by_course[course_id], key=_order):
            module_id = reference_id(module.get("_id"))
            module_lessons = lessons_by_module[module_id]
            for lesson in sorted(module_lessons, key=_order):
                chunks.extend(
                    _build_lesson_chunks(
                        course,
                        module,
                        lesson,
                    )
                )

    return chunks
