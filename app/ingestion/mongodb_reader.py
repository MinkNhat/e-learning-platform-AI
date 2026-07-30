from dataclasses import dataclass
from typing import Any

import logfire
from pymongo import MongoClient
from pymongo.database import Database

from app.config import settings

COURSES_COLLECTION = "courses"
MODULES_COLLECTION = "modules"
LESSONS_COLLECTION = "lessons"
CATEGORIES_COLLECTION = "categories"

ACTIVE_FILTER = {
    "isDeleted": {"$ne": True},
    "deletedAt": None,
}

COURSE_PROJECTION = {
    "_id": 1,
    "title": 1,
    "shortDescription": 1,
    "description": 1,
    "objectives": 1,
    "requirement": 1,
    "price": 1,
    "level": 1,
    "languages": 1,
    "category": 1,
}
MODULE_PROJECTION = {
    "_id": 1,
    "name": 1,
    "order": 1,
    "course": 1,
}
LESSON_PROJECTION = {
    "_id": 1,
    "name": 1,
    "content": 1,
    "order": 1,
    "module": 1,
}
CATEGORY_PROJECTION = {
    "_id": 1,
    "name": 1,
    "slug": 1,
}


def reference_id(value: Any) -> str:
    return str(value) if value is not None else ""


@dataclass(slots=True)
class MongoSnapshot:
    courses: list[dict[str, Any]]
    modules: list[dict[str, Any]]
    lessons: list[dict[str, Any]]
    categories: dict[str, dict[str, Any]]


class MongoCourseReader:
    def __init__(
        self,
        uri: str | None = None,
        database_name: str | None = None,
    ) -> None:
        self.uri = uri or settings.MONGODB_READ_URI
        self.database_name = database_name or settings.MONGODB_DATABASE
        if not self.uri:
            raise ValueError("MONGODB_READ_URI is required.")
        if not self.database_name:
            raise ValueError("MONGODB_DATABASE is required.")

    def read(self) -> MongoSnapshot:
        with logfire.span("[MongoDB] Read course data") as span:
            with MongoClient(
                self.uri,
                appname="elearning-rag-indexer",
                serverSelectionTimeoutMS=(settings.MONGODB_SERVER_SELECTION_TIMEOUT_MS),
            ) as client:
                client.admin.command("ping")
                snapshot = self._read_database(client[self.database_name])

            span.set_attributes(
                {
                    "course_count": len(snapshot.courses),
                    "module_count": len(snapshot.modules),
                    "lesson_count": len(snapshot.lessons),
                }
            )
            return snapshot

    def _read_database(self, database: Database) -> MongoSnapshot:
        courses = list(
            database[COURSES_COLLECTION].find(
                {**ACTIVE_FILTER, "isPublished": True},
                COURSE_PROJECTION,
            )
        )
        modules = list(
            database[MODULES_COLLECTION].find(
                {**ACTIVE_FILTER, "isActive": True},
                MODULE_PROJECTION,
            )
        )
        lessons = list(
            database[LESSONS_COLLECTION].find(
                {**ACTIVE_FILTER, "isActive": True},
                LESSON_PROJECTION,
            )
        )

        course_ids = {reference_id(course["_id"]) for course in courses}
        modules = [
            module
            for module in modules
            if reference_id(module.get("course")) in course_ids
        ]

        module_ids = {reference_id(module["_id"]) for module in modules}
        lessons = [
            lesson
            for lesson in lessons
            if reference_id(lesson.get("module")) in module_ids
            and str(lesson.get("content") or "").strip()
        ]

        category_values = [
            course["category"]
            for course in courses
            if course.get("category") is not None
        ]
        category_documents = database[CATEGORIES_COLLECTION].find(
            {
                **ACTIVE_FILTER,
                "isActive": True,
                "_id": {"$in": category_values},
            },
            CATEGORY_PROJECTION,
        )
        categories = {
            reference_id(category["_id"]): category for category in category_documents
        }

        return MongoSnapshot(
            courses=courses,
            modules=modules,
            lessons=lessons,
            categories=categories,
        )
