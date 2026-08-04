import json
from collections.abc import Iterator

import logfire

from app.agents.graph import rag_agent
from app.guardrails import guard


def _event(name: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n"


def stream_query(initial_state: dict) -> Iterator[str]:
    try:
        yield _event(
            "status",
            {
                "stage": "thinking",
                "message": "Đang suy nghĩ...",
            },
        )
        query = initial_state["messages"][-1]["content"]
        rail_fired, rail_response = guard(query)
        if rail_fired:
            yield _event("token", {"delta": rail_response})
            yield _event("sources", {"items": []})
            yield _event("done", {})
            return

        sources: list[dict] = []

        for mode, payload in rag_agent.stream(
            initial_state,
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom" and payload.get("type") == "token":
                delta = payload.get("delta")
                if isinstance(delta, str) and delta:
                    yield _event("token", {"delta": delta})
                continue

            if mode != "updates":
                continue

            if "planner" in payload:
                if payload["planner"].get("intent") == "conversational":
                    yield _event(
                        "status",
                        {
                            "stage": "synthesizing",
                            "message": "Đang tổng hợp câu trả lời...",
                        },
                    )
                else:
                    yield _event(
                        "status",
                        {
                            "stage": "searching",
                            "message": "Đang tìm kiếm dữ liệu phù hợp...",
                        },
                    )

            if "retriever" in payload:
                yield _event(
                    "status",
                    {
                        "stage": "synthesizing",
                        "message": "Đang tổng hợp câu trả lời...",
                    },
                )

            if "responder" in payload:
                sources = payload["responder"].get("sources", [])

        yield _event("sources", {"items": sources})
        yield _event("done", {})
    except Exception:  # noqa: BLE001 - SSE must convert pipeline failures to an event.
        logfire.exception("[RAG] Stream failed")
        yield _event(
            "error",
            {"message": "Không thể hoàn tất câu trả lời."},
        )
