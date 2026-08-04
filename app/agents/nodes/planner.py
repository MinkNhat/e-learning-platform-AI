import logfire
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.state import AgentState
from app.gateway import LlmTier, get_chat_llm
from app.services.retrieval.models import (
    QueryIntent,
    RecommendationFilters,
)

llm = get_chat_llm(
    LlmTier.PRIMARY,
    feature="planner",
    temperature=0,
)


class PlannerFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str | None = None
    language: str | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @field_validator("level", "language")
    @classmethod
    def normalize_text_filter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).casefold()
        return normalized or None


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: QueryIntent
    search_query: str = ""
    filters: PlannerFilters = Field(default_factory=PlannerFilters)

    @field_validator("search_query")
    @classmethod
    def clean_search_query(cls, value: str) -> str:
        return " ".join(value.split())


def _history_text(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = "Người dùng" if message["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def planner_node(state: AgentState):
    """Classify the request and produce a standalone retrieval query."""
    history = _history_text(state["messages"][:-1])
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    scope = state.get("retrieval_scope", {})
    has_lesson_context = bool(
        scope.get("allowed_course_ids")
        and any(scope.get(key) for key in ("course_id", "module_id", "lesson_id"))
    )

    prompt = f"""
Bạn là bộ lập kế hoạch định tuyến cho trợ lý học tập trực tuyến.

Hãy phân loại tin nhắn mới nhất của người học vào đúng một intent:
- conversational: lời chào, hội thoại thông thường hoặc nội dung có thể trả lời
  hoàn toàn từ lịch sử hội thoại.
- system_qa: câu hỏi dành cho kho kiến thức chung đã lập chỉ mục, không phải yêu
  cầu chọn khóa học và không cần truy cập nội dung bài học được bảo vệ.
- course_recommendation: yêu cầu tìm kiếm, khám phá, so sánh hoặc gợi ý khóa học.
- lesson_qa: câu hỏi về nội dung khóa học, module hoặc bài học, bao gồm các tham
  chiếu như "bài này", "bài học hiện tại" hoặc "module hiện tại".

Hiện có ngữ cảnh trang khóa học đã được xác thực:
{has_lesson_context}.
Tín hiệu này chỉ dùng để hiểu tham chiếu, không tự cấp quyền truy cập. Retriever
sẽ thực thi phân quyền ở bước sau.

Với mọi intent không phải conversational, hãy viết một search_query ngắn gọn,
độc lập và giữ ngôn ngữ của người dùng. Dùng tiếng Việt khi ngôn ngữ chưa rõ.
Hãy giải quyết tham chiếu dựa trên lịch sử nhưng không được tự tạo thông tin hay
mã định danh khóa học.

Chỉ với course_recommendation, hãy trích xuất các bộ lọc được nêu rõ:
level, language, min_price, max_price. Dùng null nếu không có. Chủ đề hoặc công
nghệ như Java, Python và React phải nằm trong search_query, không được chuyển
thành bộ lọc. Chuẩn hóa level thành beginner, intermediate hoặc advanced. Chuẩn
hóa tên ngôn ngữ về nhãn tiếng Anh như Vietnamese hoặc English. Giữ giá tiền
dưới dạng số VND đầy đủ, ví dụ 500k thành 500000.

LỊCH SỬ HỘI THOẠI:
{history}

TIN NHẮN MỚI NHẤT:
{user_message}

Chỉ trả về JSON theo đúng cấu trúc sau, không dùng Markdown:
{{
  "intent": "conversational|system_qa|course_recommendation|lesson_qa",
  "search_query": "",
  "filters": {{
    "level": null,
    "language": null,
    "min_price": null,
    "max_price": null
  }}
}}
"""

    with logfire.span(
        "[Planner] Select route",
        message_count=len(state["messages"]),
        history_length=len(history),
        user_message_length=len(user_message),
        has_lesson_context=has_lesson_context,
    ) as span:
        content = llm.invoke(prompt).content
        if not isinstance(content, str):
            raise TypeError("Planner must return JSON text.")
        decision = PlannerDecision.model_validate_json(content)

        if decision.intent != QueryIntent.CONVERSATIONAL and not decision.search_query:
            decision.search_query = user_message

        filters: RecommendationFilters = decision.filters.model_dump(exclude_none=True)
        span.set_attributes(
            {
                "intent": decision.intent.value,
                "search_query": decision.search_query,
                "filter_count": len(filters),
            }
        )

    return {
        "intent": decision.intent,
        "current_query": decision.search_query,
        "recommendation_filters": filters,
    }
