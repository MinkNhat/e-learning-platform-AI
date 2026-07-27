import logfire
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from app.gateway import LlmTier, get_chat_llm
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    STATIC_DIALOG_RESPONSES,
    YAML_CONTENT,
)


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """Build the Portkey-backed NeMo LLMRails singleton at app startup."""
    global _rails

    with logfire.span("[Guardrails] Initialize", gateway="portkey"):
        guard_llm = get_chat_llm(
            LlmTier.GUARDRAIL,
            feature="guardrails",
            temperature=0.1,
        )

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT,
        )

        _rails = LLMRails(config, llm=guard_llm)


def _normalize_dialog_message(message: str) -> str:
    normalized = "".join(
        character
        for character in message.casefold().replace("’", "'")
        if character.isalnum() or character.isspace() or character == "'"
    )
    return " ".join(normalized.split())


def guard(message: str) -> tuple[bool, str | None]:
    """
    Handle known dialog messages locally, then run the NeMo input rail.

    Returns:
        (True, response) — the message was handled or blocked; skip LangGraph.
        (False, None)    — the message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning(
            "[WARNING][Guardrails] Check skipped",
            reason="not_initialized",
        )
        return False, None

    with logfire.span(
        "[Guardrails] Check request",
        message_length=len(message),
    ):
        dialog_response = STATIC_DIALOG_RESPONSES.get(
            _normalize_dialog_message(message)
        )
        if dialog_response is not None:
            logfire.info(
                "[Guardrails] Request handled by static dialog",
                outcome="handled",
                check_type="static",
                llm_check_count=0,
            )
            return True, dialog_response

        messages = [{"role": "user", "content": message}]
        input_check = _rails.check(messages, rail_types=[RailType.INPUT])

        if input_check.status == RailStatus.BLOCKED:
            logfire.warning(
                "[WARNING][Guardrails] Request blocked",
                outcome="blocked",
                query_preview=message[:200],
                llm_check_count=1,
            )
            return True, input_check.content

        logfire.info(
            "[Guardrails] Request passed",
            outcome="passed",
            llm_check_count=1,
        )
        return False, None
