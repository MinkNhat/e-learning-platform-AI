from enum import StrEnum

import httpx
from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, Portkey, createHeaders

from app.config import settings


ROUTED_MODEL = "portkey-default"
HTTP_CLIENT_OPTIONS = (
    {"http_client": httpx.Client(proxy=settings.PORTKEY_HTTP_PROXY)}
    if settings.PORTKEY_HTTP_PROXY
    else {}
)


class LlmTier(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


_CONFIG_IDS = {
    LlmTier.PRIMARY: settings.PORTKEY_PRIMARY_CONFIG_ID,
    LlmTier.SECONDARY: settings.PORTKEY_SECONDARY_CONFIG_ID,
    LlmTier.GUARDRAIL: settings.PORTKEY_GUARDRAIL_CONFIG_ID,
}


def get_embedding_client(feature: str = "embedding") -> Portkey:
    return Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_EMBEDDING_CONFIG_ID,
        metadata={"feature": feature, "_user": "rag-system"},
        **HTTP_CLIENT_OPTIONS,
    )


def get_chat_llm(
    tier: LlmTier, feature: str = "rag", temperature: float | None = None
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=ROUTED_MODEL,
        temperature=temperature,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=_CONFIG_IDS[tier],
            metadata={"feature": feature, "_user": "rag-system"},
        ),
        include_response_headers=True,
        **HTTP_CLIENT_OPTIONS,
    )


def extract_cache_status(response) -> str:
    headers = response.response_metadata.get("headers", {})
    for name, value in headers.items():
        if name.lower() == "x-portkey-cache-status":
            return str(value).upper()
    return "MISS"
