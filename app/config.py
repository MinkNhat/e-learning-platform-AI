import base64
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value is not None else default


def _env_base64(name: str) -> str | None:
    value = _env(name)
    return base64.b64decode(value).decode("utf-8") if value else None


class Settings:
    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = _env("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = _env("QDRANT_API_KEY")
    QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "elearning_rag")

    # --- MONGODB READ MODEL ---
    MONGODB_READ_URI = _env("MONGODB_READ_URI")
    MONGODB_DATABASE = _env("MONGODB_DATABASE")
    MONGODB_SERVER_SELECTION_TIMEOUT_MS = _env_int(
        "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
        10_000,
    )

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = _env("PORTKEY_API_KEY")

    PORTKEY_PRIMARY_CONFIG_ID = _env("PORTKEY_PRIMARY_CONFIG_ID")
    PORTKEY_SECONDARY_CONFIG_ID = _env("PORTKEY_SECONDARY_CONFIG_ID")
    PORTKEY_GUARDRAIL_CONFIG_ID = _env("PORTKEY_GUARDRAIL_CONFIG_ID")
    PORTKEY_EMBEDDING_CONFIG_ID = _env("PORTKEY_EMBEDDING_CONFIG_ID")
    PORTKEY_HTTP_PROXY = _env("PORTKEY_HTTP_PROXY")

    # --- RAG API AUTHENTICATION ---
    RAG_JWT_PUBLIC_KEY = _env_base64("RAG_JWT_PUBLIC_KEY")


# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_application")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
