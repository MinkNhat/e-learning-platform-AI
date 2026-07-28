import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or default


class Settings:
    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = _env("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = _env("QDRANT_API_KEY")
    QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "elearning_rag")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = _env("PORTKEY_API_KEY")

    PORTKEY_PRIMARY_CONFIG_ID = _env("PORTKEY_PRIMARY_CONFIG_ID")
    PORTKEY_SECONDARY_CONFIG_ID = _env("PORTKEY_SECONDARY_CONFIG_ID")
    PORTKEY_GUARDRAIL_CONFIG_ID = _env("PORTKEY_GUARDRAIL_CONFIG_ID")
    PORTKEY_EMBEDDING_CONFIG_ID = _env("PORTKEY_EMBEDDING_CONFIG_ID")
    PORTKEY_HTTP_PROXY = _env("PORTKEY_HTTP_PROXY")

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = _env("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = _env("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = _env("LANGSMITH_PROJECT", "rag_application")
    LANGSMITH_ENDPOINT = _env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_application")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
