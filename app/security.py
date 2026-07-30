import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import settings


class RetrievalScopeClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_course_ids: list[str] = Field(default_factory=list, max_length=100)
    course_id: str | None = Field(default=None, max_length=128)
    module_id: str | None = Field(default=None, max_length=128)
    lesson_id: str | None = Field(default=None, max_length=128)


class RagPrincipal(BaseModel):
    sub: str = Field(min_length=1, max_length=128)
    permission: str
    conversation_id: str = Field(min_length=1, max_length=256)
    retrieval_scope: RetrievalScopeClaim


bearer_scheme = HTTPBearer(auto_error=False)


def require_rag_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> RagPrincipal:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing RAG access token.",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.RAG_JWT_PUBLIC_KEY,
            algorithms=["RS256"],
            issuer="learning-platform-be",
            audience="e-learning-rag",
            options={"require": ["exp", "iat", "sub", "jti"]},
        )
        principal = RagPrincipal.model_validate(payload)
    except (InvalidTokenError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired RAG access token.",
        )

    if principal.permission != "rag:query":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG query permission is required.",
        )
    return principal
