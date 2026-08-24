"""FastAPI Backend for Diabetes Chatbot - Exposes RAG functionality via REST API.

Authentication is delegated to Supabase Auth. The frontend authenticates with
supabase-js and sends the resulting JWT access token as an ``Authorization:
Bearer`` header; the backend verifies the token's signature against the
project's public JWKS endpoint (SUPABASE_JWKS_URL) using the algorithm it
advertises (RS256 or ES256), plus audience and role, and
maps its ``sub`` claim (a UUID) to the local ``profiles`` row. Account deletion
calls the ``delete-account`` edge function, which holds the secret key.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api.schemas import (
    AuthenticatedUser,
    ConversationHistoryResponse,
    ConversationMessage,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from backend.src.api.dependencies import (
    build_auth_service,
    build_chatbot_service,
    get_auth_service,
    get_chatbot_service,
)
from backend.src.application.features.auth import AuthenticationService
from backend.src.application.features.chat.chatbot_service import ChatbotService
from backend.src.infrastructure.data.cache import init_semantic_cache
from backend.src.config.env import require
from backend.src.infrastructure.data import initialize_database
from backend.src.infrastructure.security import rate_limit
from backend.src.infrastructure.security.supabase import SupabaseTokenError, verify_access_token


def _parse_frontend_origins() -> List[str]:
    raw_origins = require("FRONTEND_ORIGINS")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


# Initialize
try:
    nest_asyncio.apply()
except ValueError as e:
    logging.getLogger(__name__).warning(
        "nest_asyncio could not patch the event loop (likely uvloop): %s. Continuing without nest_asyncio.",
        e,
    )
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def _unauthorized(detail: str = "Não autenticado") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def get_current_user(
    request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
) -> AuthenticatedUser:
    token = _extract_bearer_token(request)
    if not token:
        raise _unauthorized()

    try:
        claims = verify_access_token(token)
    except SupabaseTokenError:
        raise _unauthorized()

    sub = claims.get("sub")
    if not sub:
        raise _unauthorized()

    # The username is cosmetic (display only); email is the natural identifier
    # for Supabase email+password accounts.
    username = claims.get("email") or f"user_{sub[:8]}"
    try:
        principal = auth_service.resolve_principal_from_identity(sub, username)
    except ValueError:
        # The sub claim is not a valid UUID (should never happen for Supabase
        # tokens); treat it as unauthenticated rather than crashing.
        logger.warning("Rejected access token with non-UUID sub claim")
        raise _unauthorized()

    return AuthenticatedUser(id=principal.id, username=principal.username)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage chatbot lifecycle."""
    logger.info("Initializing chatbot...")
    initialize_database()
    logger.info("Database initialized successfully")
    init_semantic_cache()
    app.state.chatbot = await build_chatbot_service()
    app.state.auth_service = build_auth_service()
    logger.info("Chatbot initialized successfully")
    yield
    logger.info("Shutting down chatbot...")


# OpenAPI schema endpoints (/docs, /redoc, /openapi.json) are always disabled
# so the API surface is never exposed for reconnaissance.
app = FastAPI(
    title="LinaChat API",
    description="Backend API for LinaChat — diabetes and insulinotherapy assistant",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _raise_api_error(exc: Exception, user_message: str) -> None:
    if isinstance(exc, HTTPException):
        # Intentional control flow (e.g. the 429 query throttle); re-raise as-is.
        raise exc
    # Exception detail is logged server-side and never reflected in the response
    # body, so internal state/schema details cannot leak to clients (L1). The
    # client always receives the generic, user-safe message.
    logger.error("API error (%s): %s", type(exc).__name__, exc)
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=user_message) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=user_message) from exc
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=user_message) from exc


# Configure CORS for communication with UI container
# SameSite=Lax + the origin allowlist already block cross-site requests, so
# methods/headers are scoped to what the frontend actually uses (never "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/auth/me", response_model=AuthenticatedUser)
def read_current_user(current_user: AuthenticatedUser = Depends(get_current_user)):
    return current_user


@app.delete("/auth/me")
def delete_current_user(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    """Delete the current account.

    Cached PII (Redis conversation message cache, checkpointer thread state) is
    purged FIRST, before the account is revoked, so stale user data cannot
    outlive the account. The Auth user is then revoked via the ``delete-account``
    edge function (the caller's access token is forwarded; the secret key lives
    only in Supabase). The local profiles/conversations/messages rows are
    removed by the ``on_auth_user_deleted`` trigger. If the revocation fails,
    the account is kept (fail closed).
    """
    token = _extract_bearer_token(request)
    if not token:
        raise _unauthorized()

    # Purge cached PII for this user's conversation BEFORE the account is
    # revoked. The semantic cache is global (keyed by prompt hash), not
    # user-scoped, and is therefore not part of per-user purging.
    chatbot.purge_user_data(current_user.id)
    if not auth_service.delete_supabase_user(current_user.id, token):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Não foi possível excluir a conta no provedor de identidade",
        )
    return {"message": "Usuário excluído com sucesso"}


@app.get("/user/conversations", response_model=ConversationHistoryResponse)
def get_user_conversations(
    current_user: AuthenticatedUser = Depends(get_current_user),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    """Get conversation history for the authenticated user."""
    try:
        messages = chatbot.get_history(current_user.id)
        logger.info("Retrieved %d messages for user %s", len(messages), current_user.id)
        return ConversationHistoryResponse(
            messages=[
                ConversationMessage(
                    role=msg["role"],
                    content=msg["content"],
                    sources=msg.get("sources", []),
                )
                for msg in messages
            ]
        )
    except Exception as e:
        logger.error("Error retrieving conversation history for user %s: %s", current_user.id, type(e).__name__)
        logger.debug("Conversation history error detail for user %s: %s", current_user.id, e)
        _raise_api_error(e, "Erro ao recuperar histórico da conversa")


@app.get("/health", response_model=HealthResponse)
async def health_check(chatbot: ChatbotService = Depends(get_chatbot_service)):
    """Health check endpoint."""
    _ = chatbot
    return HealthResponse(status="healthy", message="Chatbot API is running")


@app.post("/query", response_model=QueryResponse)
async def query_chatbot(
    request: QueryRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    """Query the chatbot with a question."""
    session_id = request.session_id or str(uuid.uuid4())
    try:
        is_allowed, _ = rate_limit.check_query_rate_limit(current_user.id)
        if not is_allowed:
            remaining = rate_limit.get_query_rate_limit_remaining_seconds(current_user.id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Limite de consultas excedido. Tente novamente em {remaining} segundos.",
            )

        logger.info(
            "Processing query for user %s / session %s (query length %d)",
            current_user.id,
            session_id,
            len(request.query),
        )

        # Query the chatbot
        result = await chatbot.chat(request.query, user_id=current_user.id, session_id=session_id)
        result["session_id"] = result.get("session_id", session_id)

        logger.info(
            "Query completed for user %s / session %s: "
            "response=%d chars, sources=%d, summarized=%s",
            current_user.id,
            session_id,
            len(result.get("response", "")),
            len(result.get("sources", [])),
            result.get("summarized", False),
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(
            "Error processing query for user %s / session %s: %s",
            current_user.id,
            session_id,
            type(e).__name__,
        )
        logger.debug("Query error detail for user %s / session %s: %s", current_user.id, session_id, e)
        _raise_api_error(e, "Erro ao processar consulta")


@app.delete("/user/conversations")
async def clear_user_conversations(
    current_user: AuthenticatedUser = Depends(get_current_user),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    """Clear conversation history for the authenticated user."""
    try:
        cleared = chatbot.end_session(current_user.id)
        if cleared:
            logger.info("Cleared conversation for user %s", current_user.id)
            return {"message": "Conversa limpa com sucesso"}
        return {"message": "Nenhuma conversa encontrada"}
    except Exception as e:
        logger.error("Error clearing conversation for user %s: %s", current_user.id, type(e).__name__)
        logger.debug("Clear conversation error detail for user %s: %s", current_user.id, e)
        _raise_api_error(e, "Erro ao limpar conversa")


@app.get("/")
async def root():
    """Root endpoint.

    Kept unauthenticated (harmless banner), but deliberately omits the version
    so the API surface is not fingerprinted (L2). /health must stay
    unauthenticated too because the container healthcheck probes it. All
    other endpoints require a valid Supabase Bearer token.
    """
    return {
        "message": "LinaChat API",
        "docs": None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
