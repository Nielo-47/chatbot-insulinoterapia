"""FastAPI Backend for Diabetes Chatbot - Exposes RAG functionality via REST API."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import List

import nest_asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.src.api.schemas import (
    AuthenticatedUser,
    ConversationHistoryResponse,
    ConversationMessage,
    HealthResponse,
    LoginRequest,
    QueryRequest,
    QueryResponse,
    TokenResponse,
)
from backend.src.api.dependencies import (
    build_auth_service,
    build_chatbot_service,
    get_auth_service,
    get_chatbot_service,
)
from backend.src.application.features.auth import AuthenticationService
from backend.src.application.features.auth.auth_service import AccountLockedException, RateLimitExceededException
from backend.src.application.features.chat.chatbot_service import ChatbotService
from backend.src.infrastructure.data.cache import init_semantic_cache
from backend.src.config.security import JWT_SECRET_KEY
from backend.src.config.env import require
from backend.src.infrastructure.data import initialize_database


# Rate limiter using client IP
limiter = Limiter(key_func=get_remote_address)


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

auth_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Nao autenticado") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
    auth_service: AuthenticationService = Depends(get_auth_service),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    principal = auth_service.resolve_principal_from_token(credentials.credentials)
    if principal is None:
        raise _unauthorized("Token de acesso invalido ou expirado")

    return AuthenticatedUser(id=principal.id, username=principal.username)


_DEFAULT_JWT_SECRET = "change-me"
_MIN_JWT_SECRET_LENGTH = 32


def _validate_jwt_secret() -> None:
    """Fail fast if JWT_SECRET_KEY is weak or left as the default value.

    In development (DEV=true) a warning is logged instead of raising, so
    local environments can start without a production-grade secret.
    """
    secret = JWT_SECRET_KEY
    is_dev = require("DEV").lower() in ("1", "true", "yes")
    weak = secret == _DEFAULT_JWT_SECRET or len(secret) < _MIN_JWT_SECRET_LENGTH
    if weak:
        if is_dev:
            logger.warning(
                "JWT_SECRET_KEY is using a weak/default value. "
                "Set a strong secret (≥%d chars) before deploying to production.",
                _MIN_JWT_SECRET_LENGTH,
            )
        else:
            raise RuntimeError(
                f"JWT_SECRET_KEY must be set to a strong secret (at least {_MIN_JWT_SECRET_LENGTH} characters). "
                "Set DEV=true to bypass this check in local development."
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage chatbot lifecycle."""
    _validate_jwt_secret()
    logger.info("Initializing chatbot...")
    initialize_database()
    logger.info("Database initialized successfully")
    init_semantic_cache()
    logger.info("Semantic cache initialized")
    app.state.chatbot = await build_chatbot_service()
    app.state.auth_service = build_auth_service()
    logger.info("Chatbot initialized successfully")
    yield
    logger.info("Shutting down chatbot...")


app = FastAPI(
    title="Diabetes Chatbot API",
    description="Backend API for diabetes chatbot with RAG functionality",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _raise_api_error(exc: Exception, user_message: str) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=user_message) from exc
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=user_message) from exc


# Configure CORS for communication with UI container
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")  # Rate limit: 5 login attempts per minute per IP
def login(request: Request, login_request: LoginRequest, auth_service: AuthenticationService = Depends(get_auth_service)):
    # Get client IP for rate limiting
    client_ip = request.client.host if request.client else None
    
    try:
        principal = auth_service.authenticate_credentials(
            login_request.username, 
            login_request.password,
            client_ip=client_ip,
        )
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nome de usuario ou senha invalidos")

        access_token = auth_service.issue_access_token(principal)
        return TokenResponse(access_token=access_token)
    
    except AccountLockedException as e:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is locked. Try again in {e.remaining_seconds} seconds.",
        )
    except RateLimitExceededException as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Try again in {e.remaining_seconds} seconds.",
        )


@app.get("/auth/me", response_model=AuthenticatedUser)
def read_current_user(current_user: AuthenticatedUser = Depends(get_current_user)):
    return current_user


@app.post("/auth/logout")
def logout(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Logout the current user by blacklisting their token."""
    # Get the token from the Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        auth_service.logout_token(token)
    
    return {"message": "Desconectado com sucesso"}


@app.delete("/auth/me")
def delete_current_user(
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    deleted = auth_service.delete_user(current_user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario nao encontrado")
    return {"message": "Usuario excluido com sucesso"}


@app.get("/user/conversations", response_model=ConversationHistoryResponse)
def get_user_conversations(
    current_user: AuthenticatedUser = Depends(get_current_user),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    """Get conversation history for the authenticated user."""
    try:
        messages = chatbot.get_history(current_user.id)
        logger.info(f"Retrieved {len(messages)} messages for user {current_user.id}")
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
        logger.error(f"Error retrieving conversation history: {type(e).__name__}: {e}")
        _raise_api_error(e, "Erro ao recuperar historico da conversa")


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
    try:
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(f"Processing query for user {current_user.id} / session {session_id}: {request.query[:50]}...")

        # Query the chatbot
        result = await chatbot.chat(request.query, user_id=current_user.id, session_id=session_id)
        result["session_id"] = result.get("session_id", session_id)

        logger.info(
            f"Query completed for user {current_user.id} / session {session_id}: "
            f"response={len(result.get('response', ''))} chars, "
            f"sources={len(result.get('sources', []))}, "
            f"summarized={result.get('summarized', False)}"
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Error processing query: {type(e).__name__}: {e}")
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
            logger.info(f"Cleared conversation for user {current_user.id}")
            return {"message": "Conversa limpa com sucesso"}
        return {"message": "No conversation found"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {type(e).__name__}: {e}")
        _raise_api_error(e, "Erro ao limpar conversa")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Diabetes Chatbot API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
