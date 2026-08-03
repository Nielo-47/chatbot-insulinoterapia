"""FastAPI Backend for Diabetes Chatbot - Exposes RAG functionality via REST API."""

import ipaddress
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List

import nest_asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

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
from backend.src.application.features.chat.chatbot_service import ChatbotService
from backend.src.infrastructure.data.cache import init_semantic_cache
from backend.src.config.security import (
    JWT_SECRET_KEY,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_HTTPONLY,
    AUTH_COOKIE_SECURE,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_PATH,
    AUTH_COOKIE_DOMAIN,
    TRUSTED_PROXY_IPS,
)
from backend.src.config.infrastructure import CHAT_CACHE_REDIS_URL, DOCS_ENABLED
from backend.src.config.env import require
from backend.src.infrastructure.data import initialize_database
from backend.src.infrastructure.security import rate_limit


def _normalize_ip(ip: str) -> str:
    """Strip the IPv4-mapped IPv6 prefix so both forms compare equal."""
    return ip[7:] if ip.lower().startswith("::ffff:") else ip


def _is_trusted_proxy(peer: str) -> bool:
    """Return True if the direct peer is a configured reverse proxy (IP/CIDR)."""
    peer = _normalize_ip(peer.strip())
    for entry in TRUSTED_PROXY_IPS:
        normalized = _normalize_ip(entry.strip())
        if normalized == peer:
            return True
        try:
            if ipaddress.ip_address(peer) in ipaddress.ip_network(normalized, strict=False):
                return True
        except ValueError:
            continue
    return False


# Rate limiter keyed on the real client IP. Forwarded headers (X-Real-IP,
# X-Forwarded-For) are honored only when the request's direct peer is a trusted
# reverse proxy (TRUSTED_PROXY_IPS); nginx overwrites them with $remote_addr,
# so an end client cannot spoof its identity through the proxy. Otherwise the
# direct peer address is used and any client-supplied headers are ignored.
def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    if _is_trusted_proxy(peer):
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            first_hop = forwarded.split(",")[0].strip()
            if first_hop:
                return first_hop
    return peer


limiter = Limiter(key_func=_client_ip, storage_uri=CHAT_CACHE_REDIS_URL)


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

def _unauthorized(detail: str = "Nao autenticado") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    return None


def _request_token(request: Request) -> str | None:
    """Read the JWT from the httpOnly session cookie first, then the Authorization header.

    The cookie is the primary transport (keeps the token out of JS-accessible
    storage); the Authorization header remains supported for API clients/tests.
    """
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token
    return _bearer_token(request)


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=AUTH_COOKIE_HTTPONLY,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        path=AUTH_COOKIE_PATH,
        domain=AUTH_COOKIE_DOMAIN or None,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path=AUTH_COOKIE_PATH,
        domain=AUTH_COOKIE_DOMAIN or None,
        secure=AUTH_COOKIE_SECURE,
        httponly=AUTH_COOKIE_HTTPONLY,
        samesite=AUTH_COOKIE_SAMESITE,
    )


def get_current_user(
    request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
) -> AuthenticatedUser:
    token = _request_token(request)
    if token is None:
        raise _unauthorized()

    principal = auth_service.resolve_principal_from_token(token)
    if principal is None:
        raise _unauthorized("Token de acesso invalido ou expirado")

    return AuthenticatedUser(id=principal.id, username=principal.username)


_DEFAULT_JWT_SECRET = "change-me"
_MIN_JWT_SECRET_LENGTH = 32


def _validate_jwt_secret() -> None:
    """Fail fast if JWT_SECRET_KEY is weak or left as the default value.

    A weak or default secret is never acceptable, including in development.
    """
    secret = JWT_SECRET_KEY
    weak = secret == _DEFAULT_JWT_SECRET or len(secret) < _MIN_JWT_SECRET_LENGTH
    if weak:
        raise RuntimeError(
            f"JWT_SECRET_KEY must be set to a strong secret (at least {_MIN_JWT_SECRET_LENGTH} characters)."
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


# OpenAPI schema endpoints (/docs, /redoc, /openapi.json) are disabled unless
# DOCS_ENABLED=true so the API surface is not exposed for reconnaissance.
app = FastAPI(
    title="Diabetes Chatbot API",
    description="Backend API for diabetes chatbot with RAG functionality",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
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
# SameSite=Lax + the origin allowlist already block cross-site requests, so
# methods/headers are scoped to what the frontend actually uses (never "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")  # Rate limit: 5 login attempts per minute per IP
def login(
    request: Request,
    login_request: LoginRequest,
    response: Response,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    # Get real client IP for rate limiting / lockout tracking
    client_ip = _client_ip(request)

    principal = auth_service.authenticate_credentials(
        login_request.username,
        login_request.password,
        client_ip=client_ip,
    )
    if principal is None:
        # Uniform generic response for every login failure (bad credentials,
        # locked account, rate limited) to avoid account enumeration and
        # status-code-based fingerprinting.
        raise _unauthorized("Credenciais invalidas")

    access_token = auth_service.issue_access_token(principal)
    # The JWT is delivered only via an httpOnly, Secure, SameSite=Lax cookie.
    # It is still returned in the body so non-browser clients (API tests,
    # curl) can use the Bearer scheme, but the browser frontend never reads
    # it into localStorage. SameSite=Lax + the CORS origin allowlist block
    # cross-site state-changing requests (CSRF).
    _set_auth_cookie(response, access_token)
    return TokenResponse(access_token=access_token)


@app.get("/auth/me", response_model=AuthenticatedUser)
def read_current_user(current_user: AuthenticatedUser = Depends(get_current_user)):
    return current_user


@app.post("/auth/logout")
def logout(
    request: Request,
    response: Response,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Logout by blacklisting the current token and clearing the session cookie.

    Does not require a valid token so an expired/invalid session can still be
    cleared client-side.
    """
    token = _request_token(request)
    if token:
        auth_service.logout_token(token)
    _clear_auth_cookie(response)
    return {"message": "Desconectado com sucesso"}


@app.delete("/auth/me")
def delete_current_user(
    response: Response,
    current_user: AuthenticatedUser = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
    chatbot: ChatbotService = Depends(get_chatbot_service),
):
    # Purge cached PII (Redis conversation message cache) for this user's
    # conversation BEFORE the DB row is removed, so stale user data cannot
    # outlive the account. The semantic cache is global (keyed by prompt hash),
    # not user-scoped, and is therefore not part of per-user purging.
    chatbot.purge_user_data(current_user.id)
    deleted = auth_service.delete_user(current_user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario nao encontrado")
    _clear_auth_cookie(response)
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
        is_allowed, _ = rate_limit.check_query_rate_limit(current_user.id)
        if not is_allowed:
            remaining = rate_limit.get_query_rate_limit_remaining_seconds(current_user.id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Limite de consultas excedido. Tente novamente em {remaining} segundos.",
            )

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
    return {
        "message": "Diabetes Chatbot API",
        "version": "1.0.0",
        "docs": "/docs" if DOCS_ENABLED else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
