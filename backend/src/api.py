"""FastAPI Backend for Diabetes Chatbot - Exposes RAG functionality via REST API."""
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backend.src.auth import create_access_token, decode_access_token, verify_password
from backend.src.chatbot import Chatbot
from backend.src.db import initialize_database
from backend.src.repositories.users_repository import UsersRepository


def _parse_frontend_origins() -> List[str]:
    raw_origins = os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
    )
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot_instance: Optional[Chatbot] = None
auth_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthenticatedUser(BaseModel):
    id: int
    username: str


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme)) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload.get("sub", ""))
    except Exception:
        raise _unauthorized("Invalid or expired access token")

    user = UsersRepository().get_user_by_id(user_id)
    if user is None:
        raise _unauthorized("Invalid or expired access token")

    return AuthenticatedUser(id=user.id, username=user.username)


_DEFAULT_JWT_SECRET = "change-me"
_MIN_JWT_SECRET_LENGTH = 32


def _validate_jwt_secret() -> None:
    """Fail fast if JWT_SECRET_KEY is weak or left as the default value.

    In development (DEV=true) a warning is logged instead of raising, so
    local environments can start without a production-grade secret.
    """
    secret = Config.JWT_SECRET_KEY
    is_dev = os.getenv("DEV", "false").lower() in ("1", "true", "yes")
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
    global chatbot_instance
    _validate_jwt_secret()
    logger.info("Initializing chatbot...")
    initialize_database()
    logger.info("Database initialized successfully")
    chatbot_instance = Chatbot()
    await chatbot_instance.initialize_rag()
    logger.info("Chatbot initialized successfully")
    yield
    logger.info("Shutting down chatbot...")


app = FastAPI(
    title="Diabetes Chatbot API",
    description="Backend API for diabetes chatbot with RAG functionality",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for communication with UI container
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: List[str]
    source_count: int
    summarized: bool
    session_id: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    messages: List[ConversationMessage]


class HealthResponse(BaseModel):
    status: str
    message: str


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user = UsersRepository().get_user_by_username(request.username)
    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(access_token=access_token)


@app.get("/auth/me", response_model=AuthenticatedUser)
def read_current_user(current_user: AuthenticatedUser = Depends(get_current_user)):
    return current_user


@app.get("/user/conversations", response_model=ConversationHistoryResponse)
def get_user_conversations(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Get conversation history for the authenticated user."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        messages = chatbot_instance.get_conversation(current_user.id)
        logger.info(f"Retrieved {len(messages)} messages for user {current_user.id}")
        return ConversationHistoryResponse(
            messages=[ConversationMessage(role=msg["role"], content=msg["content"]) for msg in messages]
        )
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation history")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return HealthResponse(status="healthy", message="Chatbot API is running")


@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest, current_user: AuthenticatedUser = Depends(get_current_user)):
    """Query the chatbot with a question."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing query for user {current_user.id} / session {session_id}: {request.query[:50]}...")
        
        # Query the chatbot
        result = chatbot_instance.query(request.query, user_id=current_user.id, session_id=session_id)
        result["session_id"] = result.get("session_id", session_id)
        
        logger.info(
            f"Query completed for user {current_user.id} / session {session_id}: "
            f"response={len(result.get('response', ''))} chars, "
            f"sources={result.get('source_count', 0)}, "
            f"summarized={result.get('summarized', False)}"
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")


@app.delete("/user/conversations")
async def clear_user_conversations(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Clear conversation history for the authenticated user."""
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        cleared = chatbot_instance.reset_conversation(current_user.id)
        if cleared:
            logger.info(f"Cleared conversation for user {current_user.id}")
            return {"message": "Conversation cleared successfully"}
        return {"message": "No conversation found"}
    except Exception as e:
        logger.error(f"Error clearing conversation: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Error clearing conversation")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Diabetes Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
