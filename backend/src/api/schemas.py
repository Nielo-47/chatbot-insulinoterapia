from typing import List, Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    path: str
    page: Optional[int] = None
    excerpt: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthenticatedUser(BaseModel):
    id: int
    username: str


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: List[SourceItem]
    summarized: bool
    session_id: str


class ConversationMessage(BaseModel):
    role: str
    content: str
    sources: List[SourceItem] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    messages: List[ConversationMessage]


class HealthResponse(BaseModel):
    status: str
    message: str
