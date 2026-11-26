from typing import List, Optional

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    path: str
    page: Optional[int] = None
    content: Optional[str] = None


class AuthenticatedUser(BaseModel):
    id: str
    username: str


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)


class QueryResponse(BaseModel):
    response: str
    sources: List[SourceItem]
    follow_up_questions: List[str] = Field(default_factory=list)
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
