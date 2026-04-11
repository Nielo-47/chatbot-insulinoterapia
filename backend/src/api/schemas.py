from typing import List, Optional

from pydantic import BaseModel


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

