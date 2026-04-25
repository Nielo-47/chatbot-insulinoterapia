from typing import Annotated, TypedDict
from pydantic import UUID4, BaseModel, Field

from langchain_core.messages import AnyMessage, add_messages
from langchain_core.documents import Document


class CritiqueSchema(BaseModel):
    is_safe: bool = Field(description="Is the response free from harmful content?")
    is_accurate: bool = Field(description="Is the response factually correct?")
    feedback: str = Field(description="Feedback for improving the response")
    needs_rewrite: bool = Field(description="True if the response must be regenerated")
    issues: list[str] = Field(default_factory=list, description="List of identified issues with the response")
    suggestions: list[str] = Field(default_factory=list, description="List of suggestions for improving")
    needs_refinement: bool = Field(False, description="True if the response needs refinement")


class QueryGraphState(TypedDict):
    query: str
    conversation_id: UUID4
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    sources: list[Document]
    initial_response: str
    final_response: str
    critique: CritiqueSchema
    was_summarized: bool
