import os

from fastapi import HTTPException, Request, status

from backend.src.application.auth import AuthenticationService, build_authentication_service
from backend.src.application.chat.chatbot_service import ChatbotService
from backend.src.application.chat.conversation_service import build_conversation_service
from backend.src.application.chat.query_processor import QueryProcessor
from backend.src.config import Config
from backend.src.infrastructure.llm.client import LLMClient
from backend.src.infrastructure.rag.factory import RAGFactory


async def build_chatbot_service() -> ChatbotService:
    if not Config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required. Set the OPENROUTER_API_KEY environment variable.")

    llm_client = LLMClient(api_key=Config.OPENROUTER_API_KEY, base_url=Config.OPENROUTER_BASE_URL)

    rag_runtime = RAGFactory.create(
        llm_client=llm_client,
        embed_host=Config.EMBED_HOST,
        embed_api_key=os.getenv("EMBEDDING_API_KEY", ""),
    )
    await rag_runtime.initialize(llm_client.complete)

    conversation_service = build_conversation_service(summary_call_llm=llm_client.complete)
    query_processor = QueryProcessor(rag_runtime, conversation_service, llm_client.complete)

    return ChatbotService(conversation_service=conversation_service, query_processor=query_processor)


def build_auth_service() -> AuthenticationService:
    return build_authentication_service()


def get_chatbot_service(request: Request) -> ChatbotService:
    chatbot = getattr(request.app.state, "chatbot", None)
    if chatbot is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chatbot not initialized")
    return chatbot


def get_auth_service(request: Request) -> AuthenticationService:
    auth_service = getattr(request.app.state, "auth_service", None)
    if auth_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication not initialized")
    return auth_service
