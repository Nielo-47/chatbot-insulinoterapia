import os

from fastapi import HTTPException, Request, status

from backend.src.application.features.auth import AuthenticationService, build_authentication_service
from backend.src.application.features.chat.chatbot_service import ChatbotService
from backend.src.application.features.chat.conversation_service import ConversationService
from backend.src.application.features.chat.query_processor import QueryProcessor
from backend.src.config.infrastructure import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from backend.src.config.rag import EMBED_HOST
from backend.src.infrastructure.llm.client import LLMClient
from backend.src.infrastructure.rag.factory import RAGFactory
from backend.src.infrastructure.repositories.conversations_repository import ConversationsRepository
from backend.src.infrastructure.repositories.messages_repository import MessagesRepository
from backend.src.infrastructure.repositories.users_repository import UsersRepository
from backend.src.infrastructure.security.password import verify_password
from backend.src.infrastructure.security.token import create_access_token, decode_access_token


async def build_chatbot_service() -> ChatbotService:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required. Set the OPENROUTER_API_KEY environment variable.")

    llm_client = LLMClient(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    rag_runtime = RAGFactory.create(
        embed_host=EMBED_HOST,
        embed_api_key=os.getenv("EMBEDDING_API_KEY", ""),
    )
    await rag_runtime.initialize(llm_client.complete)

    conversation_service = ConversationService(
        users_repository=UsersRepository(),
        conversations_repository=ConversationsRepository(),
        messages_repository=MessagesRepository(),
        summary_call_llm=llm_client.complete,
    )
    query_processor = QueryProcessor(rag_runtime, conversation_service, llm_client.complete)

    return ChatbotService(conversation_service=conversation_service, query_processor=query_processor)


def build_auth_service() -> AuthenticationService:
    return build_authentication_service(
        users_repository=UsersRepository(),
        verify_password=verify_password,
        create_access_token=create_access_token,
        decode_access_token=decode_access_token,
    )


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
