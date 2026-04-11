import os
from typing import Any, Dict, List, Literal, Optional
from backend.src.utils import call_openrouter
from backend.src.config import Config
from backend.src.application import ConversationService, build_conversation_service
from backend.src.application.chat.query_processor import QueryProcessor
from backend.src.infrastructure.rag.rag_client import RAGRuntime
from lightrag.prompt import PROMPTS


QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class Chatbot:
    def __init__(self, conversation_service: Optional[ConversationService] = None):
        if not Config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required. Set the OPENROUTER_API_KEY environment variable.")

        self.llm_api_key = Config.OPENROUTER_API_KEY
        self.llm_base_url = Config.OPENROUTER_BASE_URL
        self.embed_api_key = os.getenv("EMBEDDING_API_KEY", "")
        self.embed_base_url = f"{Config.EMBED_HOST}/v1"

        print(f"[DEBUG] Chatbot initialized")
        print(f"[DEBUG] LLM config: base_url={self.llm_base_url}")
        print(f"[DEBUG] Embed config: base_url={self.embed_base_url}")
        print(f"[DEBUG] Config.EMBED_HOST={Config.EMBED_HOST}")

        self.rag_runtime = RAGRuntime(self.embed_api_key, self.embed_base_url)
        self.conversation_service = conversation_service or build_conversation_service(summary_call_llm=self._call_llm)
        if getattr(self.conversation_service, "summary_call_llm", None) is None:
            self.conversation_service.summary_call_llm = self._call_llm
        self.query_processor = QueryProcessor(self.rag_runtime, self.conversation_service, self._call_llm)

        PROMPTS["fail_response"] = (
            "Desculpe, não encontrei informações sobre isso nos meus manuais de diabetes e insulinoterapia. "
            "Essa pergunta não está relacionada aos temas que posso ajudar (diabetes, insulina, glicemia). "
            "Se você tiver dúvidas sobre esses temas, ficarei feliz em ajudar!"
        )

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        return await call_openrouter(
            llm_api_key=self.llm_api_key,
            llm_base_url=self.llm_base_url,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def add_user_message(self, user_id: int, content: str):
        """Add user message to conversation."""
        self.conversation_service.add_message(user_id=user_id, role="user", content=content)

    def add_assistant_message(self, user_id: int, content: str):
        """Add assistant message to conversation."""
        self.conversation_service.add_message(user_id=user_id, role="assistant", content=content)

    def get_conversation(self, user_id: int) -> List[Dict]:
        return self.conversation_service.get_conversation(user_id)

    def reset_conversation(self, user_id: int):
        return self.conversation_service.reset_conversation(user_id)

    def summarize_session(self, user_id: int, max_messages: Optional[int] = None) -> str:
        """Summarize the conversation for the given user and replace the history with a single assistant message.

        Returns the summary text.
        """
        return self.conversation_service.summarize_session(user_id, max_messages=max_messages)

    async def initialize_rag(self):
        """Initialize LightRAG runtime."""
        await self.rag_runtime.initialize(self._call_llm)

    def query(
        self,
        query: str,
        user_id: int,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        return self.query_processor.query(
            query=query,
            user_id=user_id,
            mode=mode,
            session_id=session_id,
            **query_params,
        )
