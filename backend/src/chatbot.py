import os
import logging
import uuid
import numpy as np
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from typing import Any, Dict, List, Literal, Optional
from backend.src.helpers import build_refinement_query, call_openrouter, critique_response, extract_sources
from backend.src.config import Config
from backend.src.services import ConversationService, build_conversation_service
from lightrag.prompt import PROMPTS
import asyncio


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

        self.rag: Optional[LightRAG] = None
        self.conversation_service = conversation_service or build_conversation_service()
        # Track which users were just summarized
        self.sessions_summarized: set[int] = set()

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

    async def _critique_response(
        self, original_query: str, response: str, history_messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        return await critique_response(
            call_llm=self._call_llm,
            original_query=original_query,
            response=response,
            history_messages=history_messages,
        )

    def _build_refinement_query(self, original_query: str, previous_response: str, critique: Dict[str, Any]) -> str:
        return build_refinement_query(
            original_query=original_query,
            previous_response=previous_response,
            critique=critique,
        )

    # --- Conversation/session helpers ---
    def _init_session_if_missing(self, user_id: int):
        if user_id is None:
            return
        self.conversation_service.ensure_conversation(user_id)

    def _add_message_to_session(self, user_id: int, role: str, content: str):
        """Add a message to conversation history and trigger auto-summarization if needed."""
        if user_id is None:
            return
        self.conversation_service.add_message(user_id=user_id, role=role, content=content)

        # Trigger summarization if too many messages
        try:
            # Trigger summarization after assistant messages to avoid summarizing on partial turns.
            if role == "assistant" and self.conversation_service.count_messages(user_id) >= Config.SUMMARIZE_MAX_MESSAGES:
                self.summarize_session(user_id)
                # Mark that this user was just summarized
                self.sessions_summarized.add(user_id)
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to auto-summarize user %s: %s", user_id, e)

    def add_user_message(self, user_id: int, content: str):
        """Add user message to conversation."""
        self._add_message_to_session(user_id, "user", content)

    def add_assistant_message(self, user_id: int, content: str):
        """Add assistant message to conversation."""
        self._add_message_to_session(user_id, "assistant", content)

    def get_conversation(self, user_id: int) -> List[Dict]:
        if user_id is None:
            return []
        return self.conversation_service.get_conversation(user_id=user_id)

    def reset_conversation(self, user_id: int):
        if user_id is None:
            return False
        return self.conversation_service.reset_conversation(user_id=user_id)

    def summarize_session(self, user_id: int, max_messages: Optional[int] = None) -> str:
        """Summarize the conversation for the given user and replace the history with a single assistant message.

        Returns the summary text.
        """
        if user_id is None:
            return ""
        self._init_session_if_missing(user_id)
        msgs = self.conversation_service.get_conversation(user_id=user_id)
        if not msgs:
            return ""

        max_messages = max_messages or Config.SUMMARIZE_MAX_MESSAGES
        # Use the most recent messages up to max_messages to build the history
        recent = msgs[-max_messages:]
        history_lines = []
        for m in recent:
            role = m.get("role", "")
            content = str(m.get("content", "")).strip()
            if content:
                history_lines.append(f"{role.upper()}: {content}")

        history_text = "\n".join(history_lines)

        summary_prompt = Config.SUMMARY_PROMPT.format(history=history_text)

        try:
            # Use the LLM to create a short summary
            summary = asyncio.run(
                self._call_llm(
                    prompt=summary_prompt,
                    system_prompt=Config.SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=300,
                )
            )
            summary = summary.strip()
            if summary:
                self.conversation_service.replace_with_summary(user_id=user_id, summary=summary)
                logging.getLogger(__name__).info("User %s summarized into one message", user_id)
                return summary
        except Exception as e:
            logging.getLogger(__name__).warning("Error summarizing user %s: %s", user_id, e)

        return ""

    async def initialize_rag(self):
        """Initialize LightRAG with OpenRouter configuration."""
        print("[DEBUG] Initializing RAG...")

        async def openrouter_model_complete(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
            """Simple completion without refinement - used by RAG internally."""
            return await self._call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                max_tokens=500,
            )

        async def tei_embed_func(texts: List[str]) -> np.ndarray:
            """Generate embeddings using TEI OpenAI-compatible API."""
            print(f"[DEBUG] tei_embed_func called with {len(texts) if isinstance(texts, list) else 1} text(s)")
            print(f"[DEBUG] Using embed base_url: {self.embed_base_url}")

            try:
                client = OpenAI(api_key=self.embed_api_key, base_url=self.embed_base_url)

                if isinstance(texts, str):
                    texts = [texts]

                print(f"[DEBUG] Calling embeddings.create with model={Config.EMBED_MODEL}")
                response = client.embeddings.create(
                    model=Config.EMBED_MODEL,
                    input=texts,
                )
                print(f"[DEBUG] Embeddings received: {len(response.data)} embeddings")
                return np.array([item.embedding for item in response.data])
            except Exception as e:
                print(f"[ERROR] Embedding failed: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise

        self.rag = LightRAG(
            working_dir=Config.RAG_WORKING_DIR,
            llm_model_func=openrouter_model_complete,
            llm_model_name=Config.LLM_MODEL,
            enable_llm_cache=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=Config.EMBEDDING_DIM,
                max_token_size=Config.MAX_TOKENS,
                func=tei_embed_func,
            ),
        )

        print("[DEBUG] Initializing RAG storages...")
        await self.rag.initialize_storages()
        print("[DEBUG] RAG initialization complete")

    def query(
        self,
        query: str,
        user_id: int,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        """Query RAG with single-round refinement and return response with sources.

        If a session_id is provided, the method will use and update per-session conversation history
        so follow-up questions can be handled correctly.

        Returns:
            dict: {
                "response": str (the assistant's response),
                "sources": list[str] (unique source references),
                "source_count": int (number of sources),
                "summarized": bool (whether conversation was just summarized)
            }
        """
        logging.info("[DEBUG] query() called with mode=%s", mode)
        logging.info("[DEBUG] Original query: %s...", query[:100])

        if user_id is None:
            raise ValueError("user_id is required")

        session_label = session_id or str(uuid.uuid4())

        # Build conversation_history from session storage or provided params
        conversation_history = (
            self.get_conversation(user_id) if user_id is not None else query_params.get("conversation_history", [])
        )

        # Initialize sources tracking and summarization flag
        sources = []
        source_count = 0
        was_summarized = False

        # Step 1: Generate initial response using RAG
        logging.info("[Step 1/3] Generating initial response with RAG...")
        try:
            if self.rag is None:
                raise RuntimeError("RAG is not initialized")

            rag_data = self.rag.query_data(
                query,
                param=QueryParam(
                    mode=mode,
                    user_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT),
                    max_total_tokens=query_params.get("max_total_tokens", 12000),
                    top_k=query_params.get("top_k", 10),
                    enable_rerank=False,
                    conversation_history=conversation_history,
                ),
            )

            # Extract sources from rag_data
            sources, source_count = extract_sources(rag_data)
            logging.info(f"Extracted {source_count} unique sources")

            data_preview = ""
            if isinstance(rag_data, dict):
                data_value = rag_data.get("data")
                if isinstance(data_value, (list, tuple)):
                    data_preview = str(data_value)[:500]
                elif isinstance(data_value, str):
                    data_preview = data_value[:500]
                else:
                    data_preview = str(data_value)[:500]
            else:
                data_preview = str(rag_data)[:500] if rag_data is not None else ""
            print(f"[DEBUG] RAG returned {len(rag_data) if rag_data else 0} data items \n{data_preview}")

            initial_response = asyncio.run(
                self._call_llm(
                    prompt=query,
                    system_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT.format(context=rag_data)),
                    history_messages=conversation_history,
                )
            )
            logging.info("[DEBUG] RAG query completed, response length: %d chars", len(initial_response))
        except Exception as e:
            print(f"[ERROR] RAG query failed: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            raise

        final_response = initial_response

        if initial_response != PROMPTS["fail_response"]:
            # Step 2: Critique the response (no RAG, just LLM)
            logging.info("[Step 2/3] Analyzing response quality...")

            history_for_critique = list(conversation_history)
            history_for_critique.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": initial_response},
                ]
            )
            critique = asyncio.run(
                self._critique_response(query, initial_response, history_messages=history_for_critique)
            )

            # Check if refinement is needed
            if not critique.get("needs_refinement", False):
                logging.info("Response approved - no refinement needed")
            else:
                # Log issues found
                if critique.get("issues"):
                    logging.warning(f"Issues found: {', '.join(critique['issues'])}")

                # Step 3: Refine response using RAG with refinement query
                logging.info("[Step 3/3] Refining response...")
                refinement_query = self._build_refinement_query(query, initial_response, critique)
                final_response = asyncio.run(self._call_llm(refinement_query, history_messages=history_for_critique))
                logging.info("Response refined and approved")
        else:
            logging.info("No relevant context found - returning default message")

        # Persist turn after final response is known.
        self.add_user_message(user_id, query)
        self.add_assistant_message(user_id, final_response)

        # Check if conversation was just summarized
        was_summarized = user_id in self.sessions_summarized
        if was_summarized:
            self.sessions_summarized.discard(user_id)

        return {
            "response": final_response,
            "sources": sources,
            "source_count": source_count,
            "summarized": was_summarized,
            "session_id": session_label,
        }
