import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Literal, Optional

from backend.src.application.chat.conversation_service import ConversationService
from backend.src.application.chat.source_extractor import extract_sources
from backend.src.config.prompts import CRITIQUE_PROMPT, REFINEMENT_PROMPT, SYSTEM_PROMPT
from lightrag.prompt import PROMPTS

if TYPE_CHECKING:
    from backend.src.infrastructure.rag.client import RAGRuntime

logger = logging.getLogger(__name__)

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]


class QueryProcessor:
    def __init__(
        self,
        rag_runtime: "RAGRuntime",
        conversation_service: ConversationService,
        call_llm: Callable[..., Coroutine[Any, Any, str]],
    ):
        self.rag_runtime = rag_runtime
        self.conversation_service = conversation_service
        self.call_llm = call_llm

    async def _critique_response(
        self,
        original_query: str,
        response: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        critique_prompt = CRITIQUE_PROMPT.format(original_query=original_query, response=response)

        critique_text = await self.call_llm(
            prompt=critique_prompt,
            history_messages=history_messages,
            temperature=0.0,
            max_tokens=600,
        )

        try:
            critique_text = critique_text.strip()
            if critique_text.startswith("```"):
                critique_text = critique_text.split("```")[1]
                if critique_text.startswith("json"):
                    critique_text = critique_text[4:]

            return json.loads(critique_text.strip())
        except json.JSONDecodeError as e:
            logger.warning("Erro ao parsear crítica: %s", e)
            return {
                "is_safe": True,
                "is_accurate": True,
                "is_clear": True,
                "is_complete": True,
                "is_ethical": True,
                "issues": [],
                "suggestions": [],
                "needs_refinement": False,
            }

    @staticmethod
    def _build_refinement_query(original_query: str, previous_response: str, critique: Dict[str, Any]) -> str:
        issues_text = "\n- ".join(critique.get("issues", ["Nenhum problema identificado"]))
        suggestions_text = "\n- ".join(critique.get("suggestions", ["Mantenha a qualidade atual"]))

        return REFINEMENT_PROMPT.format(
            original_query=original_query,
            previous_response=previous_response,
            issues_text=issues_text,
            suggestions_text=suggestions_text,
        )

    async def query(
        self,
        query: str,
        user_id: int,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        logging.info("[DEBUG] query() called with mode=%s", mode)
        logging.info("[DEBUG] Original query: %s...", query[:100])

        if user_id is None:
            raise ValueError("user_id is required")

        session_label = session_id or str(uuid.uuid4())
        conversation_history = self.conversation_service.get_conversation(user_id)

        sources = []
        source_count = 0
        was_summarized = False

        logging.info("[Step 1/3] Generating initial response with RAG...")
        try:
            rag_data = self.rag_runtime.query_data(
                query=query,
                mode=mode,
                conversation_history=conversation_history,
                system_prompt=query_params.get("system_prompt", SYSTEM_PROMPT),
                max_total_tokens=query_params.get("max_total_tokens", 12000),
                top_k=query_params.get("top_k", 10),
            )

            sources, source_count = extract_sources(rag_data)
            logging.info("Extracted %d unique sources", source_count)

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
            logger.debug(
                "RAG returned %d data items \n%s",
                len(rag_data) if rag_data else 0,
                data_preview,
            )

            initial_response = await self.call_llm(
                prompt=query,
                system_prompt=query_params.get("system_prompt", SYSTEM_PROMPT.format(context=rag_data)),
                history_messages=conversation_history,
            )
            logging.info("[DEBUG] RAG query completed, response length: %d chars", len(initial_response))
        except Exception:
            logger.exception("RAG query failed")
            raise

        final_response = initial_response

        if initial_response != PROMPTS["fail_response"]:
            logging.info("[Step 2/3] Analyzing response quality...")

            history_for_critique = list(conversation_history)
            history_for_critique.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": initial_response},
                ]
            )
            critique = await self._critique_response(
                original_query=query,
                response=initial_response,
                history_messages=history_for_critique,
            )

            if not critique.get("needs_refinement", False):
                logging.info("Response approved - no refinement needed")
            else:
                if critique.get("issues"):
                    logging.warning("Issues found: %s", ", ".join(critique["issues"]))

                logging.info("[Step 3/3] Refining response...")
                refinement_query = self._build_refinement_query(query, initial_response, critique)
                final_response = await self.call_llm(refinement_query, history_messages=history_for_critique)
                logging.info("Response refined and approved")
        else:
            logging.info("No relevant context found - returning default message")

        self.conversation_service.add_message(user_id, "user", query)
        self.conversation_service.add_message(user_id, "assistant", final_response)
        was_summarized = self.conversation_service.consume_summarized(user_id)

        return {
            "response": final_response,
            "sources": sources,
            "source_count": source_count,
            "summarized": was_summarized,
            "session_id": session_label,
        }
