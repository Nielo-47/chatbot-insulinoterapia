import asyncio
import logging
import uuid
from typing import Any, Callable, Coroutine, Dict, Optional

from backend.src.config import Config
from backend.src.utils.llm_utils import build_refinement_query, critique_response
from backend.src.utils.sources import extract_sources
from backend.src.application import ConversationService
from backend.src.infrastructure.rag.rag_client import QueryMode, RAGRuntime
from lightrag.prompt import PROMPTS


class QueryProcessor:
    def __init__(
        self,
        rag_runtime: RAGRuntime,
        conversation_service: ConversationService,
        call_llm: Callable[..., Coroutine[Any, Any, str]],
    ):
        self.rag_runtime = rag_runtime
        self.conversation_service = conversation_service
        self.call_llm = call_llm

    def query(
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
                system_prompt=query_params.get("system_prompt", Config.SYSTEM_PROMPT),
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
            print(f"[DEBUG] RAG returned {len(rag_data) if rag_data else 0} data items \n{data_preview}")

            initial_response = asyncio.run(
                self.call_llm(
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
            logging.info("[Step 2/3] Analyzing response quality...")

            history_for_critique = list(conversation_history)
            history_for_critique.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": initial_response},
                ]
            )
            critique = asyncio.run(
                critique_response(
                    call_llm=self.call_llm,
                    original_query=query,
                    response=initial_response,
                    history_messages=history_for_critique,
                )
            )

            if not critique.get("needs_refinement", False):
                logging.info("Response approved - no refinement needed")
            else:
                if critique.get("issues"):
                    logging.warning("Issues found: %s", ", ".join(critique["issues"]))

                logging.info("[Step 3/3] Refining response...")
                refinement_query = build_refinement_query(query, initial_response, critique)
                final_response = asyncio.run(self.call_llm(refinement_query, history_messages=history_for_critique))
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
