import json
import logging
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional

from backend.src.application.contracts.chat import ConversationServiceContract, QueryMode, RAGRuntimeContract
from backend.src.config.prompts import CRITIQUE_PROMPT, REFINEMENT_PROMPT, SYSTEM_PROMPT
from langgraph.graph import END, StateGraph
from lightrag.prompt import PROMPTS
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryGraphState(BaseModel):
    query: str
    user_id: int
    mode: QueryMode
    session_id: str
    query_params: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    rag_data: Any = None
    sources: List[str] = Field(default_factory=list)
    source_count: int = 0
    initial_response: str = ""
    final_response: str = ""
    critique: Dict[str, Any] = Field(default_factory=dict)
    was_summarized: bool = False


class QueryProcessor:
    def __init__(
        self,
        rag_runtime: RAGRuntimeContract,
        conversation_service: ConversationServiceContract,
        call_llm: Callable[..., Coroutine[Any, Any, str]],
    ):
        self.rag_runtime = rag_runtime
        self.conversation_service = conversation_service
        self.call_llm = call_llm
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QueryGraphState)
        graph.add_node("load_history", self._node_load_history)
        graph.add_node("retrieve_rag", self._node_retrieve_rag)
        graph.add_node("generate_initial", self._node_generate_initial)
        graph.add_node("critique_response", self._node_critique_response)
        graph.add_node("refine_response", self._node_refine_response)
        graph.add_node("persist_messages", self._node_persist_messages)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "retrieve_rag")
        graph.add_edge("retrieve_rag", "generate_initial")
        graph.add_conditional_edges(
            "generate_initial",
            self._route_after_initial_response,
            {
                "critique": "critique_response",
                "persist": "persist_messages",
            },
        )
        graph.add_conditional_edges(
            "critique_response",
            self._route_after_critique,
            {
                "refine": "refine_response",
                "persist": "persist_messages",
            },
        )
        graph.add_edge("refine_response", "persist_messages")
        graph.add_edge("persist_messages", END)

        return graph.compile()

    def _node_load_history(self, state: QueryGraphState) -> Dict[str, Any]:
        raw_history = self.conversation_service.get_conversation(state.user_id)
        return {
            "conversation_history": self._normalize_history(raw_history),
        }

    @staticmethod
    def _normalize_history(history: Any) -> List[Dict[str, str]]:
        if not isinstance(history, list):
            return []

        cleaned: List[Dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _node_retrieve_rag(self, state: QueryGraphState) -> Dict[str, Any]:
        query_params = state.query_params
        rag_data = self.rag_runtime.query_data(
            query=state.query,
            mode=state.mode,
            conversation_history=state.conversation_history,
            system_prompt=query_params.get("system_prompt", SYSTEM_PROMPT),
            max_total_tokens=query_params.get("max_total_tokens", 12000),
            top_k=query_params.get("top_k", 10),
        )

        logger.debug("RAG returned %d data items", len(rag_data) if rag_data else 0)
        if isinstance(rag_data, dict) and "rag_data" in rag_data:
            return {
                "rag_data": rag_data.get("rag_data"),
                "sources": rag_data.get("sources", []),
                "source_count": rag_data.get("source_count", 0),
            }
        return {
            "rag_data": rag_data,
        }

    async def _node_generate_initial(self, state: QueryGraphState) -> Dict[str, Any]:
        query_params = state.query_params
        initial_response = await self.call_llm(
            prompt=state.query,
            system_prompt=query_params.get("system_prompt", SYSTEM_PROMPT.format(context=state.rag_data)),
            history_messages=state.conversation_history,
        )
        logging.info("[DEBUG] RAG query completed, response length: %d chars", len(initial_response))
        return {
            "initial_response": initial_response,
            "final_response": initial_response,
        }

    async def _node_critique_response(self, state: QueryGraphState) -> Dict[str, Any]:
        logging.info("[Step 2/3] Analyzing response quality...")
        history_for_critique = list(state.conversation_history)
        history_for_critique.extend(
            [
                {"role": "user", "content": state.query},
                {"role": "assistant", "content": state.initial_response},
            ]
        )
        critique = await self._critique_response(
            original_query=state.query,
            response=state.initial_response,
            history_messages=history_for_critique,
        )
        if not critique.get("needs_refinement", False):
            logging.info("Response approved - no refinement needed")
        elif critique.get("issues"):
            logging.warning("Issues found: %s", ", ".join(critique["issues"]))

        return {
            "critique": critique,
            "conversation_history": history_for_critique,
        }

    async def _node_refine_response(self, state: QueryGraphState) -> Dict[str, Any]:
        logging.info("[Step 3/3] Refining response...")
        refinement_query = self._build_refinement_query(
            state.query,
            state.initial_response,
            state.critique,
        )
        final_response = await self.call_llm(
            refinement_query,
            history_messages=state.conversation_history,
        )
        logging.info("Response refined and approved")
        return {
            "final_response": final_response,
        }

    def _node_persist_messages(self, state: QueryGraphState) -> Dict[str, Any]:
        self.conversation_service.add_message(state.user_id, "user", state.query)
        self.conversation_service.add_message(
            state.user_id,
            "assistant",
            state.final_response or state.initial_response,
            sources=state.sources,
        )
        return {
            "was_summarized": self.conversation_service.consume_summarized(state.user_id),
        }

    @staticmethod
    def _route_after_initial_response(state: QueryGraphState) -> str:
        if state.initial_response == PROMPTS["fail_response"]:
            logging.info("No relevant context found - returning default message")
            return "persist"
        return "critique"

    @staticmethod
    def _route_after_critique(state: QueryGraphState) -> str:
        critique = state.critique
        if critique.get("needs_refinement", False):
            return "refine"
        return "persist"

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

        logging.info("[Step 1/3] Generating initial response with RAG...")
        try:
            session_label = session_id or str(uuid.uuid4())
            final_state = await self._graph.ainvoke(
                QueryGraphState(
                    query=query,
                    user_id=user_id,
                    mode=mode,
                    session_id=session_label,
                    query_params=query_params,
                )
            )
        except Exception:
            logger.exception("RAG query failed")
            raise

        state_data = final_state.model_dump() if isinstance(final_state, QueryGraphState) else final_state

        return {
            "response": state_data.get("final_response", state_data.get("initial_response", "")),
            "sources": state_data.get("sources", []),
            "source_count": state_data.get("source_count", 0),
            "summarized": state_data.get("was_summarized", False),
            "session_id": state_data.get("session_id", session_label),
        }
