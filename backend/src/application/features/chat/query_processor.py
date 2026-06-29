import logging
import uuid
from typing import Any, AsyncGenerator, Callable, Coroutine, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from backend.src.application.contracts.chat import ConversationServiceContract, QueryMode, RAGRuntimeContract
from backend.src.config.conversation import SUMMARIZE_MAX_MESSAGES
from backend.src.config.prompts import SYSTEM_PROMPT
from backend.src.application.features.chat.critique import CritiqueService
from backend.src.application.features.chat.summarizer import SummarizationService
from backend.src.infrastructure.data.db_client import create_postgres_checkpointer
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
    summary: str = ""
    rag_data: Any = None
    sources: List[dict] = Field(default_factory=list)
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
        call_llm_stream: Optional[Callable[..., AsyncGenerator[str, None]]] = None,
    ):
        self._rag_runtime = rag_runtime
        self._conversation_service = conversation_service
        self._call_llm = call_llm
        self._call_llm_stream = call_llm_stream or self._noop_stream
        self._critique_svc = CritiqueService(call_llm)
        self._summarizer = SummarizationService(conversation_service, call_llm)
        self._checkpointer = create_postgres_checkpointer()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        g = StateGraph(QueryGraphState)

        for name, fn in [
            ("load_history", self._node_load_history),
            ("retrieve_rag", self._node_retrieve_rag),
            ("generate_initial", self._node_generate_initial),
            ("critique_response", self._node_critique_response),
            ("refine_response", self._node_refine_response),
            ("persist_messages", self._node_persist_messages),
            ("summarize_conversation", self._node_summarize),
        ]:
            g.add_node(name, fn)

        g.set_entry_point("load_history")
        g.add_edge("load_history", "retrieve_rag")
        g.add_edge("retrieve_rag", "generate_initial")
        g.add_edge("refine_response", "persist_messages")
        g.add_edge("summarize_conversation", END)

        g.add_conditional_edges(
            "generate_initial",
            self._route_after_initial_response,
            {"critique": "critique_response", "persist": "persist_messages"},
        )
        g.add_conditional_edges(
            "critique_response",
            self._route_after_critique,
            {"refine": "refine_response", "persist": "persist_messages"},
        )
        g.add_conditional_edges(
            "persist_messages",
            self._should_summarize,
            {"summarize": "summarize_conversation", "continue": END},
        )

        return g.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------ #
    # Nodes                                                                #
    # ------------------------------------------------------------------ #

    def _node_load_history(self, state: QueryGraphState) -> Dict[str, Any]:
        history = _normalize_history(self._conversation_service.get_conversation(state.user_id))
        stored_summary = self._conversation_service.get_summary(state.user_id)

        if stored_summary:
            history.insert(
                0,
                {
                    "role": "system",
                    "content": f"[Summary of previous conversation]: {stored_summary}",
                },
            )

        return {"conversation_history": history, "summary": stored_summary or ""}

    async def _node_retrieve_rag(self, state: QueryGraphState) -> Dict[str, Any]:
        params = state.query_params
        result = await self._rag_runtime.query_data(
            query=state.query,
            mode=state.mode,
            conversation_history=state.conversation_history,
            system_prompt=params.get("system_prompt", SYSTEM_PROMPT),
            max_total_tokens=params.get("max_total_tokens", 12_000),
            top_k=params.get("top_k", 10),
        )

        if isinstance(result, dict) and "rag_data" in result:
            return {
                "rag_data": result["rag_data"],
                "sources": result.get("sources", []),
            }
        return {"rag_data": result}

    async def _node_generate_initial(self, state: QueryGraphState) -> Dict[str, Any]:
        params = state.query_params
        response = await self._call_llm(
            prompt=state.query,
            system_prompt=params.get("system_prompt", SYSTEM_PROMPT.format(context=state.rag_data)),
            history_messages=state.conversation_history,
        )
        logger.debug("Initial response generated (%d chars)", len(response))
        return {"initial_response": response, "final_response": response}

    async def _node_critique_response(self, state: QueryGraphState) -> Dict[str, Any]:
        logger.info("Critiquing response...")
        extended_history = [
            *state.conversation_history,
            {"role": "user", "content": state.query},
            {"role": "assistant", "content": state.initial_response},
        ]
        critique = await self._critique_svc.critique_response(
            original_query=state.query,
            response=state.initial_response,
            history_messages=extended_history,
        )
        if critique.get("issues"):
            logger.warning("Critique issues: %s", ", ".join(critique["issues"]))

        return {"critique": critique, "conversation_history": extended_history}

    async def _node_refine_response(self, state: QueryGraphState) -> Dict[str, Any]:
        logger.info("Refining response...")
        refinement_query = self._critique_svc.build_refinement_query(
            state.query, state.initial_response, state.critique
        )
        refined = await self._call_llm(refinement_query, history_messages=state.conversation_history)
        return {"final_response": refined}

    def _node_persist_messages(self, state: QueryGraphState) -> Dict[str, Any]:
        self._conversation_service.add_message(state.user_id, "user", state.query)
        self._conversation_service.add_message(
            state.user_id,
            "assistant",
            state.final_response or state.initial_response,
            sources=state.sources,
        )
        return {}  # was_summarized is set by the summarize node, not here

    async def _node_summarize(self, state: QueryGraphState) -> Dict[str, Any]:
        logger.info("Summarizing conversation...")
        result = await self._summarizer.summarize_and_trim(state.user_id, state.conversation_history)
        return result or {"was_summarized": False}

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _route_after_initial_response(state: QueryGraphState) -> str:
        if state.initial_response == PROMPTS["fail_response"]:
            logger.info("No relevant context — skipping critique")
            return "persist"
        return "critique"

    @staticmethod
    def _route_after_critique(state: QueryGraphState) -> str:
        return "refine" if state.critique.get("needs_refinement") else "persist"

    @staticmethod
    def _should_summarize(state: QueryGraphState) -> str:
        # +2 accounts for the user/assistant messages just persisted
        if len(state.conversation_history) + 2 >= SUMMARIZE_MAX_MESSAGES:
            return "summarize"
        return "continue"

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def query(
        self,
        query: str,
        user_id: int,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> Dict[str, Any]:
        if user_id is None:
            raise ValueError("user_id e obrigatorio")

        logger.info("Processing query (mode=%s): %.100s", mode, query)
        session_label = session_id or str(uuid.uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": f"user_{user_id}"}}

        try:
            final_state = await self._graph.ainvoke(
                QueryGraphState(
                    query=query,
                    user_id=user_id,
                    mode=mode,
                    session_id=session_label,
                    query_params=query_params,
                ),
                config=config,
            )
        except Exception:
            logger.exception("Graph execution failed")
            raise

        data = final_state.model_dump() if isinstance(final_state, QueryGraphState) else final_state
        return {
            "response": data.get("final_response") or data.get("initial_response", ""),
            "sources": data.get("sources", []),
            "summarized": data.get("was_summarized", False),
            "session_id": data.get("session_id", session_label),
        }

    # ------------------------------------------------------------------ #
    # Streaming public API                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _noop_stream(*args, **kwargs) -> AsyncGenerator[str, None]:
        if False:
            yield  # pragma: no cover

    async def query_stream(
        self,
        query: str,
        user_id: int,
        mode: QueryMode = "hybrid",
        session_id: Optional[str] = None,
        **query_params,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if user_id is None:
            raise ValueError("user_id e obrigatorio")

        logger.info("Streaming query (mode=%s): %.100s", mode, query)
        session_label = session_id or str(uuid.uuid4())

        # -- load history --
        yield {"type": "stage", "stage": "loading_history"}
        history = _normalize_history(self._conversation_service.get_conversation(user_id))
        stored_summary = self._conversation_service.get_summary(user_id)
        if stored_summary:
            history.insert(0, {"role": "system", "content": f"[Summary of previous conversation]: {stored_summary}"})
        rag_history = list(history)

        # -- retrieve RAG --
        yield {"type": "stage", "stage": "retrieving"}
        try:
            params = query_params
            result = await self._rag_runtime.query_data(
                query=query,
                mode=mode,
                conversation_history=rag_history,
                system_prompt=params.get("system_prompt", SYSTEM_PROMPT),
                max_total_tokens=params.get("max_total_tokens", 12_000),
                top_k=params.get("top_k", 10),
            )
        except Exception:
            logger.exception("RAG retrieval failed in stream")
            raise

        if isinstance(result, dict) and "rag_data" in result:
            rag_data = result["rag_data"]
            sources = result.get("sources", [])
        else:
            rag_data = result
            sources = []

        # -- generate initial (streamed) --
        yield {"type": "stage", "stage": "generating"}
        full_response_parts: List[str] = []
        system_prompt = params.get("system_prompt", SYSTEM_PROMPT.format(context=rag_data))
        async for token in self._call_llm_stream(
            prompt=query,
            system_prompt=system_prompt,
            history_messages=history,
        ):
            full_response_parts.append(token)
            yield {"type": "token", "token": token}

        initial_response = "".join(full_response_parts)
        logger.debug("Initial response generated via stream (%d chars)", len(initial_response))

        # -- route: skip critique if fail response --
        if initial_response == PROMPTS["fail_response"]:
            logger.info("No relevant context — skipping critique")
            final_response = initial_response
        else:
            # -- critique --
            yield {"type": "stage", "stage": "critiquing"}
            extended_history = [
                *history,
                {"role": "user", "content": query},
                {"role": "assistant", "content": initial_response},
            ]
            critique = await self._critique_svc.critique_response(
                original_query=query,
                response=initial_response,
                history_messages=extended_history,
            )
            if critique.get("issues"):
                logger.warning("Critique issues: %s", ", ".join(critique["issues"]))

            if critique.get("needs_refinement"):
                # -- refine --
                yield {"type": "stage", "stage": "refining"}
                refinement_query = self._critique_svc.build_refinement_query(query, initial_response, critique)
                refined = await self._call_llm(refinement_query, history_messages=history)
                final_response = refined
            else:
                final_response = initial_response

        # -- persist --
        yield {"type": "stage", "stage": "persisting"}
        self._conversation_service.add_message(user_id, "user", query)
        self._conversation_service.add_message(
            user_id, "assistant", final_response, sources=sources,
        )

        # -- summarize if needed --
        was_summarized = False
        if len(history) + 2 >= SUMMARIZE_MAX_MESSAGES:
            yield {"type": "stage", "stage": "summarizing"}
            summarization_result = await self._summarizer.summarize_and_trim(user_id, history)
            if summarization_result:
                was_summarized = summarization_result.get("was_summarized", False)

        # -- done --
        yield {
            "type": "done",
            "response": final_response,
            "sources": sources,
            "summarized": was_summarized,
            "session_id": session_label,
        }


# ------------------------------------------------------------------ #
# Module-level helpers (pure, easily testable)                        #
# ------------------------------------------------------------------ #


def _normalize_history(history: Any) -> List[Dict[str, str]]:
    if not isinstance(history, list):
        return []
    return [
        {"role": str(item["role"]).strip(), "content": str(item["content"]).strip()}
        for item in history
        if isinstance(item, dict) and str(item.get("role", "")).strip() and str(item.get("content", "")).strip()
    ]
