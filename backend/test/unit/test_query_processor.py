import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from backend.src.application.features.chat.query_processor import QueryProcessor
from lightrag.prompt import PROMPTS


class DummyRAGRuntime:
    def __init__(self, rag_data: Any = None, error: Optional[Exception] = None):
        self._rag_data = rag_data or {}
        self._error = error

    def query_data(
        self,
        query: str,
        mode: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_total_tokens: int = 12000,
        top_k: int = 10,
    ) -> Any:
        _ = (query, mode, conversation_history, system_prompt, max_total_tokens, top_k)
        if self._error is not None:
            raise self._error
        return self._rag_data


class DummyConversationService:
    def __init__(self, history: Optional[List[Dict[str, str]]] = None, summarized: bool = False):
        self.history = history or []
        self.summarized = summarized
        self.added_messages: List[tuple[int, str, str]] = []

    def get_conversation(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        _ = (user_id, limit)
        return list(self.history)

    def add_message(self, user_id: int, role: str, content: str) -> None:
        self.added_messages.append((user_id, role, content))

    def consume_summarized(self, user_id: int) -> bool:
        _ = user_id
        return self.summarized


def make_processor(
    rag_runtime: DummyRAGRuntime,
    conversation_service: DummyConversationService,
    call_llm: AsyncMock,
) -> QueryProcessor:
    return QueryProcessor(rag_runtime, conversation_service, call_llm)


class QueryProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_without_refinement(self) -> None:
        rag_data = {
            "status": "success",
            "data": {
                "chunks": [
                    {"file_path": "data/raw/doc1.md"},
                    {"file_path": "data/raw/doc1.md"},
                ]
            },
        }
        conversation_service = DummyConversationService(history=[{"role": "user", "content": "Oi"}], summarized=True)
        call_llm = AsyncMock(
            side_effect=[
                "Resposta inicial",
                '{"needs_refinement": false, "issues": [], "suggestions": []}',
            ]
        )
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=42, session_id="sess-1")

        self.assertEqual(result["response"], "Resposta inicial")
        self.assertEqual(result["sources"], ["doc1.md"])
        self.assertEqual(result["source_count"], 1)
        self.assertTrue(result["summarized"])
        self.assertEqual(result["session_id"], "sess-1")
        self.assertEqual(len(conversation_service.added_messages), 2)
        self.assertEqual(conversation_service.added_messages[0], (42, "user", "Pergunta"))
        self.assertEqual(conversation_service.added_messages[1], (42, "assistant", "Resposta inicial"))
        self.assertEqual(call_llm.await_count, 2)

    async def test_query_with_refinement(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(
            side_effect=[
                "Resposta inicial",
                '{"needs_refinement": true, "issues": ["x"], "suggestions": ["y"]}',
                "Resposta refinada",
            ]
        )
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=7)

        self.assertEqual(result["response"], "Resposta refinada")
        self.assertEqual(result["source_count"], 0)
        self.assertFalse(result["summarized"])
        self.assertEqual(conversation_service.added_messages[0], (7, "user", "Pergunta"))
        self.assertEqual(conversation_service.added_messages[1], (7, "assistant", "Resposta refinada"))
        self.assertEqual(call_llm.await_count, 3)

    async def test_query_with_malformed_critique_falls_back(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(side_effect=["Resposta inicial", "not-json"])
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=9)

        self.assertEqual(result["response"], "Resposta inicial")
        self.assertEqual(call_llm.await_count, 2)
        self.assertEqual(conversation_service.added_messages[1], (9, "assistant", "Resposta inicial"))

    async def test_query_skips_critique_when_fail_response(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(return_value=PROMPTS["fail_response"])
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=10)

        self.assertEqual(result["response"], PROMPTS["fail_response"])
        self.assertEqual(call_llm.await_count, 1)
        self.assertEqual(conversation_service.added_messages[0], (10, "user", "Pergunta"))
        self.assertEqual(conversation_service.added_messages[1], (10, "assistant", PROMPTS["fail_response"]))

    async def test_query_rag_error_does_not_persist_messages(self) -> None:
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(return_value="unused")
        processor = make_processor(DummyRAGRuntime(error=RuntimeError("RAG failed")), conversation_service, call_llm)

        with self.assertRaises(RuntimeError):
            await processor.query("Pergunta", user_id=5)

        self.assertEqual(conversation_service.added_messages, [])
        self.assertEqual(call_llm.await_count, 0)


if __name__ == "__main__":
    unittest.main()
