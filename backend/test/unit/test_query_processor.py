import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

from backend.src.application.features.chat.query_processor import QueryProcessor
from backend.src.infrastructure.rag.cleaner import extract_sources
from lightrag.prompt import PROMPTS


class DummyRAGRuntime:
    def __init__(self, rag_data: Any = None, error: Optional[Exception] = None):
        self._rag_data = rag_data or {}
        self._error = error
        self.last_history: List[Dict[str, str]] = []

    def query_data(
        self,
        query: str,
        mode: str,
        conversation_history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_total_tokens: int = 12000,
        top_k: int = 10,
    ) -> Any:
        _ = (query, mode, system_prompt, max_total_tokens, top_k)
        self.last_history = list(conversation_history)
        if self._error is not None:
            raise self._error
        sources, source_count = extract_sources(self._rag_data)
        return {
            "rag_data": self._rag_data,
            "sources": sources,
            "source_count": source_count,
        }


class DummyConversationService:
    def __init__(self, history: Optional[List[Dict[str, str]]] = None, summarized: bool = False):
        self.history = history or []
        self.summarized = summarized
        self.added_messages: List[tuple[int, str, str, List[str]]] = []

    def get_conversation(self, user_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        _ = (user_id, limit)
        return list(self.history)

    def add_message(self, user_id: int, role: str, content: str, sources: Optional[List[str]] = None) -> None:
        self.added_messages.append((user_id, role, content, list(sources or [])))

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
        # Test with correct LightRAG format: references have file_path, chunks have reference_id
        rag_data = {
            "status": "success",
            "data": {
                "chunks": [
                    {"chunk_id": "chunk-1", "reference_id": "1"},
                    {"chunk_id": "chunk-2", "reference_id": "1"},
                ],
                "references": [
                    {"reference_id": "1", "file_path": "data/raw/doc1.md"},
                ],
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
        self.assertEqual(conversation_service.added_messages[0], (42, "user", "Pergunta", []))
        self.assertEqual(conversation_service.added_messages[1], (42, "assistant", "Resposta inicial", ["doc1.md"]))
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
        self.assertEqual(conversation_service.added_messages[0], (7, "user", "Pergunta", []))
        self.assertEqual(conversation_service.added_messages[1], (7, "assistant", "Resposta refinada", []))
        self.assertEqual(call_llm.await_count, 3)

    async def test_query_with_malformed_critique_falls_back(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(side_effect=["Resposta inicial", "not-json"])
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=9)

        self.assertEqual(result["response"], "Resposta inicial")
        self.assertEqual(call_llm.await_count, 2)
        self.assertEqual(conversation_service.added_messages[1], (9, "assistant", "Resposta inicial", []))

    async def test_query_skips_critique_when_fail_response(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(return_value=PROMPTS["fail_response"])
        processor = make_processor(DummyRAGRuntime(rag_data=rag_data), conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=10)

        self.assertEqual(result["response"], PROMPTS["fail_response"])
        self.assertEqual(call_llm.await_count, 1)
        self.assertEqual(conversation_service.added_messages[0], (10, "user", "Pergunta", []))
        self.assertEqual(conversation_service.added_messages[1], (10, "assistant", PROMPTS["fail_response"], []))

    async def test_query_rag_error_does_not_persist_messages(self) -> None:
        conversation_service = DummyConversationService()
        call_llm = AsyncMock(return_value="unused")
        processor = make_processor(DummyRAGRuntime(error=RuntimeError("RAG failed")), conversation_service, call_llm)

        with self.assertRaises(RuntimeError):
            await processor.query("Pergunta", user_id=5)

        self.assertEqual(conversation_service.added_messages, [])
        self.assertEqual(call_llm.await_count, 0)

    async def test_query_normalizes_history_with_reference_metadata(self) -> None:
        rag_data = {"status": "success", "data": {"chunks": []}}
        conversation_service = DummyConversationService(
            history=[
                {
                    "role": "assistant",
                    "content": "Resposta anterior",
                    "sources": ["doc.md"],
                    "source_count": 1,
                }
            ]
        )
        call_llm = AsyncMock(
            side_effect=[
                "Nova resposta",
                '{"needs_refinement": false, "issues": [], "suggestions": []}',
            ]
        )
        rag_runtime = DummyRAGRuntime(rag_data=rag_data)
        processor = make_processor(rag_runtime, conversation_service, call_llm)

        result = await processor.query("Pergunta", user_id=11)

        self.assertEqual(result["response"], "Nova resposta")
        self.assertEqual(rag_runtime.last_history, [{"role": "assistant", "content": "Resposta anterior"}])


if __name__ == "__main__":
    unittest.main()
