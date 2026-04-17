import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.src.infrastructure.llm import client as llm_client_module
from backend.src.infrastructure.llm.client import LLMClient
from backend.src.infrastructure.rag import resilient_embeddings as embeddings_module
from backend.src.infrastructure.rag.resilient_embeddings import (
    EmbeddingProviderConfig,
    EmbeddingResilienceConfig,
    embed_with_fallback,
)


def _make_embedding_response(vectors):
    return SimpleNamespace(data=[SimpleNamespace(embedding=vector) for vector in vectors])


def _make_chat_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class EmbeddingFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_uses_openrouter_after_tei_retry(self) -> None:
        class FakeAsyncOpenAI:
            calls = []

            def __init__(self, *, api_key, base_url, timeout):
                self.api_key = api_key
                self.base_url = base_url
                self.timeout = timeout
                self.embeddings = SimpleNamespace(create=self._create)

            async def _create(self, model, input):
                FakeAsyncOpenAI.calls.append((self.base_url, model, tuple(input)))
                if self.base_url == "http://tei:8080/v1":
                    raise TimeoutError("tei timeout")
                return _make_embedding_response([[0.1, 0.2], [0.3, 0.4]])

        config = EmbeddingResilienceConfig(
            embedding_dim=2,
            timeout_seconds=1,
            primary_retries=1,
            fallback_retries=0,
            primary=EmbeddingProviderConfig(
                name="tei",
                base_url="http://tei:8080/v1",
                api_key="",
                model="BAAI/bge-m3",
            ),
            fallback=EmbeddingProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="fallback-key",
                model="BAAI/bge-m3",
            ),
        )

        with patch.object(embeddings_module, "AsyncOpenAI", FakeAsyncOpenAI):
            result = await embed_with_fallback(["a", "b"], config)

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(
            FakeAsyncOpenAI.calls,
            [
                ("http://tei:8080/v1", "BAAI/bge-m3", ("a", "b")),
                ("http://tei:8080/v1", "BAAI/bge-m3", ("a", "b")),
                ("https://openrouter.ai/api/v1", "BAAI/bge-m3", ("a", "b")),
            ],
        )


class LLMFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_uses_fallback_model_after_primary_retry(self) -> None:
        class FakeAsyncOpenAI:
            calls = []

            def __init__(self, *, api_key, base_url, timeout):
                self.api_key = api_key
                self.base_url = base_url
                self.timeout = timeout
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            async def _create(self, model, messages, temperature, max_tokens, extra_headers=None, extra_body=None):
                FakeAsyncOpenAI.calls.append((self.base_url, model, temperature, max_tokens))
                if model == "primary-model":
                    raise TimeoutError("primary timeout")
                return _make_chat_response("fallback answer")

        with patch.object(llm_client_module, "AsyncOpenAI", FakeAsyncOpenAI), patch.object(
            llm_client_module, "LLM_MODEL", "primary-model"
        ), patch.object(llm_client_module, "LLM_FALLBACK_MODEL", "fallback-model"), patch.object(
            llm_client_module, "LLM_PRIMARY_RETRIES", 1
        ), patch.object(
            llm_client_module, "LLM_TIMEOUT_SECONDS", 1
        ):
            client = LLMClient(api_key="test-key", base_url="https://openrouter.ai/api/v1")
            response = await client.complete("Pergunta")

        self.assertEqual(response, "fallback answer")
        self.assertEqual(
            FakeAsyncOpenAI.calls,
            [
                ("https://openrouter.ai/api/v1", "primary-model", 0.1, 800),
                ("https://openrouter.ai/api/v1", "primary-model", 0.1, 800),
                ("https://openrouter.ai/api/v1", "fallback-model", 0.1, 800),
            ],
        )


if __name__ == "__main__":
    unittest.main()
