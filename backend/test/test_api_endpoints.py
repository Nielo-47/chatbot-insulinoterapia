import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.src import api


class DummyChatbot:
    def __init__(self) -> None:
        self.queries = []
        self.reset_calls = []

    async def initialize_rag(self) -> None:
        return None

    def query(self, query: str, session_id: str | None = None):
        self.queries.append((query, session_id))
        return {
            "response": f"echo:{query}",
            "sources": ["source-1"],
            "source_count": 1,
            "summarized": False,
        }

    def reset_conversation(self, session_id: str):
        self.reset_calls.append(session_id)
        return session_id == "known-session"


class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.init_patch = patch("backend.src.api.initialize_database", autospec=True)
        self.chatbot_patch = patch("backend.src.api.Chatbot", return_value=DummyChatbot())
        self.init_patch.start()
        self.chatbot_mock = self.chatbot_patch.start()

        self.client = TestClient(api.app)
        with self.client:
            pass

        self.chatbot = api.chatbot_instance
        assert self.chatbot is not None

    def tearDown(self) -> None:
        self.chatbot_patch.stop()
        self.init_patch.stop()
        api.chatbot_instance = None

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Diabetes Chatbot API")

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_query_endpoint_generates_session_and_returns_payload(self) -> None:
        response = self.client.post("/query", json={"query": "Como aplicar insulina?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"], "echo:Como aplicar insulina?")
        self.assertEqual(payload["source_count"], 1)
        self.assertEqual(payload["sources"], ["source-1"])
        self.assertFalse(payload["summarized"])
        self.assertIsInstance(payload["session_id"], str)
        self.assertEqual(self.chatbot.queries[0][0], "Como aplicar insulina?")
        self.assertEqual(self.chatbot.queries[0][1], payload["session_id"])

    def test_query_endpoint_uses_provided_session_id(self) -> None:
        response = self.client.post(
            "/query",
            json={"query": "Olá", "session_id": "session-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-123")
        self.assertEqual(self.chatbot.queries[-1], ("Olá", "session-123"))

    def test_clear_session_endpoint_reports_known_and_unknown_sessions(self) -> None:
        known = self.client.delete("/session/known-session")
        unknown = self.client.delete("/session/missing-session")

        self.assertEqual(known.status_code, 200)
        self.assertEqual(known.json()["message"], "Session known-session cleared successfully")
        self.assertEqual(unknown.status_code, 200)
        self.assertEqual(unknown.json()["message"], "Session missing-session not found")
        self.assertEqual(self.chatbot.reset_calls, ["known-session", "missing-session"])

    def test_health_returns_503_when_chatbot_missing(self) -> None:
        api.chatbot_instance = None

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()