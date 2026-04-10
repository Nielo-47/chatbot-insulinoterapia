import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.src import api
from backend.src.auth import hash_password
from backend.src.config import Config
from backend.src.db.models import Base
from backend.src.repositories.users_repository import UsersRepository
from backend.test.db_test_utils import bind_session_to_schema, create_isolated_test_engine, drop_isolated_schema


class DummyChatbot:
    def __init__(self) -> None:
        self.queries = []
        self.reset_calls = []

    async def initialize_rag(self) -> None:
        return None

    def query(self, query: str, user_id: int, session_id: str | None = None):
        self.queries.append((query, user_id, session_id))
        return {
            "response": f"echo:{query}",
            "sources": ["source-1"],
            "source_count": 1,
            "summarized": False,
            "session_id": session_id or "generated-session",
        }

    def reset_conversation(self, user_id: int):
        self.reset_calls.append(user_id)
        return True


class ApiEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.schema_name = create_isolated_test_engine()
        cls.schema_engine = bind_session_to_schema(cls.engine, cls.schema_name)
        Base.metadata.create_all(bind=cls.schema_engine)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.schema_engine)
        Base.metadata.create_all(bind=self.schema_engine)

        self.init_patch = patch("backend.src.api.initialize_database", autospec=True)
        self.chatbot_patch = patch("backend.src.api.Chatbot", return_value=DummyChatbot())
        self.secret_patch = patch.object(Config, "JWT_SECRET_KEY", "test-secret-key-value-long-enough-32-bytes")
        self.init_patch.start()
        self.chatbot_patch.start()
        self.secret_patch.start()

        self.client = TestClient(api.app)
        with self.client:
            pass

        self.chatbot = api.chatbot_instance
        assert self.chatbot is not None
        self.users = UsersRepository()
        self.user_id = self.users.get_or_create_user_id("alice", hash_password("password123"))

    def tearDown(self) -> None:
        self.chatbot_patch.stop()
        self.init_patch.stop()
        self.secret_patch.stop()
        api.chatbot_instance = None

    @classmethod
    def tearDownClass(cls) -> None:
        drop_isolated_schema(cls.engine, cls.schema_name)
        cls.engine.dispose()

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Diabetes Chatbot API")

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_login_endpoint_returns_bearer_token(self) -> None:
        response = self.client.post("/auth/login", json={"username": "alice", "password": "password123"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["token_type"], "bearer")
        self.assertTrue(payload["access_token"])

    def test_login_endpoint_rejects_bad_credentials(self) -> None:
        response = self.client.post("/auth/login", json={"username": "alice", "password": "wrong"})

        self.assertEqual(response.status_code, 401)

    def test_query_endpoint_requires_authentication(self) -> None:
        response = self.client.post("/query", json={"query": "Como aplicar insulina?"})

        self.assertEqual(response.status_code, 401)

    def test_authenticated_query_endpoint_returns_payload(self) -> None:
        token = self.client.post("/auth/login", json={"username": "alice", "password": "password123"}).json()["access_token"]
        response = self.client.post(
            "/query",
            json={"query": "Como aplicar insulina?"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"], "echo:Como aplicar insulina?")
        self.assertEqual(payload["source_count"], 1)
        self.assertEqual(payload["sources"], ["source-1"])
        self.assertFalse(payload["summarized"])
        self.assertIsInstance(payload["session_id"], str)
        self.assertEqual(self.chatbot.queries[0][0], "Como aplicar insulina?")
        self.assertEqual(self.chatbot.queries[0][1], self.user_id)
        self.assertEqual(self.chatbot.queries[0][2], payload["session_id"])

    def test_query_endpoint_uses_provided_session_id(self) -> None:
        token = self.client.post("/auth/login", json={"username": "alice", "password": "password123"}).json()["access_token"]
        response = self.client.post(
            "/query",
            json={"query": "Olá", "session_id": "session-123"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-123")
        self.assertEqual(self.chatbot.queries[-1], ("Olá", self.user_id, "session-123"))

    def test_clear_session_endpoint_requires_authentication(self) -> None:
        response = self.client.delete("/session/known-session")

        self.assertEqual(response.status_code, 401)

    def test_clear_session_endpoint_clears_current_user(self) -> None:
        token = self.client.post("/auth/login", json={"username": "alice", "password": "password123"}).json()["access_token"]
        response = self.client.delete(
            "/session/known-session",
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Session known-session cleared successfully")
        self.assertEqual(self.chatbot.reset_calls, [self.user_id])

    def test_me_endpoint_returns_current_user(self) -> None:
        token = self.client.post("/auth/login", json={"username": "alice", "password": "password123"}).json()["access_token"]
        response = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "alice")
        self.assertEqual(response.json()["id"], self.user_id)

    def test_health_returns_503_when_chatbot_missing(self) -> None:
        api.chatbot_instance = None

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()