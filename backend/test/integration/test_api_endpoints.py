import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

# The starlette TestClient reports its direct peer as the literal string
# "testclient". Treat it as a trusted proxy so the X-authentik-* identity
# headers are honored, mirroring the nginx reverse proxy in production. Must be
# set before the api module is imported.
os.environ["TRUSTED_PROXY_IPS"] = os.environ.get("TRUSTED_PROXY_IPS") or "testclient"

from backend.src.api import api
from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.users_repository import UsersRepository
from backend.src.infrastructure.security.authentik import AuthentikAdminClient
from backend.test.integration.db_test_utils import (
    bind_session_to_schema,
    create_isolated_test_engine,
    drop_isolated_schema,
)

ALICE_SUB = "authentik-sub-alice"
ALICE_USERNAME = "alice"


class DummyChatbot:
    def __init__(self) -> None:
        self.queries = []
        self.reset_calls = []
        self.purge_calls = []

    async def chat(self, query: str, user_id: int, session_id: str | None = None):
        self.queries.append((query, user_id, session_id))
        return {
            "response": f"echo:{query}",
            "sources": ["source-1"],
            "summarized": False,
            "session_id": session_id or "generated-session",
        }

    def get_history(self, user_id: int):
        return [
            {"role": "user", "content": "Hello", "sources": []},
            {"role": "assistant", "content": "Hi there", "sources": ["source-1"]},
        ]

    def end_session(self, user_id: int):
        self.reset_calls.append(user_id)
        return True

    def purge_user_data(self, user_id: int):
        self.purge_calls.append(user_id)


class ApiEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.schema_name = create_isolated_test_engine()
        cls.schema_engine = bind_session_to_schema(cls.engine, cls.schema_name)
        Base.metadata.create_all(bind=cls.schema_engine)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=self.schema_engine)
        Base.metadata.create_all(bind=self.schema_engine)

        self.init_patch = patch("backend.src.api.api.initialize_database", autospec=True)
        self.chatbot_patch = patch("backend.src.api.api.build_chatbot_service", autospec=True)
        self.admin_delete_patch = patch.object(AuthentikAdminClient, "delete_user", return_value=True)
        self.rate_limit_patch = patch(
            "backend.src.infrastructure.security.rate_limit.check_query_rate_limit",
            return_value=(True, 29),
        )
        self.init_patch.start()
        self.chatbot = DummyChatbot()
        self.chatbot_patch.start().return_value = self.chatbot
        self.admin_delete_patch.start()
        self.rate_limit_patch.start()

        self.client = TestClient(api.app, base_url="https://testserver")
        with self.client:
            pass

        self.users = UsersRepository()
        self.user_id, _ = self.users.get_or_create_user_by_sub(ALICE_SUB, ALICE_USERNAME)

    def _auth_headers(self) -> dict[str, str]:
        """Headers the trusted nginx proxy would forward for an Authentik session."""
        return {"X-authentik-uid": ALICE_SUB, "X-authentik-username": ALICE_USERNAME}

    def tearDown(self) -> None:
        self.rate_limit_patch.stop()
        self.admin_delete_patch.stop()
        self.chatbot_patch.stop()
        self.init_patch.stop()

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

    def test_me_requires_authentication(self) -> None:
        response = self.client.get("/auth/me")

        self.assertEqual(response.status_code, 401)

    def test_me_returns_identity_from_forwarded_headers(self) -> None:
        response = self.client.get("/auth/me", headers=self._auth_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], ALICE_USERNAME)
        self.assertEqual(response.json()["id"], self.user_id)

    def test_me_rejects_identity_from_untrusted_peer(self) -> None:
        # An end client that is not the nginx proxy must never have its
        # X-authentik-* headers honored (fail closed).
        with patch.object(api, "_is_trusted_proxy", return_value=False):
            response = self.client.get("/auth/me", headers=self._auth_headers())

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_spoofed_uid_without_username(self) -> None:
        response = self.client.get(
            "/auth/me",
            headers={"X-authentik-uid": ALICE_SUB},
        )

        self.assertEqual(response.status_code, 401)

    def test_query_endpoint_requires_authentication(self) -> None:
        response = self.client.post("/query", json={"query": "Como aplicar insulina?"})

        self.assertEqual(response.status_code, 401)

    def test_authenticated_query_endpoint_returns_payload(self) -> None:
        response = self.client.post(
            "/query",
            json={"query": "Como aplicar insulina?"},
            headers=self._auth_headers(),
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"], "echo:Como aplicar insulina?")
        self.assertEqual(payload["sources"], ["source-1"])
        self.assertFalse(payload["summarized"])
        self.assertIsInstance(payload["session_id"], str)
        self.assertEqual(self.chatbot.queries[0][0], "Como aplicar insulina?")
        self.assertEqual(self.chatbot.queries[0][1], self.user_id)
        self.assertEqual(self.chatbot.queries[0][2], payload["session_id"])

    def test_query_endpoint_uses_provided_session_id(self) -> None:
        response = self.client.post(
            "/query",
            json={"query": "Olá", "session_id": "session-123"},
            headers=self._auth_headers(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-123")
        self.assertEqual(self.chatbot.queries[-1], ("Olá", self.user_id, "session-123"))

    def test_query_endpoint_honors_rate_limit(self) -> None:
        with patch(
            "backend.src.infrastructure.security.rate_limit.check_query_rate_limit",
            return_value=(False, 0),
        ):
            response = self.client.post(
                "/query",
                json={"query": "Pergunta"},
                headers=self._auth_headers(),
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.chatbot.queries, [])

    def test_clear_session_endpoint_requires_authentication(self) -> None:
        response = self.client.delete("/user/conversations")

        self.assertEqual(response.status_code, 401)

    def test_clear_session_endpoint_clears_current_user(self) -> None:
        response = self.client.delete("/user/conversations", headers=self._auth_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Conversa limpa com sucesso")
        self.assertEqual(self.chatbot.reset_calls, [self.user_id])

    def test_get_conversations_requires_authentication(self) -> None:
        response = self.client.get("/user/conversations")

        self.assertEqual(response.status_code, 401)

    def test_get_conversations_returns_message_list(self) -> None:
        response = self.client.get("/user/conversations", headers=self._auth_headers())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("messages", payload)
        messages = payload["messages"]
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[0]["sources"], [])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Hi there")
        self.assertEqual(messages[1]["sources"], ["source-1"])

    def test_delete_me_deletes_current_user(self) -> None:
        response = self.client.request("DELETE", "/auth/me", headers=self._auth_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Usuario excluido com sucesso")
        self.assertIsNone(self.users.get_user_by_id(self.user_id))

    def test_delete_me_purges_cached_user_data_after_revoking_authentik(self) -> None:
        events: list[str] = []
        with patch.object(
            AuthentikAdminClient,
            "delete_user",
            side_effect=lambda username: events.append(f"admin:{username}") or True,
        ), patch.object(DummyChatbot, "purge_user_data", side_effect=lambda user_id: events.append(f"purge:{user_id}")):
            response = self.client.request("DELETE", "/auth/me", headers=self._auth_headers())

        self.assertEqual(response.status_code, 200)
        # SSO access is revoked BEFORE any local data is purged, so a partial
        # failure can never leave an Authentik account pointing at nothing.
        self.assertEqual(events, [f"admin:{ALICE_USERNAME}", f"purge:{self.user_id}"])

    def test_delete_me_fails_closed_when_authentik_revocation_fails(self) -> None:
        with patch.object(AuthentikAdminClient, "delete_user", return_value=False):
            response = self.client.request("DELETE", "/auth/me", headers=self._auth_headers())

        self.assertEqual(response.status_code, 502)
        # Local data must be kept untouched if the SSO account was not revoked.
        self.assertIsNotNone(self.users.get_user_by_id(self.user_id))
        self.assertEqual(self.chatbot.purge_calls, [])

    def test_docs_and_openapi_disabled_by_default(self) -> None:
        self.assertEqual(self.client.get("/docs").status_code, 404)
        self.assertEqual(self.client.get("/redoc").status_code, 404)
        self.assertEqual(self.client.get("/openapi.json").status_code, 404)

    def test_health_returns_503_when_chatbot_missing(self) -> None:
        api.app.state.chatbot = None

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
