import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

# The api module reads POCKETBASE_* configuration at import time; provide test
# defaults that can be overridden by the environment.
os.environ.setdefault("POCKETBASE_URL", "http://pocketbase:8090")
os.environ.setdefault("POCKETBASE_SUPERUSER_EMAIL", "admin@test.internal")
os.environ.setdefault("POCKETBASE_SUPERUSER_PASSWORD", "test-password")

from backend.src.api import api
from backend.src.application.features.auth.auth_service import AuthenticationService
from backend.src.infrastructure.security.pocketbase import (
    PocketBaseAccountDeletionClient,
    PocketBaseTokenError,
)

ALICE_USER_ID = "user0000alice0001"  # 15-char pocketbase-style record id
ALICE_EMAIL = "alice@example.com"


def make_token(user_id: str, email: str | None = None) -> str:
    """Opaque stand-in for a real PocketBase auth token.

    The suite replaces ``verify_access_token`` with a fake that accepts only
    tokens in this format; everything else is rejected exactly like an
    expired/revoked/garbage real-world token would be.
    """
    return f"ok|{user_id}|{email or ''}"


def fake_verify_access_token(token: str) -> dict[str, str]:
    parts = token.split("|")
    if len(parts) == 3 and parts[0] == "ok" and parts[1]:
        return {"id": parts[1], "email": parts[2]}
    raise PocketBaseTokenError("invalid token")


class DummyChatbot:
    def __init__(self) -> None:
        self.queries = []
        self.reset_calls = []
        self.purge_calls = []

    async def chat(self, query: str, user_id: str, session_id: str | None = None):
        self.queries.append((query, user_id, session_id))
        return {
            "response": f"echo:{query}",
            "sources": [{"path": "source-1", "page": None, "content": None}],
            "summarized": False,
            "session_id": session_id or "generated-session",
        }

    def get_history(self, user_id: str):
        return [
            {"role": "user", "content": "Hello", "sources": []},
            {
                "role": "assistant",
                "content": "Hi there",
            "sources": [{"path": "source-1", "page": None, "content": None}],
            },
        ]

    def end_session(self, user_id: str):
        self.reset_calls.append(user_id)
        return True

    def purge_user_data(self, user_id: str):
        self.purge_calls.append(user_id)

class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot_patch = patch("backend.src.api.api.build_chatbot_service", autospec=True)
        self.auth_patch = patch("backend.src.api.api.build_auth_service", autospec=True)
        self.verify_patch = patch(
            "backend.src.api.api.verify_access_token", side_effect=fake_verify_access_token
        )
        self.rate_limit_patch = patch(
            "backend.src.infrastructure.security.rate_limit.check_query_rate_limit",
            return_value=(True, 29),
        )
        self.chatbot = DummyChatbot()
        self.deletion_client = PocketBaseAccountDeletionClient()
        self.chatbot_patch.start().return_value = self.chatbot
        self.auth_patch.start().return_value = AuthenticationService(
            account_deletion_client=self.deletion_client,
            username_resolver=lambda user_id: ALICE_EMAIL if user_id == ALICE_USER_ID else None,
        )
        self.verify_patch.start()
        self.rate_limit_patch.start()

        self.client = TestClient(api.app, base_url="https://testserver")
        with self.client:
            pass

    def tearDown(self) -> None:
        self.rate_limit_patch.stop()
        self.verify_patch.stop()
        self.auth_patch.stop()
        self.chatbot_patch.stop()

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Headers a signed-in frontend would send."""
        return {"Authorization": f"Bearer {token}"}

    def _alice_headers(self) -> dict[str, str]:
        return self._auth_headers(make_token(ALICE_USER_ID, ALICE_EMAIL))

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "LinaChat API")

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_me_requires_authentication(self) -> None:
        response = self.client.get("/auth/me")

        self.assertEqual(response.status_code, 401)

    def test_me_returns_identity_from_valid_token(self) -> None:
        response = self.client.get("/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], ALICE_EMAIL)
        self.assertEqual(response.json()["id"], ALICE_USER_ID)

    def test_me_falls_back_to_derived_username_for_unknown_user(self) -> None:
        # Token without an email claim whose id the directory does not know.
        unknown_id = "user0000unknown01"
        token = make_token(unknown_id, email=None)
        response = self.client.get("/auth/me", headers=self._auth_headers(token))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], f"user_{unknown_id[:8]}")

    def test_me_rejects_missing_bearer_scheme(self) -> None:
        token = make_token(ALICE_USER_ID, ALICE_EMAIL)
        response = self.client.get("/auth/me", headers={"Authorization": token})

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_garbage_token(self) -> None:
        response = self.client.get("/auth/me", headers=self._auth_headers("not-a.jwt"))

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_token_introspection_reports_invalid(self) -> None:
        # Covers expired/revoked/unknown tokens: introspection refuses them.
        response = self.client.get("/auth/me", headers=self._auth_headers("bad|expired"))

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_introspection_without_subject(self) -> None:
        response = self.client.get("/auth/me", headers=self._auth_headers("ok||"))

        self.assertEqual(response.status_code, 401)

    def test_query_endpoint_requires_authentication(self) -> None:
        response = self.client.post("/query", json={"query": "Como aplicar insulina?"})

        self.assertEqual(response.status_code, 401)

    def test_authenticated_query_endpoint_returns_payload(self) -> None:
        response = self.client.post(
            "/query",
            json={"query": "Como aplicar insulina?"},
            headers=self._alice_headers(),
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"], "echo:Como aplicar insulina?")
        self.assertEqual(payload["sources"], [{"path": "source-1", "page": None, "content": None}])
        self.assertFalse(payload["summarized"])
        self.assertIsInstance(payload["session_id"], str)
        self.assertEqual(self.chatbot.queries[0][0], "Como aplicar insulina?")
        self.assertEqual(self.chatbot.queries[0][1], ALICE_USER_ID)
        self.assertEqual(self.chatbot.queries[0][2], payload["session_id"])

    def test_query_endpoint_uses_provided_session_id(self) -> None:
        response = self.client.post(
            "/query",
            json={"query": "Olá", "session_id": "session-123"},
            headers=self._alice_headers(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "session-123")
        self.assertEqual(self.chatbot.queries[-1], ("Olá", ALICE_USER_ID, "session-123"))

    def test_query_endpoint_honors_rate_limit(self) -> None:
        with patch(
            "backend.src.infrastructure.security.rate_limit.check_query_rate_limit",
            return_value=(False, 0),
        ):
            response = self.client.post(
                "/query",
                json={"query": "Pergunta"},
                headers=self._alice_headers(),
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.chatbot.queries, [])

    def test_clear_session_endpoint_requires_authentication(self) -> None:
        response = self.client.delete("/user/conversations")

        self.assertEqual(response.status_code, 401)

    def test_clear_session_endpoint_clears_current_user(self) -> None:
        response = self.client.delete("/user/conversations", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Conversa limpa com sucesso")
        self.assertEqual(self.chatbot.reset_calls, [ALICE_USER_ID])

    def test_get_conversations_requires_authentication(self) -> None:
        response = self.client.get("/user/conversations")

        self.assertEqual(response.status_code, 401)

    def test_get_conversations_returns_message_list(self) -> None:
        response = self.client.get("/user/conversations", headers=self._alice_headers())

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
        self.assertEqual(messages[1]["sources"], [{"path": "source-1", "page": None, "content": None}])

    def test_delete_me_deletes_current_user(self) -> None:
        with patch.object(
            PocketBaseAccountDeletionClient, "delete_user", return_value=True
        ) as delete_mock:
            response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Usuário excluído com sucesso")
        delete_mock.assert_called_once_with(ALICE_USER_ID)

    def test_delete_me_purges_cached_user_data_before_revoking(self) -> None:
        events: list[str] = []
        with patch.object(
            PocketBaseAccountDeletionClient,
            "delete_user",
            side_effect=lambda user_id: events.append(f"admin:{user_id}") or True,
        ), patch.object(DummyChatbot, "purge_user_data", side_effect=lambda user_id: events.append(f"purge:{user_id}")):
            response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        # Cached PII is purged BEFORE the PocketBase account is deleted, so a
        # partial failure can never leave cached user data pointing at a dead
        # account.
        self.assertEqual(events, [f"purge:{ALICE_USER_ID}", f"admin:{ALICE_USER_ID}"])

    def test_delete_me_fails_closed_when_revocation_fails(self) -> None:
        with patch.object(PocketBaseAccountDeletionClient, "delete_user", return_value=False):
            response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 502)

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
