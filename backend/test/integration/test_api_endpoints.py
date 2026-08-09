import os
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

# The api module reads SUPABASE_* configuration at import time; provide test
# defaults that can be overridden by the environment (e.g. a real JWKS URL).
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault(
    "SUPABASE_JWKS_URL", "https://test.supabase.co/auth/v1/.well-known/jwks.json"
)

from backend.src.api import api
from backend.src.infrastructure.data.models import Base
from backend.src.infrastructure.repositories.profiles_repository import ProfilesRepository
from backend.src.infrastructure.security import supabase as supabase_security
from backend.src.infrastructure.security.supabase import SupabaseAccountDeletionClient
from backend.test.integration.db_test_utils import (
    bind_session_to_schema,
    create_isolated_test_engine,
    drop_isolated_schema,
)

ALICE_USER_ID = uuid.uuid4()
ALICE_EMAIL = "alice@example.com"

# Access tokens are verified against the project's public JWKS using the
# algorithm that JWKS advertises (RS256 or ES256). The tests mint their own
# RS256-signed tokens and swap the JWKS client for a stub that hands out the
# matching public key and algorithm.
_TEST_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PUBLIC_KEY = _TEST_RSA_KEY.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
).decode()
_TEST_PRIVATE_KEY = _TEST_RSA_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()
_OTHER_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_PRIVATE_KEY = _OTHER_RSA_KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
).decode()


class _FakeSigningKey:
    def __init__(self, key: str, algorithm_name: str = "RS256") -> None:
        self.key = key
        self.algorithm_name = algorithm_name


class _FakeJWKSClient:
    """Stand-in for jwt.PyJWKClient returning a fixed public key + algorithm."""

    def __init__(self, public_key: str, algorithm_name: str = "RS256") -> None:
        self._public_key = public_key
        self._algorithm_name = algorithm_name

    def get_signing_key_from_jwt(self, token: str):
        return _FakeSigningKey(self._public_key, self._algorithm_name)


def make_token(user_id: uuid.UUID, email: str, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": "https://test.supabase.co/auth/v1",
        "sub": str(user_id),
        "aud": "authenticated",
        "role": "authenticated",
        "email": email,
        "iat": now,
        "exp": now + 3600,
    }
    claims.update(overrides)
    return jwt.encode(claims, _TEST_PRIVATE_KEY, algorithm="RS256")


class DummyChatbot:
    def __init__(self) -> None:
        self.queries = []
        self.reset_calls = []
        self.purge_calls = []

    async def chat(self, query: str, user_id: uuid.UUID, session_id: str | None = None):
        self.queries.append((query, user_id, session_id))
        return {
            "response": f"echo:{query}",
            "sources": [{"path": "source-1", "page": None, "content": None}],
            "summarized": False,
            "session_id": session_id or "generated-session",
        }

    def get_history(self, user_id: uuid.UUID):
        return [
            {"role": "user", "content": "Hello", "sources": []},
            {
                "role": "assistant",
                "content": "Hi there",
            "sources": [{"path": "source-1", "page": None, "content": None}],
            },
        ]

    def end_session(self, user_id: uuid.UUID):
        self.reset_calls.append(user_id)
        return True

    def purge_user_data(self, user_id: uuid.UUID):
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
        self.admin_delete_patch = patch.object(
            SupabaseAccountDeletionClient, "delete_user", return_value=True
        )
        self.jwks_patch = patch.object(
            supabase_security, "jwks_client", _FakeJWKSClient(_TEST_PUBLIC_KEY)
        )
        self.rate_limit_patch = patch(
            "backend.src.infrastructure.security.rate_limit.check_query_rate_limit",
            return_value=(True, 29),
        )
        self.init_patch.start()
        self.chatbot = DummyChatbot()
        self.chatbot_patch.start().return_value = self.chatbot
        self.admin_delete_patch.start()
        self.jwks_patch.start()
        self.rate_limit_patch.start()

        self.client = TestClient(api.app, base_url="https://testserver")
        with self.client:
            pass

        self.profiles = ProfilesRepository()
        self.profiles.get_or_create_profile(ALICE_USER_ID, ALICE_EMAIL)

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Headers a signed-in frontend would send."""
        return {"Authorization": f"Bearer {token}"}

    def _alice_headers(self) -> dict[str, str]:
        return self._auth_headers(make_token(ALICE_USER_ID, ALICE_EMAIL))

    def tearDown(self) -> None:
        self.rate_limit_patch.stop()
        self.jwks_patch.stop()
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

    def test_me_returns_identity_from_valid_token(self) -> None:
        response = self.client.get("/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], ALICE_EMAIL)
        self.assertEqual(response.json()["id"], str(ALICE_USER_ID))

    def test_me_rejects_missing_bearer_scheme(self) -> None:
        token = make_token(ALICE_USER_ID, ALICE_EMAIL)
        response = self.client.get("/auth/me", headers={"Authorization": token})

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_garbage_token(self) -> None:
        response = self.client.get("/auth/me", headers=self._auth_headers("not-a.jwt"))

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_anon_token(self) -> None:
        anon_token = make_token(ALICE_USER_ID, ALICE_EMAIL, role="anon")
        response = self.client.get("/auth/me", headers=self._auth_headers(anon_token))

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_expired_token(self) -> None:
        now = int(time.time())
        expired = make_token(ALICE_USER_ID, ALICE_EMAIL, iat=now - 7200, exp=now - 3600)
        response = self.client.get("/auth/me", headers=self._auth_headers(expired))

        self.assertEqual(response.status_code, 401)

    def test_me_rejects_token_signed_with_wrong_key(self) -> None:
        other = jwt.encode(
            {
                "iss": "https://other.supabase.co/auth/v1",
                "sub": str(ALICE_USER_ID),
                "aud": "authenticated",
                "role": "authenticated",
                "email": ALICE_EMAIL,
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
            },
            _OTHER_PRIVATE_KEY,
            algorithm="RS256",
        )
        response = self.client.get("/auth/me", headers=self._auth_headers(other))

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
        response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Usuário excluído com sucesso")
        self.assertIsNone(self.profiles.get_profile_by_id(ALICE_USER_ID))

    def test_delete_me_forwards_the_callers_access_token(self) -> None:
        token = make_token(ALICE_USER_ID, ALICE_EMAIL)
        captured: list[str] = []
        with patch.object(
            SupabaseAccountDeletionClient,
            "delete_user",
            side_effect=lambda _user_id, access_token: captured.append(access_token) or True,
        ):
            response = self.client.request("DELETE", "/auth/me", headers=self._auth_headers(token))

        self.assertEqual(response.status_code, 200)
        # The edge function authenticates the caller, so the backend must
        # forward the exact access token the user authenticated with.
        self.assertEqual(captured, [token])

    def test_delete_me_purges_cached_user_data_before_revoking(self) -> None:
        events: list[str] = []
        with patch.object(
            SupabaseAccountDeletionClient,
            "delete_user",
            side_effect=lambda user_id, _access_token: events.append(f"admin:{user_id}") or True,
        ), patch.object(DummyChatbot, "purge_user_data", side_effect=lambda user_id: events.append(f"purge:{user_id}")):
            response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 200)
        # Cached PII is purged BEFORE the Supabase account is revoked, so a
        # partial failure can never leave cached user data pointing at a dead
        # account.
        self.assertEqual(events, [f"purge:{ALICE_USER_ID}", f"admin:{ALICE_USER_ID}"])

    def test_delete_me_fails_closed_when_revocation_fails(self) -> None:
        with patch.object(SupabaseAccountDeletionClient, "delete_user", return_value=False):
            response = self.client.request("DELETE", "/auth/me", headers=self._alice_headers())

        self.assertEqual(response.status_code, 502)
        # The local profile must be kept untouched if the Auth account was not
        # revoked.
        self.assertIsNotNone(self.profiles.get_profile_by_id(ALICE_USER_ID))

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
