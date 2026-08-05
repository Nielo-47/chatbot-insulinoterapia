import unittest
from typing import Any, Optional

from backend.src.application.contracts.repositories import UsersRepositoryLike
from backend.src.application.features.auth.auth_service import (
    AuthenticationService,
    build_authentication_service,
)


class UsersRepositoryStub(UsersRepositoryLike):
    def __init__(self) -> None:
        self.records: dict[int, tuple[str, str]] = {}  # user_id -> (username, sub)
        self.by_sub: dict[str, int] = {}
        self.created: list[tuple[str, str]] = []
        self.deleted_user_id: Optional[int] = None
        self._next_id = 1

    def get_user_by_id(self, user_id: int) -> Optional[Any]:
        if user_id not in self.records:
            return None
        username, sub = self.records[user_id]
        return SimpleUser(user_id, username, sub)

    def get_user_by_sub(self, sub: str) -> Optional[Any]:
        user_id = self.by_sub.get(sub)
        if user_id is None:
            return None
        username, _ = self.records[user_id]
        return SimpleUser(user_id, username, sub)

    def get_or_create_user_by_sub(self, sub: str, username: str) -> tuple[int, bool]:
        existing = self.by_sub.get(sub)
        if existing is not None:
            return existing, False
        user_id = self._next_id
        self._next_id += 1
        self.records[user_id] = (username, sub)
        self.by_sub[sub] = user_id
        self.created.append((sub, username))
        return user_id, True

    def delete_user_by_id(self, user_id: int) -> bool:
        if user_id not in self.records:
            return False
        username, sub = self.records.pop(user_id)
        self.by_sub.pop(sub, None)
        self.deleted_user_id = user_id
        return True


class SimpleUser:
    def __init__(self, user_id: int, username: str, sub: Optional[str]) -> None:
        self.id = user_id
        self.username = username
        self.authentik_sub = sub


class AuthentikAdminClientStub:
    def __init__(self) -> None:
        self.deleted_username: Optional[str] = None
        self.result = True

    def delete_user(self, username: str) -> bool:
        self.deleted_username = username
        return self.result


def _build_service() -> tuple[AuthenticationService, UsersRepositoryStub, AuthentikAdminClientStub]:
    users_repository = UsersRepositoryStub()
    admin_client = AuthentikAdminClientStub()
    service = AuthenticationService(
        users_repository=users_repository,
        authentik_admin_client=admin_client,
    )
    return service, users_repository, admin_client


class AuthenticationServiceTests(unittest.TestCase):
    def test_resolve_principal_provisions_user_on_first_login(self) -> None:
        service, users_repository, _ = _build_service()

        principal = service.resolve_principal_from_identity("sub-123", "alice")

        self.assertEqual(principal.id, 1)
        self.assertEqual(principal.username, "alice")
        self.assertEqual(users_repository.created, [("sub-123", "alice")])

    def test_resolve_principal_reuses_existing_identity(self) -> None:
        service, users_repository, _ = _build_service()
        first = service.resolve_principal_from_identity("sub-123", "alice")
        second = service.resolve_principal_from_identity("sub-123", "alice")

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(users_repository.created), 1)

    def test_delete_user_delegates_to_repository(self) -> None:
        service, users_repository, _ = _build_service()
        service.resolve_principal_from_identity("sub-123", "alice")

        deleted = service.delete_user(1)

        self.assertTrue(deleted)
        self.assertEqual(users_repository.deleted_user_id, 1)

    def test_delete_user_returns_false_for_missing_user(self) -> None:
        service, _, _ = _build_service()

        deleted = service.delete_user(999)

        self.assertFalse(deleted)

    def test_delete_authentik_user_delegates_to_admin_client(self) -> None:
        service, _, admin_client = _build_service()

        deleted = service.delete_authentik_user("alice")

        self.assertTrue(deleted)
        self.assertEqual(admin_client.deleted_username, "alice")

    def test_delete_authentik_user_propagates_failure(self) -> None:
        service, _, admin_client = _build_service()
        admin_client.result = False

        deleted = service.delete_authentik_user("alice")

        self.assertFalse(deleted)


class BuildAuthenticationServiceTests(unittest.TestCase):
    def test_factory_wires_repository_and_admin_client(self) -> None:
        users_repository = UsersRepositoryStub()
        admin_client = AuthentikAdminClientStub()

        service = build_authentication_service(
            users_repository=users_repository,
            authentik_admin_client=admin_client,
        )

        self.assertIsInstance(service, AuthenticationService)
        self.assertIs(service.users_repository, users_repository)
        self.assertIs(service._authentik_admin_client, admin_client)


if __name__ == "__main__":
    unittest.main()
