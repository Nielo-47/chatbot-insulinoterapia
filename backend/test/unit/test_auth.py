import unittest
import uuid
from typing import Any, Optional

from backend.src.application.contracts.repositories import ProfilesRepositoryLike
from backend.src.application.features.auth.auth_service import (
    AuthenticationService,
    build_authentication_service,
)


class ProfilesRepositoryStub(ProfilesRepositoryLike):
    def __init__(self) -> None:
        self.records: dict[uuid.UUID, str] = {}  # user_id -> username
        self.created: list[uuid.UUID] = []
        self.deleted_user_id: Optional[uuid.UUID] = None

    def get_profile_by_id(self, user_id: uuid.UUID) -> Optional[Any]:
        username = self.records.get(user_id)
        if username is None:
            return None
        return SimpleProfile(user_id, username)

    def get_or_create_profile(self, user_id: uuid.UUID, username: str) -> tuple[uuid.UUID, bool]:
        if user_id in self.records:
            return user_id, False
        self.records[user_id] = username
        self.created.append(user_id)
        return user_id, True

    def delete_profile(self, user_id: uuid.UUID) -> bool:
        if user_id not in self.records:
            return False
        self.records.pop(user_id)
        self.deleted_user_id = user_id
        return True


class SimpleProfile:
    def __init__(self, user_id: uuid.UUID, username: str) -> None:
        self.user_id = user_id
        self.username = username


class AccountDeletionClientStub:
    def __init__(self) -> None:
        self.deleted_user_id: Optional[uuid.UUID] = None
        self.deleted_access_token: Optional[str] = None
        self.result = True

    def delete_user(self, user_id: uuid.UUID, access_token: str) -> bool:
        self.deleted_user_id = user_id
        self.deleted_access_token = access_token
        return self.result


def _build_service() -> tuple[AuthenticationService, ProfilesRepositoryStub, AccountDeletionClientStub]:
    profiles_repository = ProfilesRepositoryStub()
    deletion_client = AccountDeletionClientStub()
    service = AuthenticationService(
        profiles_repository=profiles_repository,
        account_deletion_client=deletion_client,
    )
    return service, profiles_repository, deletion_client


class AuthenticationServiceTests(unittest.TestCase):
    def test_resolve_principal_provisions_profile_on_first_login(self) -> None:
        service, profiles_repository, _ = _build_service()
        user_id = uuid.uuid4()

        principal = service.resolve_principal_from_identity(str(user_id), "alice")

        self.assertEqual(principal.id, user_id)
        self.assertEqual(principal.username, "alice")
        self.assertEqual(profiles_repository.created, [user_id])

    def test_resolve_principal_reuses_existing_profile(self) -> None:
        service, profiles_repository, _ = _build_service()
        user_id = uuid.uuid4()
        first = service.resolve_principal_from_identity(str(user_id), "alice")
        second = service.resolve_principal_from_identity(str(user_id), "alice")

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(profiles_repository.created), 1)

    def test_resolve_principal_rejects_invalid_uuid(self) -> None:
        service, _, _ = _build_service()

        with self.assertRaises(ValueError):
            service.resolve_principal_from_identity("not-a-uuid", "alice")

    def test_delete_supabase_user_forwards_token_and_deletes_profile(self) -> None:
        service, profiles_repository, deletion_client = _build_service()
        user_id = uuid.uuid4()
        service.resolve_principal_from_identity(str(user_id), "alice")

        deleted = service.delete_supabase_user(user_id, "access-token-123")

        self.assertTrue(deleted)
        self.assertEqual(deletion_client.deleted_user_id, user_id)
        self.assertEqual(deletion_client.deleted_access_token, "access-token-123")
        self.assertEqual(profiles_repository.deleted_user_id, user_id)

    def test_delete_supabase_user_fails_closed_when_revocation_fails(self) -> None:
        service, profiles_repository, deletion_client = _build_service()
        deletion_client.result = False
        user_id = uuid.uuid4()

        deleted = service.delete_supabase_user(user_id, "access-token-123")

        self.assertFalse(deleted)
        self.assertIsNone(profiles_repository.deleted_user_id)


class BuildAuthenticationServiceTests(unittest.TestCase):
    def test_factory_wires_repository_and_deletion_client(self) -> None:
        profiles_repository = ProfilesRepositoryStub()
        deletion_client = AccountDeletionClientStub()

        service = build_authentication_service(
            profiles_repository=profiles_repository,
            account_deletion_client=deletion_client,
        )

        self.assertIsInstance(service, AuthenticationService)
        self.assertIs(service.profiles_repository, profiles_repository)
        self.assertIs(service._account_deletion_client, deletion_client)


if __name__ == "__main__":
    unittest.main()
