import unittest

from backend.src.application.features.auth.auth_service import (
    AuthenticationService,
    build_authentication_service,
)


class AccountDeletionClientStub:
    def __init__(self) -> None:
        self.deleted_user_ids: list[str] = []
        self.result = True

    def delete_user(self, user_id: str) -> bool:
        self.deleted_user_ids.append(user_id)
        return self.result


def _build_service(
    deletion_result: bool = True,
) -> tuple[AuthenticationService, AccountDeletionClientStub]:
    deletion_client = AccountDeletionClientStub()
    deletion_client.result = deletion_result
    service = AuthenticationService(account_deletion_client=deletion_client)
    return service, deletion_client


class AuthenticationServiceTests(unittest.TestCase):
    def test_resolve_principal_maps_token_subject(self) -> None:
        service, _ = _build_service()

        principal = service.resolve_principal_from_identity("abc123def456ghi", "alice@example.com")

        self.assertEqual(principal.id, "abc123def456ghi")
        self.assertEqual(principal.username, "alice@example.com")

    def test_resolve_principal_rejects_empty_id(self) -> None:
        service, _ = _build_service()

        with self.assertRaises(ValueError):
            service.resolve_principal_from_identity("", "alice@example.com")

    def test_delete_account_deletes_the_user_record(self) -> None:
        service, deletion_client = _build_service()

        deleted = service.delete_account("abc123def456ghi")

        self.assertTrue(deleted)
        self.assertEqual(deletion_client.deleted_user_ids, ["abc123def456ghi"])

    def test_delete_account_fails_closed_when_revocation_fails(self) -> None:
        service, _ = _build_service(deletion_result=False)

        deleted = service.delete_account("abc123def456ghi")

        self.assertFalse(deleted)

    def test_resolve_username_falls_back_without_resolver(self) -> None:
        service, _ = _build_service()

        self.assertEqual(service.resolve_username("abcdefgh1234567"), "user_abcdefgh")

    def test_resolve_username_uses_injected_resolver(self) -> None:
        deletion_client = AccountDeletionClientStub()
        service = AuthenticationService(
            account_deletion_client=deletion_client,
            username_resolver=lambda user_id: f"{user_id}@example.com",
        )

        self.assertEqual(service.resolve_username("abc123"), "abc123@example.com")


class BuildAuthenticationServiceTests(unittest.TestCase):
    def test_factory_wires_deletion_client_and_resolver(self) -> None:
        deletion_client = AccountDeletionClientStub()
        resolver = lambda user_id: user_id  # noqa: E731 - trivial test double

        service = build_authentication_service(
            account_deletion_client=deletion_client,
            username_resolver=resolver,
        )

        self.assertIsInstance(service, AuthenticationService)
        self.assertIs(service._account_deletion_client, deletion_client)
        self.assertEqual(service.resolve_username("xyz"), "xyz")


if __name__ == "__main__":
    unittest.main()
