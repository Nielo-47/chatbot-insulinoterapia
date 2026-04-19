"""Create or reuse a backend user for local auth bootstrapping."""

import argparse
import getpass

from dotenv import load_dotenv

from backend.src.application.features.auth import hash_password
from backend.src.infrastructure.data import initialize_database
from backend.src.infrastructure.repositories.users_repository import UsersRepository


load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or reuse a backend user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", help="If omitted, the password is requested interactively")
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password: ")
    if not password:
        raise SystemExit("Password is required")

    initialize_database()
    users = UsersRepository()
    user_id = users.get_or_create_user_id(args.username, hash_password(password))
    print(f"User ready: {args.username} (id={user_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
