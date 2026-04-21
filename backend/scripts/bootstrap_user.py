"""Create or reuse a backend user for local auth bootstrapping."""

import argparse
import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that depend on them
project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(project_root / ".env")

from backend.src.application.features.auth import hash_password
from backend.src.infrastructure.data import initialize_database
from backend.src.infrastructure.repositories.users_repository import UsersRepository


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
    user_id, created_new = users.get_or_create_user_id(args.username, hash_password(password))
    if created_new:
        print(f"Created new user: {args.username} (id={user_id})")
    else:
        print(f"User already exists: {args.username} (id={user_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
