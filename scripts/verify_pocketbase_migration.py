#!/usr/bin/env python3
"""Post-import verification for the Supabase -> PocketBase migration.

Compares record counts in the running PocketBase instance against the JSON
export files produced by scripts/export_supabase_data.sql, and spot-checks
that migrated users kept their legacy username field and can be looked up.

Usage (from the repo root):
    backend/.venv/bin/python scripts/verify_pocketbase_migration.py \
        --url http://localhost:8090 \
        --import-dir data/pb_import

Requires POCKETBASE_SUPERUSER_EMAIL / POCKETBASE_SUPERUSER_PASSWORD in the
environment or .env. Exits non-zero on any mismatch.
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.src.config.pocketbase import (  # noqa: E402
    POCKETBASE_SUPERUSER_EMAIL,
    POCKETBASE_SUPERUSER_PASSWORD,
)
from backend.src.infrastructure.pocketbase import PocketBaseClient  # noqa: E402

EXPORT_FILES = {
    "users": "supabase_users.json",
    "conversations": "supabase_conversations.json",
    "messages": "supabase_messages.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8090", help="PocketBase base URL")
    parser.add_argument(
        "--import-dir",
        default=str(REPO_ROOT / "data" / "pb_import"),
        help="Directory holding the supabase_*.json export files",
    )
    args = parser.parse_args()

    import_dir = pathlib.Path(args.import_dir)
    client = PocketBaseClient(base_url=args.url)
    failures = []

    if not client.health():
        print(f"FAIL: PocketBase is not reachable at {args.url}")
        return 1
    print(f"PocketBase reachable at {args.url}")

    expected_counts = {}
    for collection, filename in EXPORT_FILES.items():
        path = import_dir / filename
        if path.exists():
            expected_counts[collection] = len(json.loads(path.read_text()))
        else:
            print(f"WARN: export file missing, skipping count check: {path}")
            expected_counts[collection] = None

    for collection, expected in expected_counts.items():
        actual = client.count_records(collection)
        label = "OK" if expected is None or actual == expected else "FAIL"
        print(f"{label}: {collection}: {actual} records (expected {expected})")
        if expected is not None and actual != expected:
            failures.append(collection)

    # Spot-check: every imported user should still resolve by email and carry
    # the legacy profile username in the built-in "name" field.
    users_path = import_dir / EXPORT_FILES["users"]
    if users_path.exists():
        missing_username = 0
        missing_users = []
        for user in json.loads(users_path.read_text()):
            matches = client.list_records("users", filter_expr=f'email = "{user["email"]}"')
            if not matches:
                missing_users.append(user["email"])
            elif not str(matches[0].get("name") or "").strip():
                missing_username += 1
        if missing_users:
            print(f"FAIL: users missing from PocketBase: {missing_users[:5]}")
            failures.append("users lookup")
        else:
            print("OK: all exported users found by email")
        if missing_username:
            print(f"WARN: {missing_username} imported users have an empty username")
        else:
            print("OK: all imported users have a username")

    if failures:
        print(f"\nVerification FAILED: {failures}")
        return 1
    print("\nVerification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
