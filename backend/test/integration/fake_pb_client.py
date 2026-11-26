"""In-memory stand-in for the PocketBase API client.

Implements the subset of PocketBaseClient used by the repositories (create,
get, update, delete, list with a simple ``field = "value"`` filter and
``-created`` sort, count). Records get sequential string ids and autodate
created/updated fields, mirroring the real service closely enough for
integration tests without any network or container.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_FILTER_RE = re.compile(r'^(\w+)\s*=\s*"([^"]*)"$')

_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

# Mirrors the real schema: unique indexes enforced by PocketBase.
_UNIQUE_FIELDS = {"conversations": ("user",)}

# Mirrors CascadeDelete relations: (child_collection, relation_field).
_CASCADES = {
    "users": [("conversations", "user")],
    "conversations": [("messages", "conversation")],
}

# Mirrors the real schema: unique indexes enforced by PocketBase.
_UNIQUE_FIELDS: Dict[str, tuple] = {"conversations": ("user",)}


class FakePocketBaseClient:
    def __init__(self) -> None:
        self.collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._counters: Dict[str, int] = {}
        self.deleted_ids: List[tuple[str, str]] = []

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #
    def _next_id(self, collection: str) -> str:
        n = self._counters.get(collection, 0) + 1
        self._counters[collection] = n
        # 15-char lowercase-alnum id, zero-padded from the counter.
        return (_ID_ALPHABET * 3)[n % 36] + str(n).zfill(14)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _match(record: Dict[str, Any], filter_expr: str) -> bool:
        if not filter_expr:
            return True
        match = _FILTER_RE.match(filter_expr.strip())
        if not match:
            raise ValueError(f"FakePocketBaseClient cannot parse filter: {filter_expr!r}")
        field, value = match.groups()
        return str(record.get(field, "")) == value

    # ------------------------------------------------------------------ #
    # client surface                                                      #
    # ------------------------------------------------------------------ #
    def health(self) -> bool:
        return True

    def create_record(self, collection: str, body: Dict[str, Any]) -> Dict[str, Any]:
        records = self.collections.setdefault(collection, {})
        record_id = str(body.get("id") or self._next_id(collection))
        if record_id in records:
            from backend.src.infrastructure.pocketbase import PocketBaseError

            raise PocketBaseError("duplicate record", status=400)
        for field in _UNIQUE_FIELDS.get(collection, ()):
            value = body.get(field)
            if value is None:
                continue
            if any(r.get(field) == value for r in records.values()):
                from backend.src.infrastructure.pocketbase import PocketBaseError

                raise PocketBaseError(
                    f"unique constraint failed: {collection}.{field}", status=400
                )
        record: Dict[str, Any] = dict(body)
        record["id"] = record_id
        now = self._now()
        record.setdefault("created", now)
        record["updated"] = now
        records[record_id] = record
        return dict(record)

    def get_record(self, collection: str, record_id: str) -> Dict[str, Any]:
        records = self.collections.setdefault(collection, {})
        if record_id not in records:
            from backend.src.infrastructure.pocketbase import PocketBaseError

            raise PocketBaseError("record not found", status=404)
        return dict(records[record_id])

    def update_record(self, collection: str, record_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        record = self.get_record(collection, record_id)
        record.update(body)
        record["updated"] = self._now()
        self.collections[collection][record_id] = record
        return dict(record)

    def delete_record(self, collection: str, record_id: str) -> bool:
        records = self.collections.setdefault(collection, {})
        if record_id not in records:
            from backend.src.infrastructure.pocketbase import PocketBaseError

            raise PocketBaseError("record not found", status=404)
        del records[record_id]
        self.deleted_ids.append((collection, record_id))
        # Simulate PocketBase CascadeDelete relations.
        for child_collection, field in _CASCADES.get(collection, ()):
            for child in list(
                self.list_records(child_collection, filter_expr=f'{field} = "{record_id}"')
            ):
                self.delete_record(child_collection, str(child["id"]))
        return True

    def count_records(self, collection: str, filter_expr: str = "") -> int:
        records = self.collections.setdefault(collection, {})
        return sum(1 for record in records.values() if self._match(record, filter_expr))

    def list_records(
        self,
        collection: str,
        filter_expr: str = "",
        sort: str = "",
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        records = self.collections.setdefault(collection, {})
        matched = [dict(r) for r in records.values() if self._match(r, filter_expr)]
        if sort:
            reverse = sort.startswith("-")
            key = sort.lstrip("-")
            matched.sort(key=lambda r: str(r.get(key, "")), reverse=reverse)
        return matched


def make_fake_client_factory(client: FakePocketBaseClient):
    """Patch target for modules that call ``get_pocketbase_client``."""
    return lambda: client
