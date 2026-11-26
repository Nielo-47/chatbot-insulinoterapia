from typing import Optional

from backend.src.infrastructure.pocketbase import PocketBaseClient, get_pocketbase_client

CONVERSATIONS_COLLECTION = "conversations"


class ConversationsRepository:
    """PocketBase-backed conversations storage.

    Each user has at most one conversation (unique ``user`` relation). The
    unique constraint on the relation field is enforced by PocketBase; a race
    on first creation surfaces as a 400 from the API and is resolved by
    re-reading the existing record.
    """

    def __init__(self, client: Optional[PocketBaseClient] = None):
        self._client = client or get_pocketbase_client()

    @staticmethod
    def _user_filter(user_id: str) -> str:
        return f'user = "{user_id}"'

    def get_conversation_id_by_user(self, user_id: str) -> Optional[str]:
        records = self._client.list_records(
            CONVERSATIONS_COLLECTION, filter_expr=self._user_filter(user_id)
        )
        return str(records[0]["id"]) if records else None

    def _create_for_user(self, user_id: str) -> str:
        record = self._client.create_record(CONVERSATIONS_COLLECTION, {"user": user_id})
        return str(record["id"])

    def get_or_create_conversation_id(self, user_id: str) -> str:
        try:
            return self._create_for_user(user_id)
        except Exception as exc:
            # Unique-relation races and transient errors fall back to a read;
            # if no record exists after all, rethrow so callers fail loudly.
            conversation_id = self.get_conversation_id_by_user(user_id)
            if conversation_id is not None:
                return conversation_id
            raise exc

    def touch_conversation(self, conversation_id: str) -> None:
        # The ``updated`` field is an auto-date managed by PocketBase: any
        # successful update refreshes it, so an empty patch acts as a touch.
        self._client.update_record(CONVERSATIONS_COLLECTION, conversation_id, {})

    def get_summary(self, conversation_id: str) -> Optional[str]:
        try:
            record = self._client.get_record(CONVERSATIONS_COLLECTION, conversation_id)
        except Exception:
            return None
        summary = record.get("summary")
        return summary if isinstance(summary, str) and summary else None

    def update_summary(self, conversation_id: str, summary: str) -> None:
        self._client.update_record(
            CONVERSATIONS_COLLECTION,
            conversation_id,
            {"summary": summary},
        )
