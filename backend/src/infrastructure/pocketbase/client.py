"""Minimal stdlib PocketBase API client (no extra runtime dependency).

A single superuser session is shared by every consumer (repositories, account
deletion, username directory). The superuser token is fetched lazily on the
first call and transparently re-authenticated once when it expires (PocketBase
answers 401). A lock guards concurrent refreshes because FastAPI runs sync
endpoints in a thread pool.
"""

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from backend.src.config.pocketbase import (
    POCKETBASE_SUPERUSER_EMAIL,
    POCKETBASE_SUPERUSER_PASSWORD,
    POCKETBASE_TIMEOUT_SECONDS,
    POCKETBASE_URL,
)

logger = logging.getLogger(__name__)


class PocketBaseError(RuntimeError):
    """Raised when a PocketBase API call fails (non-2xx or malformed body)."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class PocketBaseClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or POCKETBASE_URL).rstrip("/")
        self._email = email or POCKETBASE_SUPERUSER_EMAIL
        self._password = password or POCKETBASE_SUPERUSER_PASSWORD
        self._timeout = timeout or POCKETBASE_TIMEOUT_SECONDS
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        allow_reauth: bool = True,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", token)

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = response.read()
                return json.loads(payload.decode("utf-8")) if payload else None
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
                detail = str(error_body.get("message", ""))[:300]
            except Exception:  # noqa: BLE001 - best-effort detail extraction only
                pass
            # An expired cached superuser token surfaces as 401: authenticate
            # once and replay the original request.
            if exc.code == 401 and token and token == self._token and allow_reauth:
                logger.info("PocketBase superuser token rejected; re-authenticating once")
                self.authenticate(force=True)
                return self._request(
                    method, path, body=body, token=self._token, allow_reauth=False
                )
            raise PocketBaseError(
                f"PocketBase {method} {path} failed with HTTP {exc.code}: {detail}",
                status=exc.code,
            ) from exc
        except OSError as exc:
            raise PocketBaseError(f"PocketBase unreachable ({method} {path}): {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PocketBaseError(f"PocketBase returned invalid JSON for {method} {path}") from exc

    def authenticate(self, force: bool = False) -> str:
        """Return a valid superuser auth token, authenticating if needed."""
        with self._lock:
            if self._token and not force:
                return self._token
            token = self._request(
                "POST",
                "/api/collections/_superusers/auth-with-password",
                body={"identity": self._email, "password": self._password},
                allow_reauth=False,
            )
            if not isinstance(token, dict) or not token.get("token"):
                raise PocketBaseError("PocketBase superuser authentication returned no token")
            self._token = str(token["token"])
            logger.info("Authenticated against PocketBase as superuser")
            return self._token

    def _admin_request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        return self._request(method, path, body=body, token=self.authenticate())

    def health(self) -> bool:
        try:
            self._request("GET", "/api/health")
            return True
        except PocketBaseError as exc:
            logger.warning("PocketBase health check failed: %s", exc)
            return False

    def create_record(self, collection: str, body: Dict[str, Any]) -> Dict[str, Any]:
        result = self._admin_request("POST", f"/api/collections/{collection}/records", body=body)
        return result if isinstance(result, dict) else {}

    def get_record(self, collection: str, record_id: str) -> Dict[str, Any]:
        result = self._admin_request("GET", f"/api/collections/{collection}/records/{record_id}")
        return result if isinstance(result, dict) else {}

    def update_record(self, collection: str, record_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        result = self._admin_request(
            "PATCH", f"/api/collections/{collection}/records/{record_id}", body=body
        )
        return result if isinstance(result, dict) else {}

    def delete_record(self, collection: str, record_id: str) -> bool:
        self._admin_request("DELETE", f"/api/collections/{collection}/records/{record_id}")
        return True

    def count_records(self, collection: str, filter_expr: str = "") -> int:
        """Return the total number of matching records via a single request."""
        query = "?perPage=1"
        if filter_expr:
            query += f"&filter={urllib.request.quote(filter_expr, safe='')}"
        page = self._admin_request("GET", f"/api/collections/{collection}/records{query}")
        if isinstance(page, dict):
            try:
                return int(page.get("totalItems", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def list_records(
        self,
        collection: str,
        filter_expr: str = "",
        sort: str = "",
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch ALL matching records, paging through the list API."""
        records: List[Dict[str, Any]] = []
        offset = 0
        while True:
            query = f"?perPage={batch_size}&skipTotal=1&offset={offset}"
            if filter_expr:
                query += f"&filter={urllib.request.quote(filter_expr, safe='')}"
            if sort:
                query += f"&sort={urllib.request.quote(sort, safe='')}"
            page = self._admin_request("GET", f"/api/collections/{collection}/records{query}")
            if not isinstance(page, dict):
                break
            items = page.get("items") or []
            records.extend(item for item in items if isinstance(item, dict))
            if len(items) < batch_size:
                break
            offset += batch_size
        return records


_shared_client: Optional[PocketBaseClient] = None
_client_lock = threading.Lock()


def get_pocketbase_client() -> PocketBaseClient:
    """Process-wide shared superuser client."""
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = PocketBaseClient()
        return _shared_client
