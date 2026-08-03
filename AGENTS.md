# AGENTS.md — Project Memory

## Context
Diabetes chatbot: FastAPI backend (`backend/`), React frontend (`frontend/`), docker-compose deployment.
Security hardening of the backend auth/rate-limiting/DB was implemented on `main` (see `git log` around 956f335..HEAD).
The original external security report PDF (`relatorio_chatbot_insulinoterapia.pdf`) is no longer present on disk; these notes are the reconstructed + code-verified record.

## Verified unsolved security issues (DO NOT forget) — ranked by criticality

### MEDIUM
1. **`/docs` + `/redoc` exposed** — `backend/src/api/api.py` FastAPI app uses default `docs_url`/`redoc_url`, publicly reachable. Reveals full API schema for reconnaissance. Should be disabled or gated.

2. **`DELETE /auth/me` cleanup is FK-cascade only** — `backend/src/infrastructure/repositories/users_repository.py` `delete_user_by_id` relies on `ondelete="CASCADE"` in `models.py`; Redis caches (conversation message cache, semantic cache) for the deleted user are NOT invalidated → stale cached PII may remain in Redis.

### LOW
3. **Permissive CORS** — `backend/src/api/api.py` CORSMiddleware: `allow_methods=["*"]`, `allow_headers=["*"]` with `allow_credentials=True`. UI is same-origin via the nginx `/api/` proxy, so CORS is not exercised in the normal deployment; remove or tightly scope it.

4. **Bootstrap password in process args** — `backend/scripts/bootstrap_user.py` accepts `--password` on the command line (visible via `ps`/shell history). Requires local access to exploit; should require interactive `getpass` input or an env var.

## Solved (for reference)
- Real client IP via `X-Forwarded-For` + Redis-backed slowapi limiter (`api.py`, `_client_ip`).
- Uniform generic 401 on login (no 423/429 enumeration); lockout keyed on (username, IP).
- Fail-closed Redis behavior (rate limit, lockout, query throttle, token blacklist).
- Per-user `/query` rate limit (default 30/min, `QUERY_RATE_LIMIT`/`QUERY_RATE_WINDOW_SECONDS`).
- Strong JWT secret enforced at startup (no DEV bypass).
- Input caps: `QueryRequest.query <= 2000`, `session_id <= 64`, `username <= 64`, `password <= 128`.
- DB hardening: compose postgres/backend ports no longer published; required `POSTGRES_DB/USER/PASSWORD`, `DATABASE_URL`, `JWT_SECRET_KEY` (no `chatbot` defaults).
- **JWT moved out of localStorage into an httpOnly cookie** — `frontend/src/lib/auth.ts` deleted; `api.py` sets `access_token` cookie (`HttpOnly`, `Secure`, `SameSite=lax`, `max_age` = JWT expiry) on login, reads cookie-or-Bearer in `get_current_user` (`_request_token`), clears it on logout/delete. `frontend/src/lib/api.ts` uses `credentials: 'include'` and no Authorization header; logout calls POST `/auth/logout`. Config: `AUTH_COOKIE_*` in `config/security.py` (`AUTH_COOKIE_SECURE=false` only for plain-HTTP dev). CSRF is mitigated by SameSite=Lax + CORS origin allowlist. Bearer still works for API clients/tests.
- **UI served over TLS with security headers** — `dockerfile.ui` nginx now: HTTP :80 → 301 → HTTPS :443; self-signed cert generated at container start by `/docker-entrypoint.d/10-security-config.sh` (renders `/etc/nginx/ui.conf.tmpl` via envsubst); real certs mountable at `/etc/nginx/certs/server.{crt,key}` (enables HSTS; HSTS stays OFF for self-signed to avoid browser lockout). Headers: CSP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy`, `Permissions-Policy`. Compose: `UI_HTTP_PORT` (default 3000→80), `UI_HTTPS_PORT` (default 443→443), `./certs` volume (gitignored).

## Test commands
- Unit: `backend/.venv/bin/python -m pytest backend/test/unit` (needs `.env` loaded: use python-dotenv, not `source .env` — values contain shell special chars).
- Integration: `scripts/run_backend_integration_tests.sh` (spins up loopback-only temp postgres). Needs `.env` loaded first (python-dotenv wrapper) AND `CHAT_CACHE_REDIS_URL=redis://localhost:6379/<db>` (the `.env` value `redis://redis:6379/1` only resolves inside compose) AND `POSTGRES_PORT` that is free (5432 is occupied by the deployed stack — use e.g. 55432). `nest_asyncio` and `psycopg_binary` were installed into `backend/.venv` to make this possible.
- Known pre-existing failures (NOT caused by security work):
  - `test_query_processor.py::test_query_without_refinement` (unit).
  - Integration drift: `test_api_endpoints.py` DummyChatbot returns string `sources` vs the structured `SourceItem` schema (400) in `test_authenticated_query_endpoint_returns_payload`, `test_query_endpoint_uses_provided_session_id`, `test_get_conversations_returns_message_list`; English-vs-Portuguese message expectations in `test_clear_session_endpoint_clears_current_user`, `test_delete_me_endpoint_deletes_current_user`.
  - `test_conversation_cache.py::test_cache_round_trip_and_invalidate` (`get_messages` normalizes and adds `sources: []`; assertion expects exact stored shape).
