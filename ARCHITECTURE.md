# Diabetes Chatbot - React + FastAPI Architecture

## Overview

The application is a React SPA frontend and a FastAPI backend. Authentication and
the app database are owned by the self-hosted **PocketBase** container (email +
password auth with HS256 tokens, and SQLite collections for conversations and
messages); the backend stays the sole data-access proxy. The whole stack runs on
a single docker-compose host, with **ngrok** pinned to a free static dev domain
so the app is reachable over HTTPS with a stable URL.

```text
Browser
  |
  | HTTPS :443 / HTTP :3000 (redirects to 443)
  v
ngrok (static dev domain, e.g. https://<domain>.ngrok-free.app)
  |
  v
UI Container (React + Vite build served by Nginx + TLS)
  |  pocketbase-js -> /pb/* (nginx proxy) -> PocketBase Auth
  |  Bearer JWT
  |  /api/* (proxied to backend)
  v
Backend Container (FastAPI + RAG)
  |
  +--> PocketBase (superuser API: conversations/messages data,
  |    username lookups, account deletion)
  +--> Redis (conversation cache + query rate limiting)
  +--> OpenRouter (LLM + embeddings via RAG runtime)
```

## Services (docker-compose)

### 1. pocketbase
- Technology: official PocketBase binary (single Go executable + embedded
  SQLite) on Alpine; image built from `dockerfile.pocketbase` (version pinned
  via the `PB_VERSION` build arg).
- Port: 8090 (internal Docker network only)
- Volumes:
  - `./data/pocketbase:/pb/pb_data` — SQLite files + uploaded state
  - `./backend/data/pb_migrations:/pb/pb_migrations` — JSVM migrations (schema creation +
    one-shot Supabase import), applied automatically at boot
- Responsibilities:
  - Email + password authentication (`users` auth collection; email
    confirmation disabled — accounts are active immediately after signup)
  - App data storage (`conversations`, `messages` collections)
  - Issues the access tokens AND validates them: a JSVM-migration route
    (`/api/linachat/token-introspect`) checks the Bearer token presented by
    the backend (PocketBase signs tokens with a per-record key component, so
    verification must happen inside PocketBase)
- API rules on app collections are nil (superuser-only): all data access goes
  through the backend's superuser client, which enforces user scoping.

### 2. backend
- Technology: FastAPI + uvicorn
- Port: 8000 (internal Docker network only)
- Dockerfile: dockerfile.backend
- Responsibilities:
  - Query processing with RAG (LightRAG runtime; LLM + embeddings via OpenRouter)
  - Persistent conversation storage in PocketBase collections via a
    stdlib-urllib superuser client (`backend/src/infrastructure/pocketbase/`)
  - Conversation read cache + per-user query rate limiting (Redis)
  - Validates access tokens by forwarding them to PocketBase's
    `/api/linachat/token-introspect` route (one fast in-network request per
    authenticated call; PocketBase enforces expiry/revocation itself)
  - Account deletion by removing the `users` record through the superuser API
    (CascadeDelete relations purge conversations and messages); fails closed
- Depends on: pocketbase (healthy), redis (healthy)

### 3. ui
- Technology: React + TypeScript + TailwindCSS (built with Vite)
- Runtime: Nginx static file serving + TLS termination
- Ports: 3000 (HTTP, redirects to HTTPS) and 443 (HTTPS)
- Dockerfile: dockerfile.ui
- Responsibilities:
  - Authenticate via pocketbase-js against the same-origin `/pb` location,
    which nginx proxies to the pocketbase container
  - Send the JWT access token to the backend as a Bearer header
  - Chat interface, session persistence, conversation reset flow
- Depends on: backend (healthy)

### 4. ngrok
- Technology: `ngrok/ngrok:alpine`
- Exposes `https://ui:443` at `NGROK_STATIC_DOMAIN` (a free static dev domain,
  so the URL never rotates).
- The stable URL scopes the frontend's localStorage session.

### 5. redis
- Conversation message cache (`CHAT_CACHE_REDIS_URL`, TTL `CHAT_CACHE_TTL_SECONDS`)
  and query rate-limiting counters. Redis is a hard dependency: the rate limiter
  fails closed when Redis is unavailable.

## Backend API Contract

All routes except `/`, `/health` (and the nginx-exposed `/api/health`) require a
valid PocketBase auth token as `Authorization: Bearer <JWT>`. Unauthenticated
requests receive `401` with `WWW-Authenticate: Bearer`. Invalid/expired tokens,
and tokens signed with any key other than the users collection's current secret,
are rejected.

### GET /auth/me
Response:
```json
{
  "id": "user0000alice0001",
  "username": "alice@example.com"
}
```
The token carries only the record id, so the username is resolved with a
short-TTL cached directory lookup (falling back to `user_<id prefix>` when the
record no longer exists or has no username).

### DELETE /auth/me
Purges cached PII first (Redis conversation cache; the LangGraph checkpointer
is currently disabled), then deletes the `users` record via the superuser API.
PocketBase's CascadeDelete relation removes the user's `conversations` row,
which cascades to its `messages`. Fails closed (502) if the deletion fails.

### POST /query
Request:
```json
{
  "query": "Como aplicar insulina?",
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "response": "Para aplicar insulina...",
  "sources": [
    {"path": "insulina.md", "page": 12, "excerpt": "Aplicar a insulina..."}
  ],
  "summarized": false,
  "session_id": "session-id"
}
```
Throttled per user (`QUERY_RATE_LIMIT` per `QUERY_RATE_WINDOW_SECONDS`) with
`429`.

### GET /user/conversations
Response:
```json
{
  "messages": [
    {"role": "user", "content": "Oi", "sources": []},
    {"role": "assistant", "content": "Olá", "sources": []}
  ]
}
```

### DELETE /user/conversations
Clears the current user's conversation (cache invalidated first).

### GET /health (and /api/health via nginx)
```json
{
  "status": "healthy",
  "message": "Chatbot API is running"
}
```

## Data Model (PocketBase collections)

Migrations live in `backend/data/pb_migrations/` and are applied automatically on container
boot:

- `users` (auth collection) — email/password identity plus a legacy `username`
  text field (merged in from the old Supabase `profiles` table during import).
- `conversations` — `user` relation to `users` (UNIQUE, CASCADE delete),
  `summary` text, autodate created/updated.
- `messages` — `conversation` relation to `conversations` (CASCADE delete),
  `role`, `content`, `sources_json`, indexed for conversation+time ordering.

Record ids are PocketBase-generated strings (max 15 chars). Passwords are
bcrypt-hashed inside PocketBase; the backend never sees them. Existing Supabase
data was imported once with `backend/data/pb_migrations/1756000001_import_supabase_export.js`
from JSON exports produced by `scripts/export_supabase_data.sql`; verify with
`scripts/verify_pocketbase_migration.py`. New users sign up through
pocketbase-js (`POST /api/collections/users/records`).

## Frontend Structure

```text
frontend/
  src/
    app/
      App.tsx                 # subscribes to pocketbase authStore, gates on AuthStatus
    features/
      auth/
        SignInPage.tsx        # email+password form -> pb.collection('users').authWithPassword()
      chat/
        ChatPage.tsx
        components/
          Composer.tsx
          MessageBubble.tsx
          SourceDrawer.tsx
    hooks/
      useDebounce.ts
    lib/
      api.ts                  # sync token read from authStore, 401 -> throw
      env.ts                  # env.pocketbaseUrl (default /pb)
      pocketbase.ts           # single pocketbase-js client (auto-cancellation off)
      storage.ts
    types/
      app.ts
      chat.ts
    index.css
    main.tsx
```

## Environment Variables (root .env)

- `POCKETBASE_URL` — internal PocketBase base URL (`http://pocketbase:8090`).
- `POCKETBASE_SUPERUSER_EMAIL` / `POCKETBASE_SUPERUSER_PASSWORD` — superuser
  credentials (created at first boot by the pocketbase entrypoint); used by the
  backend for all data access and account deletion.
- `VITE_POCKETBASE_URL` — baked into the UI build (default `/pb`; nginx proxies
  same-origin `/pb/` to PocketBase, so no cross-origin CSP entries are needed).
- `NGROK_AUTHTOKEN` / `NGROK_STATIC_DOMAIN` — required (ngrok boots the tunnel).
- `FRONTEND_ORIGINS` — required; powers CORS (local origins + the ngrok domain).
- `OPENROUTER_*`, `CHAT_CACHE_*`, `QUERY_RATE_*`,
  `SEMANTIC_CACHE_ENABLED`, `UI_HTTP_PORT`/`UI_HTTPS_PORT` — see `.env.example`
  for the full list with defaults.

## Running

```bash
# 1. Fill .env (PocketBase superuser credentials are created automatically on
#    first boot).
# 2. Start the stack (ngrok is boot-blocking):
docker compose up --build
```

- UI: `https://<NGROK_STATIC_DOMAIN>` (and `https://localhost` with self-signed
  cert unless `./certs/server.{crt,key}` are mounted).
- PocketBase Admin UI: reachable only inside the compose network; expose it
  temporarily with e.g.
  `docker compose exec pocketbase /pb/pocketbase superuser upsert ...`.
- API docs (`/docs`, `/redoc`, `/openapi.json`) are always disabled.

### Tests

```bash
scripts/run_backend_unit_tests.sh        # in-memory fakes, no services needed
scripts/run_backend_integration_tests.sh # FakePocketBaseClient, no containers needed
scripts/run_backend_tests.sh             # both
```

Both runners source the repo `.env` and fall back to test-only PocketBase
values so the suite runs without real services or credentials.

## Security Notes

- **Authentication is delegated to the self-hosted PocketBase (email +
  password).** The frontend signs in via pocketbase-js; the backend validates
  the presented token by forwarding it to PocketBase's introspection route,
  so expiry, revocation and signing are enforced by PocketBase itself. No
  passwords ever reach the backend.
- **Session persistence:** the pocketbase-js auth store lives in browser
  localStorage and is scoped to the origin (the ngrok static domain), so logins
  survive reloads. Tokens expire per the users collection's token duration.
- **The backend stores no credentials.** Login throttling, lockout and token
  lifecycle are owned by PocketBase.
- **Data access is superuser-only.** App collections have nil API rules, so
  every read/write goes through the backend's superuser client, which scopes
  queries by the authenticated user id; there is no direct client-to-data path
  other than auth itself.
- **Account deletion is fail-closed:** cached PII (Redis conversation cache) is
  purged before the `users` record is deleted via the superuser API; if the
  deletion fails the account is kept (502). PocketBase CascadeDelete relations
  guarantee conversations/messages removal even for deletions made directly in
  the Admin UI.
- **CORS** uses explicit origins through `FRONTEND_ORIGINS`; the API docs are
  disabled by default; the root banner omits the version; the CSP `connect-src`
  is `'self'` (PocketBase traffic is same-origin through the nginx `/pb` proxy).
- Conversation history is persisted in PocketBase (SQLite volume under
  `data/pocketbase/`) and cached in Redis.
