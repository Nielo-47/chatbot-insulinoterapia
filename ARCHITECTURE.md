# Diabetes Chatbot - React + FastAPI Architecture

## Overview

The application is a React SPA frontend and a FastAPI backend. Authentication and
the SQL database are cloud-hosted on **Supabase** (Supabase Auth with email +
password, and Supabase Postgres); the backend stays the sole data-access proxy.
The whole stack runs on a single docker-compose host, with **ngrok** pinned to a
free static dev domain so the app is reachable over HTTPS with a stable URL.

```text
Browser
  |
  | HTTPS :443 / HTTP :3000 (redirects to 443)
  v
ngrok (static dev domain, e.g. https://<domain>.ngrok-free.app)
  |
  v
UI Container (React + Vite build served by Nginx + TLS)
  |  supabase-js -> Supabase Auth (hosted; PKCE, email+password)
  |  Bearer JWT
  |  /api/* (proxied to backend)
  v
Backend Container (FastAPI + RAG)
  |
  +--> Supabase Postgres (pooled DATABASE_URL, port 6543)
  +--> Redis (conversation cache + query rate limiting)
  +--> OpenRouter (LLM + embeddings via RAG runtime)
```

## Services (docker-compose)

### 1. backend
- Technology: FastAPI + uvicorn
- Port: 8000 (internal Docker network only)
- Dockerfile: dockerfile.backend
- Responsibilities:
  - Query processing with RAG (LightRAG runtime; LLM + embeddings via OpenRouter)
  - Persistent conversation storage in Supabase Postgres (SQLAlchemy + psycopg)
  - Conversation read cache + per-user query rate limiting (Redis)
  - Verifies Supabase JWTs (RS256 against the public JWKS endpoint,
    `SUPABASE_JWKS_URL`) for every authenticated route
  - Account deletion via the `delete-account` Supabase edge function (built
    with `@supabase/server`; the secret key lives in Supabase, never in this
    stack)
- Depends on: redis (healthy)

### 2. ui
- Technology: React + TypeScript + TailwindCSS (built with Vite)
- Runtime: Nginx static file serving + TLS termination
- Ports: 3000 (HTTP, redirects to HTTPS) and 443 (HTTPS)
- Dockerfile: dockerfile.ui
- Responsibilities:
  - Authenticate via supabase-js directly with Supabase Auth (PKCE)
  - Send the JWT access token to the backend as a Bearer header
  - Chat interface, session persistence, conversation reset flow
- Depends on: backend (healthy)

### 3. ngrok
- Technology: `ngrok/ngrok:alpine`
- Exposes `https://ui:443` at `NGROK_STATIC_DOMAIN` (a free static dev domain,
  so the URL never rotates).
- This stable URL is what Supabase email-confirmation redirects and the CSP
  `connect-src` are allowlisted against, and it scopes the frontend's
  localStorage session.

### 4. redis
- Conversation message cache (`CHAT_CACHE_REDIS_URL`, TTL `CHAT_CACHE_TTL_SECONDS`)
  and query rate-limiting counters. Redis is a hard dependency: the rate limiter
  fails closed when Redis is unavailable.

## Backend API Contract

All routes except `/`, `/health` (and the nginx-exposed `/api/health`) require a
valid Supabase access token as `Authorization: Bearer <JWT>`. Unauthenticated
requests receive `401` with `WWW-Authenticate: Bearer`. Invalid/expired/anonymous
tokens, and tokens signed with any key other than the project's current signing
keys, are all rejected.

### GET /auth/me
Response:
```json
{
  "id": "2f0b9f9d-2a55-4d85-8c1f-1b2a3c4d5e6f",
  "username": "alice@example.com"
}
```

### DELETE /auth/me
Purges cached PII (Redis conversation cache + LangGraph checkpointer thread)
first, then revokes the Supabase Auth user via the `delete-account` edge
function (the caller's access token is forwarded; the secret key lives only in
Supabase). The local `profiles`/`conversations`/`messages` rows are
removed by the `on_auth_user_deleted` trigger. Fails closed (502) if the
revocation fails.

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

## Data Model (Supabase Postgres)

- `profiles` — `user_id uuid PK` (equals the JWT `sub`), `username text`.
- `conversations` — `id uuid PK`, `user_id uuid` UNIQUE FK to `profiles` ON DELETE CASCADE.
- `messages` — `id uuid PK`, `conversation_id uuid` FK CASCADE, `role`, `content`,
  `sources_json`, indexed for conversation+time ordering.

Passwords are never stored locally; they belong to Supabase Auth. The schema is
applied once via `scripts/supabase_migration.sql` in the Supabase SQL editor
(code-first); the backend only verifies the connection at startup
(`initialize_database()`) and never runs DDL.

## Frontend Structure

```text
frontend/
  src/
    app/
      App.tsx                 # bootstraps supabase session, gates on AuthStatus
    features/
      auth/
        SignInPage.tsx        # email+password form -> supabase.auth.signInWithPassword()
      chat/
        ChatPage.tsx
        components/
          Composer.tsx
          MessageBubble.tsx
          SourceDrawer.tsx
    hooks/
      useDebounce.ts
    lib/
      api.ts                  # Bearer header, 401 -> refreshSession() retry
      env.ts                  # env.supabaseUrl / env.supabaseAnonKey
      storage.ts
      supabase.ts             # single supabase-js client (PKCE + localStorage)
    types/
      app.ts
      chat.ts
    index.css
    main.tsx
```

## Environment Variables (root .env)

- `DATABASE_URL` — Supabase pooled Postgres URL
  (`postgresql+psycopg://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require`).
- `SUPABASE_URL` / `SUPABASE_JWKS_URL` — required (public JWKS endpoint the
  backend uses to verify access tokens with RS256).
- `VITE_SUPABASE_URL` / `VITE_SUPABASE_PUBLISHABLE_KEY` — baked into the UI build and
  used at container start to extend the CSP `connect-src`.
- `NGROK_AUTHTOKEN` / `NGROK_STATIC_DOMAIN` — required (ngrok boots the tunnel).
- `FRONTEND_ORIGINS` — required; powers CORS (local origins + the ngrok domain).
-   `OPENROUTER_*`, `CHAT_CACHE_*`, `DB_POOL_SIZE`/`DB_MAX_OVERFLOW`, `QUERY_RATE_*`,
  `SEMANTIC_CACHE_ENABLED`, `UI_HTTP_PORT`/`UI_HTTPS_PORT` — see `.env.example`
  for the full list with defaults.

## Running

```bash
# 1. Apply scripts/supabase_migration.sql in the Supabase SQL editor.
# 2. Deploy the delete-account edge function and set its secret:
#    supabase functions deploy delete-account
#    supabase secrets set SUPABASE_SECRET_KEY=<your secret key>
# 3. Fill .env and start the stack (ngrok is boot-blocking).
docker compose up --build
```

- UI: `https://<NGROK_STATIC_DOMAIN>` (and `https://localhost` with self-signed
  cert unless `./certs/server.{crt,key}` are mounted).
- API docs (`/docs`, `/redoc`, `/openapi.json`) are always disabled.

### Tests

```bash
scripts/run_backend_unit_tests.sh        # in-memory fakes, no services needed
scripts/run_backend_integration_tests.sh # spins up a throwaway local postgres container
scripts/run_backend_tests.sh             # both
```

Both runners source the repo `.env` and fall back to a test-only
`SUPABASE_JWKS_URL` so the suite runs without real Supabase credentials.

## Security Notes

- **Authentication is delegated to Supabase Auth (email + password, PKCE).** The
  frontend signs in via supabase-js; the backend verifies the presented JWT's
  RS256 signature against the project's public JWKS endpoint, checks
  `aud="authenticated"` and `role="authenticated"`, and maps its `sub` (UUID) to
  the local `profiles` row. No passwords ever reach the backend.
- **Session persistence:** the supabase-js session (with refresh token) lives in
  browser localStorage and is scoped to the origin (the ngrok static domain), so
  logins survive reloads.
- **The backend stores no credentials.** The `profiles` table has no password
  column; login throttling, lockout and token lifecycle are owned by Supabase Auth.
- **Account deletion is fail-closed:** cached PII (Redis conversation cache,
  checkpointer thread state) is purged before the Auth user is revoked via the
  `delete-account` edge function (@supabase/server, which verifies the caller
  with the forwarded access token and holds the secret key only in Supabase); if
  the revocation fails the account is kept (502). The `on_auth_user_deleted`
  trigger on `auth.users` is a safety net for deletions that happen outside
  the API.
- **CORS** uses explicit origins through `FRONTEND_ORIGINS`; the API docs are
  disabled by default; the root banner omits the version; the CSP `connect-src`
  is scoped to `'self'` plus the Supabase origin.
- Conversation history is persisted in Supabase Postgres and cached in Redis.
